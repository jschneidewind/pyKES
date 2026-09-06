"""
SQLite schema for the photocatalysis database index.

The index holds one row per *entity*. A photocatalysis experiment is an entity
that happens to carry a payload file; a catalyst batch or a precursor chemical
is an entity that happens not to. Keeping them in one table is what makes the
reference structure generic: nothing in this module knows what a precursor is.

Scientific metadata lives in JSON columns rather than in typed columns, because
the metadata grows — experiments run next year carry fields that do not exist
today, and absorbing them must not require a migration. Only the columns the
application itself depends on are typed.

See docs/database_index.md for the design and docs/photocatalytic_database.md
for how this fits the wider system.
"""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# =============================================================================
# Versioning
# =============================================================================

# Bumped when the table layout changes in a way an existing index cannot simply
# be reopened with. A rebuild from the retained uploads is always the fallback.
INDEX_SCHEMA_VERSION = "1.0"


# =============================================================================
# Vocabulary fixed by the group
# =============================================================================

# The kinds of entry the database holds. Adding a type here is the only code
# change a new kind of entry needs; the reference structure itself is generic.
ENTITY_TYPES = (
    "experiment",
    "catalyst_batch",
    "finished_semiconductor",
    "precursor_semiconductor",
    "precursor_chemical",
    "commercial_chemical",
    "stock_solution",
    "other_entity",
)

# Assigned to entries whose type cannot be determined from the upload.
DEFAULT_ENTITY_TYPE = "other_entity"


# =============================================================================
# Reference resolution
# =============================================================================

# Separates the role names of a reference path from the metadata key it reaches,
# e.g. 'catalyst_batch/finished_semiconductor/Synthesis temperature [°C]'.
#
# Real metadata keys contain slashes too ('Catalyst concentration [g/L]'), so a
# stored key escapes its own slashes to `KEY_SLASH_PLACEHOLDER` first — the same
# convention `save_nested_dict_to_hdf5` already uses to keep HDF5 paths
# splittable. A slash in a stored key is therefore always a separator, and no
# key has to be refused for containing one.
ROLE_PATH_SEPARATOR = "/"

# How far inherited metadata is followed. Chains in practice are three or four
# hops; the cap exists so a malformed graph fails loudly instead of hanging.
MAX_REFERENCE_DEPTH = 8


# =============================================================================
# Name collisions
# =============================================================================

# Appended to an entity id when a name already in the database is uploaded
# again, giving 'EA-555__v2'. Both entries are kept: the group's policy is that
# a collision is resolved by versioning, never by overwriting.
VERSION_SUFFIX_SEPARATOR = "__v"


# =============================================================================
# Table definitions
# =============================================================================

SCHEMA_STATEMENTS = """
CREATE TABLE IF NOT EXISTS uploads (
    id           INTEGER PRIMARY KEY,
    sha256       TEXT    NOT NULL UNIQUE,
    filename     TEXT    NOT NULL,
    stored_path  TEXT    NOT NULL,
    byte_count   INTEGER NOT NULL,
    kind         TEXT    NOT NULL,
    uploaded_by  TEXT    NOT NULL,
    uploaded_at  TEXT    NOT NULL,
    entity_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id        TEXT    PRIMARY KEY,
    base_id          TEXT    NOT NULL,
    version          INTEGER NOT NULL DEFAULT 1,
    entity_type      TEXT    NOT NULL,
    display_group    TEXT,
    color            TEXT,
    active           INTEGER,
    owner            TEXT    NOT NULL,
    created_at       TEXT    NOT NULL,
    updated_at       TEXT    NOT NULL,
    upload_id        INTEGER REFERENCES uploads(id),
    payload_path     TEXT,
    payload_bytes    INTEGER,
    payload_sha256   TEXT,
    pykes_version    TEXT,
    external_app     TEXT,
    external_version TEXT,
    last_processed   TEXT,
    metadata         TEXT    NOT NULL,
    results          TEXT    NOT NULL,
    effective        TEXT    NOT NULL
);

-- One target per (source, role): a repeated role would produce two identical
-- qualified metadata keys and the second would silently overwrite the first,
-- so the primary key makes that impossible rather than merely unlikely.
CREATE TABLE IF NOT EXISTS edges (
    source   TEXT    NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    role     TEXT    NOT NULL,
    target   TEXT    NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (source, role)
);

CREATE TABLE IF NOT EXISTS metadata_keys (
    key             TEXT    PRIMARY KEY,
    leaf_name       TEXT    NOT NULL,
    role_path       TEXT,
    canonical_key   TEXT,
    inferred_type   TEXT    NOT NULL,
    unit            TEXT,
    occurrences     INTEGER NOT NULL DEFAULT 0,
    first_seen      TEXT    NOT NULL,
    last_seen       TEXT    NOT NULL,
    distinct_sample TEXT,
    entity_types    TEXT
);

CREATE TABLE IF NOT EXISTS result_keys (
    label       TEXT    PRIMARY KEY,
    path        TEXT    NOT NULL,
    unit        TEXT,
    value_format TEXT,
    defined_by  INTEGER REFERENCES uploads(id),
    conflicting INTEGER NOT NULL DEFAULT 0,
    first_seen  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entities_type    ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_base    ON entities(base_id);
CREATE INDEX IF NOT EXISTS idx_entities_owner   ON entities(owner);
CREATE INDEX IF NOT EXISTS idx_entities_group   ON entities(display_group);
CREATE INDEX IF NOT EXISTS idx_edges_target     ON edges(target);
CREATE INDEX IF NOT EXISTS idx_metadata_leaf    ON metadata_keys(leaf_name);
"""


# =============================================================================
# Filesystem layout
# =============================================================================

@dataclass
class IndexPaths:
    """
    Locations the index and its two file tiers live in.

    Parameters
    ----------
    root : Path
        Directory holding the index and both file tiers.
    index_path, payload_directory, upload_directory : Path
        Derived from ``root``; supplied explicitly only in tests.
    """

    root: Path
    index_path: Path = field(default=None)
    payload_directory: Path = field(default=None)
    upload_directory: Path = field(default=None)

    def __post_init__(self) -> None:
        self.root = Path(self.root)

        if self.index_path is None:
            self.index_path = self.root / "index.sqlite"
        if self.payload_directory is None:
            self.payload_directory = self.root / "payloads"
        if self.upload_directory is None:
            self.upload_directory = self.root / "uploads"

        self.payload_directory.mkdir(parents=True, exist_ok=True)
        self.upload_directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Connection handling
# =============================================================================

def open_index(paths: IndexPaths) -> sqlite3.Connection:
    """
    Open the index database, creating and initialising it if absent.

    Parameters
    ----------
    paths : IndexPaths
        Filesystem layout of the database.

    Returns
    -------
    connection : sqlite3.Connection
        Connection in WAL mode with foreign keys enforced. WAL is what lets
        searches run while an upload is being ingested.
    """
    connection = sqlite3.connect(paths.index_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")

    initialise_index(connection)

    return connection


def initialise_index(connection: sqlite3.Connection) -> None:
    """
    Create the tables and indexes if they do not already exist.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.

    Returns
    -------
    None : None
    """
    connection.executescript(SCHEMA_STATEMENTS)
    connection.execute(
        "INSERT OR IGNORE INTO index_meta (key, value) VALUES ('schema_version', ?)",
        (INDEX_SCHEMA_VERSION,),
    )
    connection.commit()


def read_index_schema_version(connection: sqlite3.Connection) -> Optional[str]:
    """
    Report the schema version the index was created with.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.

    Returns
    -------
    version : str or None
        Stored schema version, or None for a database predating ``index_meta``.
    """
    row = connection.execute(
        "SELECT value FROM index_meta WHERE key = 'schema_version'"
    ).fetchone()

    return row["value"] if row is not None else None


def analyse_index(connection: sqlite3.Connection) -> None:
    """
    Refresh SQLite's query-planner statistics.

    Worth calling after a bulk ingestion or after promoting a metadata key to an
    indexed column: without it the planner has been measured choosing a
    low-cardinality index over a selective one and running the same query six
    times slower.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.

    Returns
    -------
    None : None
    """
    connection.execute("ANALYZE")
    connection.commit()
