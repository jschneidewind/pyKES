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
INDEX_SCHEMA_VERSION = "2.0"


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
    "modified_catalyst_batch",
    "other_entity",
)

# Assigned to entries whose type cannot be determined from the upload.
DEFAULT_ENTITY_TYPE = "other_entity"

# Columns introduced after the first release, applied to an existing index by
# `add_missing_columns`. Only ever additive: a column with a default is
# something an old database can gain, whereas a changed one is a rebuild.
ADDED_COLUMNS = (
    ("metadata_keys", "sub_keys", "TEXT"),
    ("entities", "search_text", "TEXT NOT NULL DEFAULT ''"),
    ("edges", "ordinal", "INTEGER NOT NULL DEFAULT 0"),
)

# Index versions this code can open. A database written before inherited
# metadata moved out of the `effective` column into its own table cannot be
# read correctly by it -- the searches would silently return nothing rather
# than fail -- so it is refused with the repair to run rather than opened.
SUPPORTED_SCHEMA_VERSIONS = ("2.0",)


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

# Separator between the kind of entry a metadata field belongs to and the field
# itself, in a stored key: `finished_semiconductor/Dopants [mol%]`. The same
# character the role path used, and the same `__SLASH__` escaping applies to the
# field name, so a field called `Purity [%/g]` is still splittable.
TYPE_KEY_SEPARATOR = ROLE_PATH_SEPARATOR

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
    search_text      TEXT    NOT NULL DEFAULT ''
);

-- Several targets per role: a semiconductor names every precursor chemical it
-- was made from in one field. What used to make that impossible -- two targets
-- producing identical qualified keys, the second overwriting the first -- is
-- gone now that inherited metadata is a set of contributions rather than a
-- value. `ordinal` keeps the order they were written in, which is the order
-- they should be read in.
CREATE TABLE IF NOT EXISTS edges (
    source   TEXT    NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    role     TEXT    NOT NULL,
    target   TEXT    NOT NULL,
    ordinal  INTEGER NOT NULL DEFAULT 0,
    resolved INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (source, role, target)
);

-- Every metadata value an entity inherits, one row per contributing entry per
-- field. This replaces the `effective` JSON column, and the change is not a
-- storage detail: a metadata field is now named by the *kind of entry that owns
-- it* rather than by the route that reached it, so `Dopants [finished
-- semiconductor]` is one filter however the graph is routed. Because one entry
-- can reach several entries of a kind -- three precursor chemicals, or a
-- semiconductor reached both directly and through a modified batch -- such a
-- field holds a *set*, and a filter on it asks whether *some* contributor
-- satisfies it.
--
-- The primary key is the deduplication rule: an entry reached by two different
-- routes contributes once, which is what stops a diamond in the graph from
-- counting twice. How a contributor was reached belongs to the pair rather than
-- to each of its fields, so it lives in `contribution_sources` and is not
-- repeated on all thirty rows an entry contributes.
CREATE TABLE IF NOT EXISTS contribution_sources (
    entity_id        TEXT    NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    contributor      TEXT    NOT NULL,
    contributor_type TEXT    NOT NULL,
    role_paths       TEXT,
    depth            INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (entity_id, contributor)
);

CREATE TABLE IF NOT EXISTS contributions (
    entity_id        TEXT    NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    entity_type      TEXT    NOT NULL,
    contributor      TEXT    NOT NULL,
    contributor_type TEXT    NOT NULL,
    key              TEXT    NOT NULL,
    sub_key          TEXT    NOT NULL DEFAULT '',
    value            TEXT,
    number           REAL,
    PRIMARY KEY (entity_id, contributor, key, sub_key)
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
    entity_types    TEXT,
    sub_keys        TEXT
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

-- Numbers and text are indexed separately because a range filter and an
-- equality filter want different orders. Both lead with the kind of entry being
-- searched and the pair that names a column, so a lookup narrows to one field
-- of one kind before it looks at a value. `entity_type` is on the row for that
-- reason alone: without it the planner drives from `idx_contrib_entity` and
-- probes once per entity, which measured 97.8 ms against 1.7 ms for a facet's
-- bounds -- paid once per numeric facet on every page load.
CREATE INDEX IF NOT EXISTS idx_contrib_number
    ON contributions(entity_type, contributor_type, key, sub_key, number);
CREATE INDEX IF NOT EXISTS idx_contrib_value
    ON contributions(entity_type, contributor_type, key, sub_key, value);
CREATE INDEX IF NOT EXISTS idx_contrib_entity   ON contributions(entity_id);
CREATE INDEX IF NOT EXISTS idx_sources_entity   ON contribution_sources(entity_id);
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

def open_index(paths: IndexPaths,
               allow_incompatible: bool = False) -> sqlite3.Connection:
    """
    Open the index database, creating and initialising it if absent.

    Parameters
    ----------
    paths : IndexPaths
        Filesystem layout of the database.
    allow_incompatible : bool, optional
        Open an index whose schema version this code cannot read. Only the
        repair tooling passes True: `rebuild_index` needs a connection to the
        very database the version check refuses, so without this the repair
        the refusal names could not be performed.

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

    # Checked before `initialise_index`, so a database this code will not serve
    # is left exactly as it was found. Checking afterwards means the additive
    # migration has already altered a database on its way to being rejected,
    # which is the one state a rollback to the older code cannot read.
    if not allow_incompatible:
        check_schema_version(connection)

    initialise_index(connection)

    return connection


def check_schema_version(connection: sqlite3.Connection) -> None:
    """
    Refuse an index this code cannot read correctly.

    A database written before inherited metadata moved into the contributions
    table still opens, and its searches still run — they just return nothing,
    because the values they look for are in a column that is no longer read.
    Silently answering "no matches" to a question with matches is the worst
    failure this database has, so the version is checked rather than trusted.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.

    Returns
    -------
    None : None

    Raises
    ------
    RuntimeError
        If the index was written by an incompatible version.
    """
    version = read_index_schema_version(connection)

    if version in SUPPORTED_SCHEMA_VERSIONS:
        return

    # No recorded version and nothing stored is simply a database that does
    # not exist yet, which `initialise_index` is about to create. No recorded
    # version with entries in it is a database whose layout nothing can vouch
    # for, and it is refused like any other unreadable one.
    if version is None and not index_holds_entries(connection):
        return

    raise RuntimeError(
        f"This index was built at schema version {version}; this pyKES reads "
        f"{', '.join(SUPPORTED_SCHEMA_VERSIONS)}. Inherited metadata moved out "
        f"of the 'effective' column into its own table, so the stored values "
        f"cannot be read as they are. Rebuild it from the retained uploads "
        f"with `photocat-rebuild --data-root <root>`, which is what keeping "
        f"every upload verbatim is for."
    )


def index_holds_entries(connection: sqlite3.Connection) -> bool:
    """
    Report whether this database already holds entries.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection, possibly to a file that holds no tables yet.

    Returns
    -------
    populated : bool
        True when an `entities` table exists and is not empty.
    """
    present = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'entities'"
    ).fetchone()

    if present is None:
        return False

    return connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0] > 0


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
    add_missing_columns(connection)

    # Recorded after the additive migration has succeeded, and updated rather
    # than ignored, so the stored value describes the layout the database now
    # has instead of the version that first created it. Written with INSERT OR
    # IGNORE it never moved, so an index that had absorbed every column of a
    # newer release still reported the old number — and the next release to
    # list two supported versions would have refused it.
    connection.execute(
        """INSERT INTO index_meta (key, value) VALUES ('schema_version', ?)
           ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
        (INDEX_SCHEMA_VERSION,),
    )
    connection.commit()


def add_missing_columns(connection: sqlite3.Connection) -> None:
    """
    Add columns a newer version introduced to an index that predates them.

    ``CREATE TABLE IF NOT EXISTS`` leaves an existing table exactly as it was,
    so a column added to the schema never reaches a database that already
    exists. Adding them here means an index built before this version keeps
    working without being rebuilt — the column starts empty and fills as
    entries are ingested or the registry is rebuilt.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.

    Returns
    -------
    None : None
    """
    for table, column, declaration in ADDED_COLUMNS:
        present = {row["name"] for row in
                   connection.execute(f"PRAGMA table_info({table})")}

        if column not in present:
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


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
        Stored schema version, or None for a database that has no
        ``index_meta`` table yet — one this code is about to create, or one
        predating the table.
    """
    # Read before `initialise_index` has run, so the table may not exist. A
    # missing table and a missing row mean the same thing to every caller.
    present = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'index_meta'"
    ).fetchone()

    if present is None:
        return None

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
