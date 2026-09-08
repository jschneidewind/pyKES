"""
Rebuild the index from the retained uploads, from a shell.

`rebuild_index` is the repair the schema-version check names, and it is the
fallback behind every claim that keeping uploads verbatim makes the database
reconstructible. It empties and repopulates the whole index, holding the write
lock for minutes on a real archive, so it does not belong behind a button in a
web page that a browser refresh can interrupt — and it needs a connection to
an index the application itself would refuse to open.

Run it with the application stopped::

    photocat-rebuild --data-root /srv/photocat/data --dry-run
    photocat-rebuild --data-root /srv/photocat/data --yes

A rebuild reverts corrections made on the entry page: `update_entity_metadata`
writes to the entity row and there is no journal to replay. They are exported
before anything is deleted, so they can be re-applied by hand afterwards.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from pyKES.database.entity_schema import load_entity_schemas
from pyKES.database.index_ingest import (
    ingest_options_for,
    rebuild_index,
    recover_entity_types,
)
from pyKES.database.index_schema import (
    INDEX_SCHEMA_VERSION,
    IndexPaths,
    analyse_index,
    open_index,
    read_index_schema_version,
)
from pyKES.database_app.config import DatabaseAppConfig


# =============================================================================
# Reporting
# =============================================================================

# Counted before and after, because these are the two numbers that expose an
# unfaithful rebuild: entries collapsing into one kind, and a reference graph
# that quietly disappeared.
def describe_index(connection) -> Dict[str, Any]:
    """
    Summarise what the index currently holds.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.

    Returns
    -------
    summary : dict
        Entry counts by kind, plus edge and upload totals.
    """
    return {
        "entities_by_type": {row["entity_type"]: row["entries"] for row in
                             connection.execute(
                                 "SELECT entity_type, COUNT(*) AS entries "
                                 "FROM entities GROUP BY 1 ORDER BY 1")},
        "edges": connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
        "uploads": connection.execute("SELECT COUNT(*) FROM uploads").fetchone()[0],
    }


def print_summary(label: str, summary: Dict[str, Any]) -> None:
    """
    Write one index summary to standard output.

    Parameters
    ----------
    label : str
        Heading for this summary.
    summary : dict
        As returned by `describe_index`.

    Returns
    -------
    None : None
    """
    print(f"\n{label}:")
    for entity_type, entries in summary["entities_by_type"].items():
        print(f"  {entries:6d}  {entity_type}")
    print(f"  {summary['edges']:6d}  references")
    print(f"  {summary['uploads']:6d}  uploads")


def read_corrections(connection) -> List[Dict[str, Any]]:
    """
    Read the entries that have been corrected since they were uploaded.

    A rebuild reverts these, so they are exported while they still exist.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.

    Returns
    -------
    corrections : list of dict
        One row per corrected entry.
    """
    return [dict(row) for row in connection.execute(
        """SELECT entity_id, entity_type, owner, created_at, updated_at, metadata
           FROM entities WHERE updated_at > created_at ORDER BY entity_id""")]


# =============================================================================
# Repairs that are not a rebuild
# =============================================================================

def find_orphan_uploads(connection, paths: IndexPaths) -> List[Path]:
    """
    List retained files that no upload row refers to.

    The verbatim copy is written before the row is committed, so an ingestion
    killed part-way through — a restart during a large batch — rolls the row
    back and leaves the file. It is harmless: nothing reads it, a rebuild
    iterates the table rather than the directory, and re-uploading the same
    file overwrites it and inserts the row. But it is a file the archive
    believes it does not have, and on a small disk they accumulate, so an
    operator should be able to see them.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.
    paths : IndexPaths
        Filesystem layout of the database.

    Returns
    -------
    orphans : list of Path
        Files in the upload store with no matching row.
    """
    known = {Path(row["stored_path"]).name for row in
             connection.execute("SELECT stored_path FROM uploads")}

    return sorted(path for path in paths.upload_directory.iterdir()
                  if path.is_file() and path.name not in known)


def backfill_stored_paths(connection) -> int:
    """
    Rewrite absolute upload paths as names within the upload store.

    Rows written before the store became relative name a directory that is
    wrong as soon as the data root moves. Reading tolerates both shapes, so
    this is tidying rather than a migration.

    Parameters
    ----------
    connection : sqlite3.Connection
        Open connection to the index database.

    Returns
    -------
    rewritten : int
        Number of rows changed.
    """
    rows = connection.execute(
        "SELECT id, stored_path FROM uploads WHERE stored_path LIKE '%/%'"
    ).fetchall()

    for row in rows:
        connection.execute("UPDATE uploads SET stored_path = ? WHERE id = ?",
                           (Path(row["stored_path"]).name, row["id"]))
    connection.commit()

    return len(rows)


# =============================================================================
# Command line
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """
    Describe the command-line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Rebuild the photocatalysis index from its retained uploads.")
    parser.add_argument("--data-root", default=None,
                        help="Directory holding index.sqlite, uploads/ and "
                             "payloads/. Defaults to $PHOTOCAT_DATA_ROOT.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be re-ingested and stop.")
    parser.add_argument("--export-corrections", default=None, metavar="PATH",
                        help="Write the entry-page corrections a rebuild "
                             "reverts to this JSON file first.")
    parser.add_argument("--backfill-paths", action="store_true",
                        help="Rewrite absolute upload paths as relative names "
                             "and stop.")
    parser.add_argument("--yes", action="store_true",
                        help="Required to actually rebuild.")

    return parser


def main() -> None:
    """
    Rebuild the index, or report what a rebuild would do.

    Returns
    -------
    None : None
    """
    arguments = build_parser().parse_args()

    config = DatabaseAppConfig(data_root=arguments.data_root) \
        if arguments.data_root else DatabaseAppConfig()
    try:
        paths = IndexPaths(root=config.data_root, create=False)
    except FileNotFoundError as absent:
        sys.exit(str(absent))

    # The version check is what a rebuild repairs, so it must not stand in the
    # way of running one.
    connection = open_index(paths, allow_incompatible=True)
    schemas = load_entity_schemas(config.schema_directory)

    print(f"Index:  {paths.index_path}")
    print(f"Schema: recorded {read_index_schema_version(connection)}, "
          f"code {INDEX_SCHEMA_VERSION}")

    if arguments.backfill_paths:
        print(f"Rewrote {backfill_stored_paths(connection)} absolute upload paths.")
        return

    before = describe_index(connection)
    print_summary("Before", before)

    orphans = find_orphan_uploads(connection, paths)
    if orphans:
        print(f"\n{len(orphans)} retained files have no upload row, left by an "
              f"ingestion that was interrupted. A rebuild ignores them; delete "
              f"them if the disk matters:")
        for orphan in orphans[:10]:
            print(f"  {orphan}")
        if len(orphans) > 10:
            print(f"  … and {len(orphans) - 10} more")

    corrections = read_corrections(connection)
    if corrections:
        print(f"\n{len(corrections)} entries carry corrections made on the entry "
              f"page. A rebuild reverts them.")
    if arguments.export_corrections:
        Path(arguments.export_corrections).write_text(
            json.dumps(corrections, indent=2), encoding="utf-8")
        print(f"Corrections written to {arguments.export_corrections}")

    uploads = connection.execute("SELECT * FROM uploads ORDER BY id").fetchall()
    recovered = recover_entity_types(connection)

    print("\nUploads to re-ingest:")
    for upload in uploads:
        options = ingest_options_for(upload, recovered, {}, schemas)
        print(f"  {upload['id']:4d}  {upload['filename']}  "
              f"[{upload['kind']}] as {options['entity_type']}")

    if arguments.dry_run:
        print("\nDry run: nothing was changed.")
        return

    if not arguments.yes:
        sys.exit("\nRefusing to rebuild without --yes. Take a backup first, "
                 "and stop the application: a rebuild holds the write lock "
                 "for minutes and serves a half-built index to anyone reading.")

    print("\nRebuilding…")
    reports = rebuild_index(connection, paths, schemas=schemas)
    analyse_index(connection)

    print_summary("After", describe_index(connection))
    print(f"\nRe-ingested {len(reports)} uploads, "
          f"{sum(len(report.added) for report in reports)} entries.")


if __name__ == "__main__":
    main()
