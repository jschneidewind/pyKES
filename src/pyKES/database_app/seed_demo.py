"""
Seed a database so the application can be opened and tried.

Run this before `streamlit run` to get a prototype with something in it::

    python -m pyKES.database_app.seed_demo --root /tmp/photocat-demo
    PHOTOCAT_DATA_ROOT=/tmp/photocat-demo streamlit run src/pyKES/database_app/Home.py

With no arguments it builds a synthetic archive whose reference chain matches
the group's own — experiment, catalyst batch, finished semiconductor, precursor
chemical — so every feature of the application has something to show. Point
``--files`` at a directory of real exports to seed from those instead.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pyKES.database.database_experiments import Experiment, ExperimentalDataset
from pyKES.database.index_ingest import ingest_entity_sheet, ingest_hdf5_upload
from pyKES.database.index_schema import IndexPaths, analyse_index, open_index
from pyKES.database_app.config import GROUP_REFERENCE_INSTRUCTIONS


# =============================================================================
# Shape of the synthetic archive
# =============================================================================

# Small enough to seed in seconds, large enough that facets, pagination and the
# property map all have something to do.
EXPERIMENT_COUNT = 60
BATCH_COUNT = 12
SEMICONDUCTOR_COUNT = 6
PRECURSOR_COUNT = 3

# Samples per synthetic trace. Enough for the payload to look like a real
# measurement and to exercise gzip.
TRACE_POINTS = 900

# Values the synthetic metadata is drawn from, chosen to match the group's own
# vocabulary so the demo reads like the real thing.
SYNTHESIS_TEMPERATURES = (1000, 1050, 1100, 1150, 1200)
PHOTODEPOSITION_WAVELENGTHS = (365, 405, 455)
IRRADIANCES = (12.0, 25.0, 44.25, 80.0, 120.0)
COCATALYSTS = ("Rh", "Cr", "Pt", "Ru")
OPERATORS = ("ae", "nb", "mz", "vsa")

# Curves are coloured by the experiment's own colour, so a demo whose entries
# are all black would make the comparison plot useless. Eleven colours, a prime,
# so a filter selecting every nth experiment still gets a spread rather than
# landing repeatedly on the same one.
TRACE_COLORS = ("#22c55e", "#38bdf8", "#f59e0b", "#f472b6", "#a78bfa",
                "#f87171", "#2dd4bf", "#facc15", "#c084fc", "#fb923c",
                "#60a5fa")

RANDOM_SEED = 20260906


def build_precursor_sheet(directory: Path) -> Path:
    """
    Write the precursor-chemical sheet.

    Parameters
    ----------
    directory : Path
        Directory the sheet is written to.

    Returns
    -------
    path : Path
        The written file.
    """
    rows = [{"Experiment": f"BC-{index + 1}",
             "group": "Reference",
             "Supplier": ("Acme", "Merck", "Alfa")[index % 3],
             "Purity [%]": (99.9, 99.99, 99.0)[index % 3],
             "Notes": ""}
            for index in range(PRECURSOR_COUNT)]

    path = directory / "precursor_chemicals.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False)

    return path


def build_semiconductor_sheet(directory: Path, generator) -> Path:
    """
    Write the finished-semiconductor sheet, referencing two precursors each.

    Two references of the same kind under *distinct* roles is exactly the case
    that would silently lose data under a single shared role, so the demo
    exercises it.

    Parameters
    ----------
    directory : Path
        Directory the sheet is written to.
    generator : numpy.random.Generator
        Source of the drawn values.

    Returns
    -------
    path : Path
        The written file.
    """
    rows = []
    for index in range(SEMICONDUCTOR_COUNT):
        rows.append({
            "Experiment": f"SEMI-{index + 1:03d}",
            "group": "Reference",
            "Precursor Chemical A": f"BC-{(index % PRECURSOR_COUNT) + 1}",
            "Precursor Chemical B": f"BC-{((index + 1) % PRECURSOR_COUNT) + 1}",
            "Catalyst material": "Al:SrTiO3",
            "Synthesis temperature [°C]": int(
                generator.choice(SYNTHESIS_TEMPERATURES)),
            "Synthesis route": ("Osterloh", "Lercher")[index % 2],
            "Notes": "",
        })

    path = directory / "finished_semiconductors.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False)

    return path


def build_batch_sheet(directory: Path, generator) -> Path:
    """
    Write the catalyst-batch sheet, each referencing one semiconductor.

    Parameters
    ----------
    directory : Path
        Directory the sheet is written to.
    generator : numpy.random.Generator
        Source of the drawn values.

    Returns
    -------
    path : Path
        The written file.
    """
    rows = []
    for index in range(BATCH_COUNT):
        rows.append({
            "Experiment": f"ABC-{index + 1:03d}",
            "group": "Reference",
            "Finished Semiconductor": f"SEMI-{(index % SEMICONDUCTOR_COUNT) + 1:03d}",
            "Loading method [photodeposition/wet impregnation]": str(
                generator.choice(("photodeposition", "wet impregnation"))),
            "Photodeposition wavelength [nm]": int(
                generator.choice(PHOTODEPOSITION_WAVELENGTHS)),
            "Photodeposition time [min]": int(generator.choice((20, 30, 45))),
            "Co-catalyst A": str(generator.choice(COCATALYSTS)),
            "Co-catalyst A loading [wt%]": float(
                generator.choice((0.05, 0.1, 0.2, 0.5))),
            "Notes": "",
        })

    path = directory / "catalyst_batches.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False)

    return path


def build_experiment_batch(directory: Path, generator) -> Path:
    """
    Write a measured HDF5 batch whose experiments reference catalyst batches.

    The traces are a saturating exponential plus noise, which is enough for the
    detail page to draw something recognisable and for the payload to be a real
    HDF5 file rather than a stub.

    Parameters
    ----------
    directory : Path
        Directory the file is written to.
    generator : numpy.random.Generator
        Source of the drawn values.

    Returns
    -------
    path : Path
        The written file.
    """
    dataset = ExperimentalDataset()
    time_s = np.linspace(0.0, 3600.0, TRACE_POINTS)

    for index in range(EXPERIMENT_COUNT):
        irradiance = float(generator.choice(IRRADIANCES))
        rate_constant = 1.0 / float(generator.uniform(600.0, 2400.0))
        plateau = irradiance * float(generator.uniform(0.02, 0.08))

        amount = plateau * (1.0 - np.exp(-rate_constant * time_s))
        amount = amount + generator.normal(0.0, plateau * 0.01, time_s.size)
        rate = np.gradient(amount, time_s)

        dataset.add_experiment(Experiment(
            experiment_name=f"EXP-{index + 1:04d}",
            raw_data_file=f"EXP-{index + 1:04d}.csv",
            color=TRACE_COLORS[index % len(TRACE_COLORS)],
            group="Reference",
            metadata={
                "Experiment": f"EXP-{index + 1:04d}",
                "Catalyst Batch [experiment no.]": f"ABC-{(index % BATCH_COUNT) + 1:03d}",
                "Irradiance A [mW/cm2]": irradiance,
                "Irradiation wavelength A [nm]": int(
                    generator.choice((365, 455, 525))),
                "Temperature [°C]": int(generator.choice((20, 25, 30))),
                "Catalyst concentration [g/L]": float(
                    generator.choice((0.5, 1.0, 2.0))),
                "Liquid phase volume [mL]": 2,
                "Measured Analyte [O2 or H2]": str(generator.choice(("O2", "H2"))),
                "Measurement phase [liquid/gas]": str(
                    generator.choice(("liquid", "gas"))),
                "Active": True,
                "Operator": str(generator.choice(OPERATORS)),
                "Notes": "",
            },
            raw_data={"time_s": time_s, "amount_umol": amount},
            processed_data={
                "time_reaction_s": time_s,
                "data_reaction_umol": amount,
                "rate_umol_s": rate,
                "max_rate_umol_s": float(rate.max()),
                "max_rate_time_s": float(time_s[int(rate.argmax())]),
                "time_unit": "s",
                "data_unit": "umol",
                "rate_unit": "umol/s",
                "apparent_quantum_yield": float(
                    rate.max() / irradiance * 1e4),
                "light_to_hydrogen_efficiency": float(
                    rate.max() / irradiance * 1e3),
            },
        ))

    dataset.plotting_instruction = {
        # The curves the database application offers for comparison. Carried
        # into every payload, so a payload describes its own plots.
        "time_series_instructions": {
            "Reaction": {"x": "processed_data/time_reaction_s",
                         "y": "processed_data/data_reaction_umol",
                         "unit_x": "processed_data/time_unit",
                         "unit_y": "processed_data/data_unit"},
            "Rate": {"x": "processed_data/time_reaction_s",
                     "y": "processed_data/rate_umol_s",
                     "x_point": "processed_data/max_rate_time_s",
                     "y_point": "processed_data/max_rate_umol_s",
                     "unit_x": "processed_data/time_unit",
                     "unit_y": "processed_data/rate_unit"},
            "Raw": {"x": "raw_data/time_s", "y": "raw_data/amount_umol"},
        },
        "index_instructions": {
            "Max. rate (umol/s)": {"result": "processed_data/max_rate_umol_s"},
            "Apparent quantum yield (%)": {
                "result": "processed_data/apparent_quantum_yield",
                "format": ".4f"},
            "Light-to-hydrogen efficiency (%)": {
                "result": "processed_data/light_to_hydrogen_efficiency",
                "format": ".4f"},
        },
        "reference_instructions": GROUP_REFERENCE_INSTRUCTIONS["experiment"],
    }

    path = directory / "measured_batch.h5"
    dataset.save_to_hdf5(str(path), compression="gzip", verbose=False)

    return path


def seed(root: Path, source_files: Path = None) -> dict:
    """
    Build a database at ``root``, ready for the application to open.

    Entries are ingested deliberately out of order — experiments first, then
    batches, then semiconductors, then precursors — so that forward references
    are exercised and seen to resolve.

    Parameters
    ----------
    root : Path
        Directory the index and its file tiers are created in.
    source_files : Path, optional
        Directory of real exports to seed from instead of synthetic data. It
        must hold one ``.h5`` batch and the metadata sheets, named so that the
        kind of each is recognisable.

    Returns
    -------
    counts : dict
        Entries added per kind.
    """
    paths = IndexPaths(root=root)
    connection = open_index(paths)
    generator = np.random.default_rng(RANDOM_SEED)

    staging = Path(tempfile.mkdtemp())
    if source_files is None:
        batch_file = build_experiment_batch(staging, generator)
        sheets = [
            (build_batch_sheet(staging, generator), "catalyst_batch"),
            (build_semiconductor_sheet(staging, generator), "finished_semiconductor"),
            (build_precursor_sheet(staging), "precursor_chemical"),
        ]
    else:
        batch_file = next(Path(source_files).glob("*.h5"))
        sheets = [(path, kind) for path, kind in
                  ((next(Path(source_files).glob(f"*{pattern}*.xlsx"), None), kind)
                   for pattern, kind in (("Catalyst", "catalyst_batch"),
                                         ("Semiconductor", "finished_semiconductor")))
                  if path is not None]

    counts = {}
    report = ingest_hdf5_upload(connection, paths, batch_file, "ae", "experiment")
    counts["experiment"] = len(report.added)

    for sheet, entity_type in sheets:
        report = ingest_entity_sheet(
            connection, paths, sheet, entity_type, "nb",
            GROUP_REFERENCE_INSTRUCTIONS.get(entity_type, {}))
        counts[entity_type] = len(report.added)

    analyse_index(connection)
    connection.close()
    shutil.rmtree(staging, ignore_errors=True)

    return counts


def main() -> None:
    """
    Command-line entry point.

    Returns
    -------
    None : None
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/tmp/photocat-demo",
                        help="directory to build the database in")
    parser.add_argument("--files", default=None,
                        help="directory of real exports to seed from instead")
    parser.add_argument("--fresh", action="store_true",
                        help="delete any existing database at --root first")
    arguments = parser.parse_args()

    root = Path(arguments.root)
    if arguments.fresh and root.exists():
        shutil.rmtree(root)

    counts = seed(root, Path(arguments.files) if arguments.files else None)

    print(f"Seeded {root}:")
    for entity_type, count in counts.items():
        print(f"  {count:5d}  {entity_type}")
    print(f"\nPHOTOCAT_DATA_ROOT={root} streamlit run "
          f"src/pyKES/database_app/Home.py")


if __name__ == "__main__":
    main()
