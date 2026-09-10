"""
Metadata lookup of the example app.

Handed to pyKES through ``FileUploadHandler.metadata_retrival_function``,
which is called as ``(experiment_name, overview_df) -> metadata_dict``. The
whole overview row becomes the metadata of the experiment, which is what makes
the overview sheet the single description of a run.

Only ``'experiment_name'`` is required of the returned dictionary; ``'color'``
and ``'group'`` are picked up by the plotting pages when present. Keys that
are not overview columns — ``'experiment_name'`` here — belong to the app and
are left alone by the dataset, while every key that *is* an overview column is
kept equal to the sheet (see `pyKES.database.database_experiments`).
"""

import pandas as pd


def metadata_retrival_function(experiment_name: str, overview_df: pd.DataFrame) -> dict:
    """
    Look one experiment's row up in the overview sheet.

    Parameters
    ----------
    experiment_name : str
        Experiment to look up.
    overview_df : pandas.DataFrame
        Overview sheet of the dataset.

    Returns
    -------
    dict
        The row as a dictionary, plus ``'experiment_name'``.

    Raises
    ------
    ValueError
        If the sheet holds no row, or more than one row, for this experiment.
        Both mean the sheet cannot say what the experiment is, which is worth
        stopping for rather than guessing at.
    """

    matching_rows = overview_df.loc[overview_df['Experiment'] == experiment_name]

    if matching_rows.empty:
        raise ValueError(f"No experiment named {experiment_name!r} in the overview sheet")

    if len(matching_rows) > 1:
        raise ValueError(f"The overview sheet holds {len(matching_rows)} rows named "
                         f"{experiment_name!r}; experiment names have to be unique")

    metadata_dict = matching_rows.iloc[0].to_dict()
    metadata_dict['experiment_name'] = metadata_dict['Experiment']

    return metadata_dict
