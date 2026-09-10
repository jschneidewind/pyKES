"""
Analysis parameters and plotting instructions of the example app.

Everything an external repository has to decide about *its own* science lives
here: which metadata columns the pipeline reads, how the traces are reduced,
and which curves and numbers the pyKES pages should show. The components
themselves are used verbatim.

The whole dictionary is stored in the dataset's ``processing_parameters``, so a
finished HDF5 file records the settings that produced it. pyKES never hands it
back to the processing functions — those import the same constants from here —
so the stored copy is provenance, not configuration.
"""

# =============================================================================
# Overview columns the pipeline depends on
# =============================================================================

# Columns that select *which* measurement an experiment is. Correcting one
# does not correct the dataset, it describes a different one, so the metadata
# editor renders them read-only: the fix is a corrected sheet and the raw
# files again.
METADATA_USED_FOR_RAW_DATA_LOADING = [
    'Experiment',
    'group',
    'File name H2',
    'File name O2',
]

# Columns the processing functions read. Editing one leaves the stored results
# describing metadata that is no longer there, so the editor clears the
# experiment's Processed flag and the page asks for a reprocessing run.
METADATA_USED_FOR_PROCESSING = [
    'Unisense Irradiation start [s]',
    'Unisense Irradiation end [s]',
    'Pyroscience Irradiation start [s]',
    'Pyroscience Irradiation end [s]',
    'Irradiance [mW/cm2]',
    'Irradiation wavelength [nm]',
    'Irradiated area [cm2]',
    'Fraction of photons reaching inside [-]',
    'Liquid phase volume [mL]',
    'Gas phase volume [mL]',
]

# =============================================================================
# Reduction of the raw traces
# =============================================================================

# Bin width the sensor traces are averaged into before the rate is extracted.
# The sensors log about every 0.4 s, which is far finer than the chemistry;
# averaging suppresses uncorrelated noise instead of merely thinning it.
RESAMPLING_INTERVAL_S = 10

# Length of the sustained-rate window handed to `extract_max_rate`. Explicit
# rather than left to the default, so every experiment of a series is reduced
# over the same stretch of time and the numbers stay comparable.
MAX_RATE_WINDOW_S = 300

# Electrons transferred per product molecule, which is what turns a rate into
# an apparent quantum yield.
ELECTRONS_PER_H2 = 2
ELECTRONS_PER_O2 = 4

# =============================================================================
# Stored in the dataset as provenance
# =============================================================================

PROCESSING_PARAMETERS = {
    'metadata_used_for_raw_data_loading': METADATA_USED_FOR_RAW_DATA_LOADING,
    'metadata_used_for_processing': METADATA_USED_FOR_PROCESSING,
    'resampling_interval_s': RESAMPLING_INTERVAL_S,
    'max_rate_window_s': MAX_RATE_WINDOW_S,
    'electrons_per_H2': ELECTRONS_PER_H2,
    'electrons_per_O2': ELECTRONS_PER_O2,
}

# =============================================================================
# Grouping
# =============================================================================

# Group name -> the metadata path that characterises the group. The Time Series
# and Results Table pages annotate their experiment checkboxes with it, and the
# Analysis Results page splits a group into subsets by its distinct values.
# None means the group has no characteristic parameter.
GROUP_MAPPING = {
    'Reference': None,
    'Intensity': 'metadata/Irradiance [mW/cm2]',
}

# =============================================================================
# What the pages show
# =============================================================================

PLOTTING_INSTRUCTIONS = {

    # Time Series page: one entry per curve the user can switch on. 'unit_y'
    # gives the curve its own y axis, so amounts and rates do not share one.
    'time_series_instructions': {
        'H2 amount': {
            'x': 'processed_data/H2_time_reaction_s',
            'y': 'processed_data/H2_amount_umol',
            'unit_y': 'Amount / umol',
            'unit_x': 'Time / s',
        },
        'O2 amount': {
            'x': 'processed_data/O2_time_reaction_s',
            'y': 'processed_data/O2_amount_umol',
            'unit_y': 'Amount / umol',
            'unit_x': 'Time / s',
        },
        'H2 amount (smoothed)': {
            'x': 'processed_data/H2_time_reaction_s',
            'y': 'processed_data/H2_amount_smoothed_umol',
            'unit_y': 'Amount / umol',
            'unit_x': 'Time / s',
        },
        'O2 amount (smoothed)': {
            'x': 'processed_data/O2_time_reaction_s',
            'y': 'processed_data/O2_amount_smoothed_umol',
            'unit_y': 'Amount / umol',
            'unit_x': 'Time / s',
        },
        'H2 rate': {
            'x': 'processed_data/H2_time_reaction_s',
            'y': 'processed_data/H2_rate_umol_h',
            'unit_y': 'Rate / umol h^-1',
            'unit_x': 'Time / s',
        },
        'O2 rate': {
            'x': 'processed_data/O2_time_reaction_s',
            'y': 'processed_data/O2_rate_umol_h',
            'unit_y': 'Rate / umol h^-1',
            'unit_x': 'Time / s',
        },
    },

    # Analysis Results page: one scalar per entry, plotted against experiments
    # or against a metadata column.
    'kinetic_results_instructions': {
        'H2 maximum rate': {'Value': 'processed_data/H2_max_rate_umol_h',
                            'Unit': 'Rate / umol h^-1'},
        'O2 maximum rate': {'Value': 'processed_data/O2_max_rate_umol_h',
                            'Unit': 'Rate / umol h^-1'},
        'H2 apparent quantum yield': {'Value': 'processed_data/H2_apparent_quantum_yield_percent',
                                      'Unit': 'Apparent quantum yield / %'},
        'O2 apparent quantum yield': {'Value': 'processed_data/O2_apparent_quantum_yield_percent',
                                      'Unit': 'Apparent quantum yield / %'},
        'H2 amount evolved': {'Value': 'processed_data/H2_amount_evolved_umol',
                              'Unit': 'Amount / umol'},
    },

    # Results Table page: one column per entry. 'error' gives the uncertainty
    # a sortable column of its own, 'format' the digits, 'unit' the suffix.
    'results_table_instructions': {
        'H2 max. rate': {'result': 'processed_data/H2_max_rate_umol_h',
                         'error': 'processed_data/H2_max_rate_std_umol_h',
                         'unit': 'umol / h',
                         'format': '.3f'},
        'O2 max. rate': {'result': 'processed_data/O2_max_rate_umol_h',
                         'error': 'processed_data/O2_max_rate_std_umol_h',
                         'unit': 'umol / h',
                         'format': '.3f'},
        'H2 AQY': {'result': 'processed_data/H2_apparent_quantum_yield_percent',
                   'unit': '%',
                   'format': '.2f'},
        'O2 AQY': {'result': 'processed_data/O2_apparent_quantum_yield_percent',
                   'unit': '%',
                   'format': '.2f'},
        # No 'format', so this column shows the default '.4g' — which is what
        # keeps a number readable across the orders of magnitude a rate spans.
        'H2 evolved': {'result': 'processed_data/H2_amount_evolved_umol',
                       'unit': 'umol'},
        'H2/O2 ratio': {'result': 'processed_data/H2_to_O2_ratio',
                        'format': '.2f'},
        'Rate quality flags': {'result': 'processed_data/H2_max_rate_flags'},
    },
}
