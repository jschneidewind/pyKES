# pyKES Development Guidelines

## Documentation
Use type hints for all functions, classes, and methods.

Always generate docstrings for new functions, classes, and modules using the Numpydoc style. Use type hints for function parameters and return types wherever possible. Maintain an up-to-date `docs/` folder with Sphinx documentation.

## Project Overview
pyKES is a Python package for kinetic modeling of photochemical reaction networks, specifically designed for water-splitting and photocatalysis experiments. It combines ODE-based reaction network simulation with experimental data management and optimization-based parameter fitting.

## Architecture

### Core Components
1. **Reaction Network Engine** (`reaction_ODE.py`): ODE solver for chemical kinetics
2. **Fitting Framework** (`fitting_ODE.py`): Optimization-based parameter estimation
3. **Data Management** (`database/`): HDF5-based experimental dataset storage
4. **Utilities** (`utilities/`): Helper functions for data filtering and experiment selection

### Data Flow
```
Raw experimental files → data_processing.py (multiprocessing) → ExperimentalDataset → HDF5 storage
                                                                          ↓
                                                                   Fitting_Model → optimization → results
```

## Key Patterns & Conventions

### 1. Reaction String Format
Reactions use a specific DSL: `"[Species] + 2 [Other] > [Product], rate_constant ; multiplier1, multiplier2"`
- Species names in square brackets: `[RuII]`, `[S2O8]`
- Stoichiometry as prefixes: `2 [H2O]`
- Rate constant after comma: `, k1`
- Optional multipliers after semicolon: `; hv_functionA` (used for light-dependent reactions)

Example:
```python
'[RuII-ex] + [S2O8] > [RuIII] + [SO4], k7'
'[RuII] > [RuII-ex], k1 ; hv_functionA'  # Light-induced excitation
```

### 2. Experiment Data Structure
`Experiment` dataclass has nested dictionaries for flexibility:
```python
experiment.metadata['temperature']  # Experimental conditions
experiment.raw_data['current']      # NumPy arrays
experiment.processed_data['baseline_corrected']['efficiency']  # Nested processing results
```

### 3. Attribute Path Resolution
`fitting_ODE.py` uses string paths to reference experiment attributes dynamically:
```python
model.data_to_be_fitted = {
    '[O2]': {'x': 'time_series_data.x_diff',  # Resolves to experiment.time_series_data.x_diff
             'y': 'time_series_data.y_diff'}
}
```
The `resolve_experiment_attributes()` function handles both dict keys and object attributes via `getattr()`.

### 4. Function Multipliers Pattern
Dynamic multipliers (e.g., for competing light absorption) use a nested dict structure:
```python
'hv_functionA': {
    'function': calculate_excitations_per_second_competing,
    'arguments': {
        'photon_flux': 'photon_flux',      # String references resolved at runtime
        'concentration_A': '[RuII]',        # Can reference species concentrations
        'extinction_coefficient_A': 'Ru_II_extinction_coefficient'
    }
}
```

### 5. HDF5 Storage Strategy
- Nested dictionaries stored hierarchically using `save_nested_dict_to_hdf5()`
- NumPy arrays saved as datasets, complex types as JSON or pickle
- Overview DataFrames stored with pandas `.to_hdf()` (PyTables format)
- Each experiment in its own group: `f['experiment_name']`

### 6. Multiprocessing for Data Ingestion
`data_processing.py` uses `ProcessPoolExecutor` for parallel file reading:
- Returns success/failure dicts to handle exceptions gracefully
- Full traceback preserved in error results for debugging
- Partial functions pass database and processing functions to workers

## Development Workflows

### Local Installation
Install in editable mode from other projects:
```bash
pip install -e /Users/jacob/Documents/Water_Splitting/Projects/pyKES/pyKES
```

### VS Code Setup
Add to `.vscode/settings.json` in consuming projects:
```json
{
    "python.analysis.extraPaths": ["/Users/jacob/Documents/Water_Splitting/Projects/pyKES/pyKES/src"]
}
```

### Running Optimization
```python
model = Fitting_Model(reaction_network)
model.rate_constants_to_optimize = {'k1': (0.1, 1.0), 'k2': (0.1, 1.0)}  # (min, max) bounds
model.experiments = [exp1, (exp2, weight), ...]  # Optional weights for experiments
model.optimize(workers=-1)  # Uses all CPU cores
```

### Merging Multiple HDF5 Files
```python
merged = ExperimentalDataset.merge_hdf5_files(
    ['exp1.h5', 'exp2.h5'],
    output_filename='merged.h5'
)
```

## Critical Implementation Details

### Dictionary Order Preservation
Optimization relies on Python 3.7+ dict order preservation:
- `list(rate_constants_to_optimize.values())` → bounds order
- `dict(zip(rate_constants_to_optimize.keys(), optimized_values))` → reconstruction
Both must iterate in identical order.

### Excel Temp Files
File filtering excludes Excel temp files (starting with `~$`):
```python
if any(keyword in file for keyword in keywords) and not file.startswith('~$')
```

### Type Hints for Optional Parameters
Use `Optional[type] = default` pattern with required params first:
```python
def function(required: str, optional: Optional[list] = None):
```

### Formatted Output
Scientific notation with 4 decimals for rate constants:
```python
formatted = {k: f"{float(v):.4e}" for k, v in constants.items()}
```

## Common Pitfalls

1. **HDF5 attribute access**: Use `exp_group.attrs['key']` not `f.attrs['key']` when iterating experiment groups
2. **Loss function signatures**: Must return `(error, transformed_data)` tuple for visualization
3. **Species indexing**: Always use `species.index(species_name)` to get solution array column
4. **Experiment weights**: Support both `experiment` and `(experiment, weight)` tuple formats in lists

## Dependencies
- Core: numpy, scipy, pandas, h5py, matplotlib
- Optional: streamlit (for visualization apps), openpyxl (Excel I/O)
- Minimum Python: 3.8 (but 3.7+ recommended for dict order)
