PROCESSING_PARAMETERS = {
    'H2_processing_parameters': {
                        'offset': 60,
                        'savgol_window': 30,
                        'savgol_polyorder': 1,
                        'savgol_window_diff': 50,
                        'savgol_polyorder_diff': 1,
                        'interval_resampling': 10,
                        'poly_order': 4,
                    },
    'O2_processing_parameters': { 
                        'offset': 60,
                        'savgol_window': 10,
                        'savgol_polyorder': 3,
                        'savgol_window_diff': 50,
                        'savgol_polyorder_diff': 1,
                        'interval_resampling': 10,
                        'poly_order': 4,
                    },
    'H2_processing_parameters_gas_phase': {
                        'offset': 60,
                        'savgol_window': 30,
                        'savgol_polyorder': 1,
                        'savgol_window_diff': 50,
                        'savgol_polyorder_diff': 1,
                        'interval_resampling': 10,
                        'poly_order': 3,
                    },
    'O2_processing_parameters_gas_phase': { 
                        'offset': 60,
                        'savgol_window': 10,
                        'savgol_polyorder': 3,
                        'savgol_window_diff': 50,
                        'savgol_polyorder_diff': 1,
                        'interval_resampling': 10,
                        'poly_order': 3,
                    },
    'fitting_parameters': {
                    'reaction_network': [
                           '[H2O] > [H2O-int], k1',
                           '[H2O-int] > 2 [H2-aq] + [O2-aq], k2',
                           '[O2-aq] > [O2-g], k3',
                           '[H2-aq] > [H2-g], k3 ; factor1',
                            ],   
                    'rate_constants_to_optimize': {
                            'k1': (1e-10, 1e-8),
                            'k2': (1e-4, 1e-2),
                            'k3': (1e-4, 1e-2)
                            },
                    'data_to_be_fitted': {
                            '[O2-aq]': {'x': 'processed_data/common_time_reaction',
                                        'y': 'processed_data/O2_data_aligned'},
                            '[H2-aq]': {'x': 'processed_data/common_time_reaction',
                                        'y': 'processed_data/H2_data_aligned'},
                            },   
                    'initial_conditions': {
                            '[H2O]': 55.5 * 1e6,  # Convert from mol/L to μmol/L
                            },
                    'other_multipliers': {
                            'factor1': 1.68, # solubility ratio of O2/H2
                            },
                    'times': {
                        'times': 'processed_data/common_time_reaction'
                        },
            },
    'fitting_parameters_mapping': {
                    'species_mapping': {
                            '[O2-aq]': 'O2',
                            '[H2-aq]': 'H2',
                            },
                    'rate_constant_mapping': {
                            'O2': 1,
                            'H2': 1,
                            },
            },
    'fitting_parameters_fixed': {
                    'reaction_network': [
                           '[H2O] > [H2O-int], k1',
                           '[H2O-int] > 2 [H2-aq] + [O2-aq], k2',
                           '[O2-aq] > [O2-g], k3',
                           '[H2-aq] > [H2-g], k3 ; factor1',
                            ],   
                    'rate_constants_to_optimize': {
                            'k2': (1e-5, 1e-1),
                            },
                    # 'fixed_rate_constants': {
                    #         'k1': 6.0e-9, # fixed based on average of fitting all experiments with all three rate constants
                    #         'k3': 0.0073, # fixed based on average of fitting all experiments with all three rate constants
                    #         },  
                    'fixed_rate_constants': {
                            'k1': 4.3e-9, # fixed based on average of fitting all experiments with all three rate constants
                            'k3': 0.0045, # fixed based on average of fitting all experiments with all three rate constants
                            },    
                    'data_to_be_fitted': {
                            '[O2-aq]': {'x': 'processed_data/common_time_reaction',
                                        'y': 'processed_data/O2_data_aligned'},
                            '[H2-aq]': {'x': 'processed_data/common_time_reaction',
                                        'y': 'processed_data/H2_data_aligned'},
                            },   
                    'initial_conditions': {
                            '[H2O]': 55.5 * 1e6,  # Convert from mol/L to μmol/L
                            },
                    'other_multipliers': {
                            'factor1': 1.68, # solubility ratio of O2/H2
                            },
                    'times': {
                        'times': 'processed_data/common_time_reaction'
                        },
            },
    'fitting_parameters_fixed_mapping': {
                    'species_mapping': {
                            '[O2-aq]': 'O2_fixed',
                            '[H2-aq]': 'H2_fixed',
                            },
                    'rate_constant_mapping': {
                            'O2_fixed': 0,
                            'H2_fixed': 0,
                            },
            },
    'fitting_parameters_flexible_H2': {
                    'reaction_network': [
                            '[H2O] > [Gas-int], k1',
                            '[Gas-int] > [Gas-aq], k2',
                            '[Gas-aq] > [Gas-g], k3',
                            ],   
                    'rate_constants_to_optimize': {
                            'k1': (1e-8, 1e-6),
                            'k2': (1e-4, 1e-2),
                            'k3': (1e-2, 1e-0),
                            },  
                    'data_to_be_fitted': {
                            '[Gas-aq]': {'x': 'processed_data/common_time_reaction',
                                         'y': 'processed_data/H2_data_aligned'},
                            },
                    'initial_conditions': {
                            '[H2O]': 55.5 * 1e6,  # Convert from mol/L to μmol/L
                            },
                    'times': {
                        'times': 'processed_data/common_time_reaction'
                        },
                }, 
    'fitting_parameters_flexible_H2_mapping': {   
                    'species_mapping': {
                            '[Gas-aq]': 'H2_flexible',
                            },
                    'rate_constant_mapping': {
                            'H2_flexible': 1,
                            },
                },        
    'fitting_parameters_flexible_O2': {
                    'reaction_network': [
                            '[H2O] > [Gas-int], k1',
                            '[Gas-int] > [Gas-aq], k2',
                            '[Gas-aq] > [Gas-g], k3',
                            ],   
                    'rate_constants_to_optimize': {
                            'k1': (1e-8, 1e-6),
                            'k2': (1e-4, 1e-2),
                            'k3': (1e-2, 1e-0),
                            },  
                    'data_to_be_fitted': {
                            '[Gas-aq]': {'x': 'processed_data/common_time_reaction',
                                         'y': 'processed_data/O2_data_aligned'},
                            },
                    'initial_conditions': {
                            '[H2O]': 55.5 * 1e6,  # Convert from mol/L to μmol/L
                            },
                    'times': {
                        'times': 'processed_data/common_time_reaction'
                        },
            }, 
    'fitting_parameters_flexible_O2_mapping': {   
                    'species_mapping': {
                            '[Gas-aq]': 'O2_flexible',
                            },
                    'rate_constant_mapping': {
                            'O2_flexible': 1,
                            },
            },
}

GROUP_MAPPING = {
        'Reference': None,
        'Intensity': 'metadata/Irradiance [mW/cm2]',
        'Loading': 'metadata/Catalyst loading [wt% Rh/Cr]',
        'D2O': 'metadata/D2O',
        'Temperature': 'metadata/Temperature [°C]',
        'Gas phase': None,
        'Gas phase D2O': 'metadata/D2O',
        }

PLOTTING_INSTRUCTIONS = {
    'time_series_instructions': {
                    'Raw (H2)': {
                        'x': 'raw_data/H2_time_s',
                        'y': 'raw_data/H2_umol_L'
                        },
                    'Raw (O2)': {
                        'x': 'raw_data/O2_time_s',
                        'y': 'raw_data/O2_data'
                        },
                    'Raw (H2, gas phase)': {
                        'x': 'raw_data/H2_time_s',
                        'y': 'raw_data/H2_Pa'
                        },
                    'Raw (O2, gas phase)': {
                        'x': 'raw_data/O2_time_s',
                        'y': 'raw_data/O2_data'
                        },
                    'Reaction (H2)': {
                        'x': 'processed_data/H2_time_reaction',
                        'y': 'processed_data/H2_data_reaction'
                        },
                    'Reaction (O2)': {
                        'x': 'processed_data/O2_time_reaction',
                        'y': 'processed_data/O2_data_reaction'
                        },
                    'Reaction (H2, gas phase)': {
                        'x': 'processed_data/H2_gas_time_reaction',
                        'y': 'processed_data/H2_gas_data_reaction'
                        },
                    'Reaction (O2, gas phase)': {
                        'x': 'processed_data/O2_gas_time_reaction',
                        'y': 'processed_data/O2_gas_data_reaction'
                        },
                    'Fit (H2)': {
                        'x': 'processed_data/common_time_reaction',
                        'y': 'processed_data/H2_fit'
                        },
                    'Fit (O2)': {
                        'x': 'processed_data/common_time_reaction',
                        'y': 'processed_data/O2_fit'
                        },
                    'Fixed fit (H2)': {
                        'x': 'processed_data/common_time_reaction',
                        'y': 'processed_data/H2_fixed_fit'
                        },
                    'Fixed fit (O2)': {
                        'x': 'processed_data/common_time_reaction',
                        'y': 'processed_data/O2_fixed_fit'  
                        },
                    'Poly fit (H2)': {
                        'x': 'processed_data/H2_time_reaction',
                        'y': 'processed_data/H2_poly_fit'
                        },
                    'Poly fit (O2)': {
                        'x': 'processed_data/O2_time_reaction',
                        'y': 'processed_data/O2_poly_fit'
                        },
                    'Poly fit (H2, gas phase)': {
                        'x': 'processed_data/H2_gas_time_reaction',
                        'y': 'processed_data/H2_gas_poly_fit'
                        },
                    'Poly fit (O2, gas phase)': {
                        'x': 'processed_data/O2_gas_time_reaction',
                        'y': 'processed_data/O2_gas_poly_fit'   
                        },
                    'Flexible fit (H2)': {
                        'x': 'processed_data/common_time_reaction',
                        'y': 'processed_data/H2_flexible_fit'
                        },
                    'Flexible fit (O2)': {
                        'x': 'processed_data/common_time_reaction',
                        'y': 'processed_data/O2_flexible_fit'
                        },
                    'Smoothed (H2)': {
                        'x': 'processed_data/H2_time_reaction',
                        'y': 'processed_data/H2_data_smoothed'
                        },
                    'Smoothed (O2)': {
                        'x': 'processed_data/O2_time_reaction',
                        'y': 'processed_data/O2_data_smoothed'
                        },
                    'Rate (H2)': {
                        'x': 'processed_data/H2_time_diff',
                        'y': 'processed_data/H2_data_diff'
                        },
                    'Rate (O2)': {
                        'x': 'processed_data/O2_time_diff',
                        'y': 'processed_data/O2_data_diff'
                        },
                    'Rate (H2, gas phase)': {
                        'x': 'processed_data/H2_gas_time_diff',
                        'y': 'processed_data/H2_gas_data_diff'
                        },
                    'Rate (O2, gas phase)': {
                        'x': 'processed_data/O2_gas_time_diff',
                        'y': 'processed_data/O2_gas_data_diff'
                        },
                    'Rate (H2) - Smoothed': {
                        'x': 'processed_data/H2_time_diff',
                        'y': 'processed_data/H2_data_diff_smoothed'
                        },
                    'Rate (O2) - Smoothed': {
                        'x': 'processed_data/O2_time_diff',
                        'y': 'processed_data/O2_data_diff_smoothed'
                        },
                    'Rate fit (H2)': {
                        'x': 'processed_data/flexible_diff_time',
                        'y': 'processed_data/H2_fit_rate'
                        },
                    'Rate fit (O2)': {
                        'x': 'processed_data/flexible_diff_time',
                        'y': 'processed_data/O2_fit_rate'
                        },
                    'Rate poly fit (H2)': {
                        'x': 'processed_data/H2_time_diff',
                        'y': 'processed_data/H2_poly_fit_diff'
                        },
                    'Rate poly fit (O2)': {
                        'x': 'processed_data/O2_time_diff',
                        'y': 'processed_data/O2_poly_fit_diff'
                        },
                    'Rate poly fit (H2, gas phase)': {
                        'x': 'processed_data/H2_gas_time_diff',
                        'y': 'processed_data/H2_gas_poly_fit_diff'
                        },
                    'Rate poly fit (O2, gas phase)': {
                        'x': 'processed_data/O2_gas_time_diff',
                        'y': 'processed_data/O2_gas_poly_fit_diff'
                        },
                    'Rate flexible fit (H2)': {
                        'x': 'processed_data/flexible_diff_time',
                        'y': 'processed_data/H2_flexible_fit_rate'
                        },
                    'Rate flexible fit (O2)': {
                        'x': 'processed_data/flexible_diff_time',
                        'y': 'processed_data/O2_flexible_fit_rate'
                    },
                },
    'kinetic_results_instructions': {
        'O2 rate constant': {'Value': 'processed_data/O2_rate_constant',
                             'Unit': 'Rate constant / 1/s'},   
        'H2 rate constant': {'Value': 'processed_data/H2_rate_constant',
                             'Unit': 'Rate constant / 1/s'},
        'O2 flexible rate constant': {'Value': 'processed_data/O2_flexible_rate_constant',
                             'Unit': 'Rate constant / 1/s'},   
        'H2 flexible rate constant': {'Value': 'processed_data/H2_flexible_rate_constant',
                             'Unit': 'Rate constant / 1/s'},
        'O2 rate constant (fixed)': {'Value': 'processed_data/O2_fixed_rate_constant',
                             'Unit': 'Rate constant / 1/s'},   
        'H2 rate constant (fixed)': {'Value': 'processed_data/H2_fixed_rate_constant',
                             'Unit': 'Rate constant / 1/s'},
        'H2 max rate (polyfit)': {'Value': 'processed_data/H2_max_rate',
                        'Unit': 'Rate / umol L^-1 s^-1'},
        'O2 max rate (polyfit)': {'Value': 'processed_data/O2_max_rate',
                        'Unit': 'Rate / umol L^-1 s^-1'},
        'H2 max rate (polyfit, gas phase)': {'Value': 'processed_data/H2_gas_max_rate',
                        'Unit': 'Rate / umol L^-1 s^-1'},
        'O2 max rate (polyfit, gas phase)': {'Value': 'processed_data/O2_gas_max_rate',
                        'Unit': 'Rate / umol L^-1 s^-1'},
        'H2 max rate (fit)': {'Value': 'processed_data/H2_fit_max_rate',
                              'Unit': 'Rate / umol L^-1 s^-1'},
        'O2 max rate (fit)': {'Value': 'processed_data/O2_fit_max_rate',
                              'Unit': 'Rate / umol L^-1 s^-1'},
        'H2 max rate (flexible)': {'Value': 'processed_data/H2_flexible_fit_max_rate',
                        'Unit': 'Rate / umol L^-1 s^-1'},
        'O2 max rate (flexible)': {'Value': 'processed_data/O2_flexible_fit_max_rate',
                        'Unit': 'Rate / umol L^-1 s^-1'},
    },
}

