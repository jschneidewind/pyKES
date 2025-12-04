import streamlit as st
import plotly.graph_objects as go
import numpy as np
from collections import defaultdict
import pprint as pp

from pyKES.utilities.resolve_attributes import resolve_experiment_attributes

st.set_page_config(
    page_title="Analysis Results Visualization", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and filename display
col_title, col_filename = st.columns([3, 1])
with col_title:
    st.title("Analysis Results Visualization")
with col_filename:
    if 'hdf5_filename' in st.session_state and st.session_state.hdf5_filename:
        st.markdown(f"<p style='text-align: right; font-size: 0.8em; color: gray; margin-top: 1.5em;'>{st.session_state.hdf5_filename}</p>", unsafe_allow_html=True)

# Function to handle checkbox changes for subsets
def update_subset_selection(subset_key, experiments_in_subset, value):
    if value:
        # Add all experiments in subset
        for exp_name in experiments_in_subset:
            if exp_name not in st.session_state.selected_subsets:
                st.session_state.selected_subsets[subset_key] = experiments_in_subset
    else:
        # Remove subset
        if subset_key in st.session_state.selected_subsets:
            del st.session_state.selected_subsets[subset_key]

# Function to handle multiselect changes
def sync_from_multiselect():
    selected_subset_keys = st.session_state.multiselect_widget
    # Rebuild selected_subsets dict based on selection
    new_selected = {}
    for subset_key in selected_subset_keys:
        if subset_key in st.session_state.all_subsets:
            new_selected[subset_key] = st.session_state.all_subsets[subset_key]
    st.session_state.selected_subsets = new_selected

if st.session_state.experimental_dataset is not None:
    
    experimental_dataset = st.session_state.experimental_dataset
    
    # Initialize session state
    if 'selected_subsets' not in st.session_state:
        st.session_state.selected_subsets = {}  # Dict: subset_key -> list of exp_names
    if 'all_subsets' not in st.session_state:
        st.session_state.all_subsets = {}
    if 'selected_analysis_results' not in st.session_state:
        st.session_state.selected_analysis_results = []
    if 'selected_x_axis' not in st.session_state:
        st.session_state.selected_x_axis = 'Experiments'

    # Get group_mapping and plotting_instruction from dataset
    group_mapping = experimental_dataset.group_mapping
    plotting_instruction = experimental_dataset.plotting_instruction
    
    # Get kinetic results instructions
    if 'kinetic_results_instructions' not in plotting_instruction:
        st.error("No 'kinetic_results_instructions' found in plotting_instruction")
        st.stop()
    
    kinetic_results_instructions = plotting_instruction['kinetic_results_instructions']
    
    # Filter active experiments only
    active_experiments = {}
    for exp_name, exp_data in experimental_dataset.experiments.items():
        try:
            is_active = resolve_experiment_attributes(
                {'Active': 'metadata/Active'}, 
                exp_data, 
                mode='permissive'
            )
            if is_active.get('Active', False):
                active_experiments[exp_name] = exp_data
        except (ValueError, KeyError, AttributeError):
            # If Active field not found or can't be resolved, skip experiment
            continue
    
    # Group experiments and create subsets
    subsets_by_group = defaultdict(list)  # group_name -> list of (subset_label, [exp_names])
    st.session_state.all_subsets = {}
    
    for group_name in group_mapping.keys():

        experiments_in_group = []
        
        # Find all active experiments in this group
        for exp_name, exp_data in active_experiments.items():
            if exp_data.group == group_name:
                experiments_in_group.append((exp_name, exp_data))
        
        if not experiments_in_group:
            continue
        
        # For Reference group, create one subset with all experiments
        if group_name == 'Reference' or group_mapping[group_name] is None:
            subset_label = f"Reference (n={len(experiments_in_group)})"
            subset_key = f"{group_name}||Reference"
            exp_names = [exp_name for exp_name, _ in experiments_in_group]
            subsets_by_group[group_name].append((subset_label, exp_names))
            st.session_state.all_subsets[subset_key] = exp_names
        else:
            # Group by metadata value
            metadata_path = group_mapping[group_name]
            subsets_by_metadata = defaultdict(list)
            
            for exp_name, exp_data in experiments_in_group:

                try:
                    metadata_value = resolve_experiment_attributes(
                        {'value': metadata_path}, 
                        exp_data, 
                        mode='permissive'
                    )
                    
                    if 'value' in metadata_value:
                        value = metadata_value['value']
                        subsets_by_metadata[value].append(exp_name)
                except (ValueError, KeyError, AttributeError):
                    continue
            
            # Create subset labels
            for metadata_value, exp_names in sorted(subsets_by_metadata.items()):
                subset_label = f"{metadata_value} (n={len(exp_names)})"
                subset_key = f"{group_name}||{metadata_value}"
                subsets_by_group[group_name].append((subset_label, exp_names))
                st.session_state.all_subsets[subset_key] = exp_names
    
    # Create two-column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Group Selection")
        
        # Iterate through each group
        for group_name in sorted(subsets_by_group.keys()):
            subsets = subsets_by_group[group_name]
            
            # Create group title
            if group_name == 'Reference' or group_mapping[group_name] is None:
                group_title = f"Reference (n={sum(len(exps) for _, exps in subsets)})"
            else:
                # Extract column name from metadata path
                metadata_path = group_mapping[group_name]
                column_name = metadata_path.split('/', 1)[-1] if '/' in metadata_path else metadata_path
                total_count = sum(len(exps) for _, exps in subsets)
                group_title = f"{column_name} (n={total_count})"
            
            with st.expander(group_title, expanded=False):
                # Display each subset in the group
                for subset_label, exp_names in sorted(subsets, key=lambda x: x[0]):
                    subset_key = None
                    for key, exps in st.session_state.all_subsets.items():
                        if exps == exp_names:
                            subset_key = key
                            break
                    
                    if subset_key is None:
                        continue
                    
                    is_selected = subset_key in st.session_state.selected_subsets
                    
                    # Create checkbox for subset
                    st.checkbox(
                        subset_label,
                        value=is_selected,
                        key=f"checkbox_{subset_key}",
                        on_change=update_subset_selection,
                        args=(subset_key, exp_names, not is_selected)
                    )
    
    with col2:
        st.header("Visualization Panel")
        
        # Create sub-columns for controls
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
        
        with ctrl_col1:
            # Multiselect for selected subsets overview
            all_subset_keys = list(st.session_state.all_subsets.keys())
            # Create display labels for multiselect
            subset_display_labels = {}
            for key in all_subset_keys:
                group_name, value = key.split('||', 1)
                exp_names = st.session_state.all_subsets[key]
                if value == 'Reference':
                    display_label = f"Reference (n={len(exp_names)})"
                else:
                    display_label = f"{group_name}: {value} (n={len(exp_names)})"
                subset_display_labels[key] = display_label

            #default=list(st.session_state.selected_subsets.keys()),
                        
            st.multiselect(
                "Selected Experiments",
                options=all_subset_keys,
                default=None,
                format_func=lambda x: subset_display_labels.get(x, x),
                key="multiselect_widget",
                on_change=sync_from_multiselect
            )
        
        with ctrl_col2:
            # X-axis selection
            x_axis_options = ['Experiments']
            x_axis_group_mapping = {}  # Map display name to group name
            for group_name, metadata_path in group_mapping.items():
                if metadata_path is not None:
                    column_name = metadata_path.split('/', 1)[-1] if '/' in metadata_path else metadata_path
                    x_axis_options.append(column_name)
                    x_axis_group_mapping[column_name] = group_name
            
            st.session_state.selected_x_axis = st.selectbox(
                'Select X-Axis',
                x_axis_options,
                index=0
            )
        
        with ctrl_col3:
            # Analysis results selection
            available_results = list(kinetic_results_instructions.keys())
            
            # Initialize with first result if empty
            if not st.session_state.selected_analysis_results and available_results:
                st.session_state.selected_analysis_results = [available_results[0]]
            
            # Use the multiselect with key for automatic state management
            selected_results = st.multiselect(
                "Select Analysis Results",
                options=available_results,
                default=None,
                key="analysis_results_multiselect"
            )
            
            # Update session state with the current selection
            st.session_state.selected_analysis_results = selected_results
        
        # Create the plot
        fig = go.Figure()
        
        # Determine y-axis label (use last selected result's unit)
        y_axis_label = "Value"
        if st.session_state.selected_analysis_results:
            last_result = st.session_state.selected_analysis_results[-1]
            if last_result in kinetic_results_instructions:
                y_axis_label = kinetic_results_instructions[last_result].get('Unit', 'Value')
        
        # Determine x-axis label
        x_axis_label = st.session_state.selected_x_axis
        
        # Collect data for plotting
        plot_data = []  # List of (subset_key, subset_label, exp_name, exp_data, x_value, result_name, y_value)
        
        for subset_key, exp_names in st.session_state.selected_subsets.items():
            group_name, subset_value = subset_key.split('||', 1)
            
            for exp_name in exp_names:
                if exp_name not in active_experiments:
                    continue
                
                exp_data = active_experiments[exp_name]
                
                # Determine x-value
                if st.session_state.selected_x_axis == 'Experiments':
                    x_value = exp_name
                else:
                    # Extract group name from x-axis selection using mapping
                    x_group_name = x_axis_group_mapping.get(st.session_state.selected_x_axis)
                    if x_group_name and x_group_name in group_mapping and group_mapping[x_group_name] is not None:
                        metadata_path = group_mapping[x_group_name]
                        try:
                            metadata_value = resolve_experiment_attributes(
                                {'value': metadata_path}, 
                                exp_data, 
                                mode='permissive'
                            )
                            x_value = metadata_value.get('value', None)
                        except (ValueError, KeyError, AttributeError):
                            x_value = None
                    else:
                        x_value = None
                
                if x_value is None:
                    continue
                
                # Resolve analysis results
                for result_name in st.session_state.selected_analysis_results:
                    if result_name not in kinetic_results_instructions:
                        continue
                    
                    result_config = kinetic_results_instructions[result_name]
                    value_path = result_config.get('Value', None)
                    
                    if value_path is None:
                        continue
                    
                    try:
                        result_value = resolve_experiment_attributes(
                            {'value': value_path}, 
                            exp_data, 
                            mode='permissive'
                        )
                        y_value = result_value.get('value', None)
                        
                        if y_value is not None:
                            plot_data.append((subset_key, subset_value, exp_name, exp_data, x_value, result_name, y_value))
                    except (ValueError, KeyError, AttributeError):
                        continue
        
        # Sort plot_data by x_value for line connections
        if st.session_state.selected_x_axis == 'Experiments':
            # For experiment mode, sort by experiment name
            plot_data.sort(key=lambda x: x[2])  # Sort by exp_name
        else:
            # For metadata mode, sort by x_value
            plot_data.sort(key=lambda x: (x[0], x[5], x[4]))  # Sort by subset_key, result_name, x_value
        
        # Group by subset and result for plotting
        from itertools import groupby
        
        for (subset_key, result_name), group in groupby(plot_data, key=lambda x: (x[0], x[5])):
            group_data = list(group)
            
            if not group_data:
                continue
            
            # Get subset info
            _, subset_value, _, first_exp_data, _, _, _ = group_data[0]
            exp_color = first_exp_data.color
            
            # Extract data for plotting
            x_values = []
            y_values = []
            exp_names = []
            
            for _, _, exp_name, exp_data, x_value, _, y_value in group_data:
                x_values.append(x_value)
                y_values.append(y_value)
                exp_names.append(exp_name)
            
            # Create hover template
            hover_template = (
                "<b>%{text}</b><br>"
                f"{x_axis_label}: %{{x}}<br>"
                f"{result_name}: %{{y:.2e}}<br>"
                "<extra></extra>"
            )
            
            # Create trace name
            group_name = subset_key.split('||')[0]
            if subset_value == 'Reference':
                trace_name = f"Reference - {result_name}"
            else:
                trace_name = f"{group_name}: {subset_value} - {result_name}"
            
            # Add trace (dots + lines)
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                name=trace_name,
                mode='lines+markers',
                marker=dict(color=exp_color, size=8),
                line=dict(color=exp_color, width=2),
                text=exp_names,
                hovertemplate=hover_template,
                hoverlabel=dict(font_color=exp_color)
            ))
        
        # Update layout
        fig.update_layout(
            title="Analysis Results",
            xaxis_title=x_axis_label,
            yaxis_title=y_axis_label,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_color='black'
            ),
            showlegend=True,
            legend=dict(
                itemclick="toggleothers",
                itemdoubleclick="toggle"
            ),
            height=600
        )
        
        # For "Experiments" mode, set categorical x-axis
        if st.session_state.selected_x_axis == 'Experiments':
            # Get unique experiment names in order
            unique_exp_names = []
            seen = set()
            for item in plot_data:
                exp_name = item[2]
                if exp_name not in seen:
                    unique_exp_names.append(exp_name)
                    seen.add(exp_name)
            
            fig.update_xaxes(
                type='category',
                categoryorder='array',
                categoryarray=unique_exp_names
            )
        
        # Update axis colors and format y-axis in scientific notation
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.2)')
        fig.update_yaxes(
            gridcolor='rgba(255,255,255,0.2)',
            tickformat='.2e'  # Scientific notation with 2 decimal places
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metadata for selected experiments
        if st.session_state.selected_subsets:
            st.markdown("#### Selected Experiments Metadata")
            
            # Collect all selected experiment names
            all_selected_exp_names = []
            for exp_names in st.session_state.selected_subsets.values():
                all_selected_exp_names.extend(exp_names)
            
            if not experimental_dataset.overview_df.empty:
                mask = experimental_dataset.overview_df['Experiment'].isin(all_selected_exp_names)
                filtered_df = experimental_dataset.overview_df[mask]
                
                if not filtered_df.empty:
                    st.dataframe(filtered_df, use_container_width=True)
                else:
                    st.info("No overview data available for selected experiments.")
            else:
                # Display basic info if no overview_df
                st.markdown("**Selected Experiments:**")
                for exp_name in all_selected_exp_names:
                    if exp_name in active_experiments:
                        exp_data = active_experiments[exp_name]
                        st.markdown(f"- **{exp_name}** (Group: {exp_data.group}, Color: {exp_data.color})")
        
    # Add explainer section at the bottom
    st.markdown("---")
    st.header("How to Use This Tool")
    
    with st.expander("📚 Detailed Explanation", expanded=False):
        st.markdown("""
        ### Understanding the Analysis Results Visualization Tool
        
        This tool helps visualize kinetic analysis results from experiments organized by groups.
        
        #### 1. Group Selection (Left Panel)
        - Experiments are organized by their group attribute
        - Within each group, experiments with the same metadata values are grouped into subsets
        - Only experiments marked as Active (metadata/Active = True) are included
        - Select checkboxes to add entire subsets to the visualization
        - The number in parentheses (n=X) shows how many experiments are in each subset
        
        #### 2. Visualization Panel (Right Panel)
        - **Selected Experiments**: Shows selected subsets (not individual experiments)
        - **Select X-Axis**: Choose what to plot on the x-axis
          - "Experiments": Show individual experiments side-by-side
          - Group options: Plot against metadata values (e.g., "Intensity / Irradiance [mW/cm2]")
        - **Select Analysis Results**: Choose which kinetic parameters to visualize
        
        #### 3. Understanding the Plot
        - Each subset is shown in its designated color
        - Individual experiments appear as dots
        - Experiments within the same subset are connected by lines
        - Hover over dots to see experiment name and values
        - Multiple analysis results can be compared simultaneously
        - Y-axis unit is determined by the last selected analysis result
        
        #### 4. Experiment Metadata
        The table at the bottom shows detailed metadata for all experiments in selected subsets.
        
        #### Tips
        - Use subset selection for quick comparison of experimental conditions
        - Switch between "Experiments" and metadata x-axes to view different perspectives
        - Compare multiple kinetic parameters by selecting several analysis results
        - Colors are consistent across different groups for easy identification
        """)
    
    st.markdown("---")
    st.caption("pyKES Analysis Results Visualization | Powered by Streamlit")

else:
    st.info("Please upload a HDF5 file on the Home page first.")
