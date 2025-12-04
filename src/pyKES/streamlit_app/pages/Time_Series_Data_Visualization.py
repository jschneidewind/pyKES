import streamlit as st
import plotly.graph_objects as go
import numpy as np

from pyKES.utilities.resolve_attributes import resolve_experiment_attributes

st.set_page_config(
    page_title="Time-Series Data Visualization", 
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and filename display
col_title, col_filename = st.columns([3, 1])
with col_title:
    st.title("Time-Series Data Visualization")
with col_filename:
    if st.session_state.hdf5_filename:
        st.markdown(f"<p style='text-align: right; font-size: 0.8em; color: gray; margin-top: 1.5em;'>{st.session_state.hdf5_filename}</p>", unsafe_allow_html=True)

# Function to handle checkbox changes
def update_selection(exp_name, value):
    if value and exp_name not in st.session_state.selected_experiments:
        st.session_state.selected_experiments.append(exp_name)
    elif not value and exp_name in st.session_state.selected_experiments:
        st.session_state.selected_experiments.remove(exp_name)

# Function to handle multiselect changes
def sync_from_multiselect():
    # This gets called when the multiselect value changes
    selected_names = st.session_state.multiselect_widget
    st.session_state.selected_experiments = selected_names

if st.session_state.experimental_dataset is not None:
    
    experimental_dataset = st.session_state.experimental_dataset
    
    # Initialize session state
    if 'selected_experiments' not in st.session_state:
        st.session_state.selected_experiments = []
    if 'selected_plot_types' not in st.session_state:
        st.session_state.selected_plot_types = []

    # Get group_mapping and plotting_instruction from dataset
    group_mapping = experimental_dataset.group_mapping
    plotting_instruction = experimental_dataset.plotting_instruction['time_series_instructions']
    
    # Group experiments by their group attribute
    experiments_by_group = {}
    for exp_name, exp_data in experimental_dataset.experiments.items():
        group = exp_data.group
        if group not in experiments_by_group:
            experiments_by_group[group] = []
        experiments_by_group[group].append((exp_name, exp_data))
    
    # Create two-column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Group Selection")
        
        # Iterate through each group
        for group_name in sorted(experiments_by_group.keys()):
            experiments_in_group = experiments_by_group[group_name]
            
            with st.expander(f"{group_name} (n={len(experiments_in_group)})", expanded=False):
                
                # Display each experiment in the group
                for exp_name, exp_data in sorted(experiments_in_group, key=lambda x: x[0]):
                    
                    is_selected = exp_name in st.session_state.selected_experiments
                    
                    # Build checkbox label with metadata
                    checkbox_label = exp_name
                    
                    # Add metadata if group is not 'Reference' and has mapping
                    if group_name in group_mapping and group_mapping[group_name] is not None:
                        metadata_path = group_mapping[group_name]
                        try:
                            # Resolve the metadata value
                            metadata_value = resolve_experiment_attributes(
                                {group_name: metadata_path}, 
                                exp_data, 
                                mode='permissive'
                            )
                            if group_name in metadata_value:
                                checkbox_label = f"{exp_name} ({metadata_value[group_name]})"
                        except (ValueError, KeyError, AttributeError):
                            # If resolution fails, just use experiment name
                            pass
                    
                    # Create checkbox for experiment
                    st.checkbox(
                        checkbox_label,
                        value=is_selected,
                        key=f"checkbox_{exp_name}",
                        on_change=update_selection,
                        args=(exp_name, not is_selected)
                    )
    
    with col2:
        st.header("Visualization Panel")
        
        # Create sub-columns for controls
        ctrl_col1, ctrl_col2 = st.columns(2)
        
        with ctrl_col1:
            # Multiselect for selected experiments overview
            all_experiment_names = sorted(experimental_dataset.experiments.keys())
            st.multiselect(
                "Selected Experiments",
                options=all_experiment_names,
                default=st.session_state.selected_experiments,
                key="multiselect_widget",
                on_change=sync_from_multiselect
            )
        
        with ctrl_col2:
            # Multiselect for plot types based on plotting_instruction
            if plotting_instruction:
                available_plots = list(plotting_instruction.keys())
                st.session_state.selected_plot_types = st.multiselect(
                    "Select Data to Display",
                    available_plots,
                    default=None
                )
            else:
                st.warning("No plotting instructions defined in dataset.")
        
        # Create the plot
        fig = go.Figure()
        
        # Add traces for selected experiments
        for exp_name in st.session_state.selected_experiments:
            if exp_name not in experimental_dataset.experiments:
                continue
                
            exp_data = experimental_dataset.experiments[exp_name]
            exp_color = exp_data.color
            
            # Resolve all plotting instructions for this experiment
            try:
                resolved_plots = resolve_experiment_attributes(
                    plotting_instruction, 
                    exp_data, 
                    mode='permissible'
                )
            except ValueError as e:
                st.warning(f"Could not resolve plotting data for {exp_name}: {str(e)}")
                continue
            
            # Add traces for each selected plot type
            for plot_type in st.session_state.selected_plot_types:
                if plot_type not in resolved_plots:
                    continue
                
                plot_data = resolved_plots[plot_type]
                
                
                x_data = plot_data['x']
                y_data = plot_data['y']
                
                # Create hover template
                hover_template = (
                    f"<b>{exp_name} - {plot_type}</b><br>"
                    "X: %{x}<br>"
                    "Y: %{y}<br>"
                    "<extra></extra>"
                )
                
                # Determine line style based on plot type
                # Use markers for raw data, lines for smoothed/fitted data
                if 'raw' in plot_type.lower() or 'rate' in plot_type.lower() and not 'smoothed' in plot_type.lower():
                    mode = 'markers'
                    marker = dict(color=exp_color, size=5, opacity=0.7)
                    line = None
                else:
                    mode = 'lines'
                    marker = None
                    line = dict(color=exp_color, width=2)
                
                # Add trace
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    name=f"{exp_name} - {plot_type}",
                    mode=mode,
                    marker=marker,
                    line=line,
                    hovertemplate=hover_template,
                    hoverlabel=dict(font_color=exp_color)
                ))
        
        # Update layout
        fig.update_layout(
            title="Time-Series Data Visualization",
            xaxis_title="Time (s)",
            yaxis_title="Value",
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
        
        # Update axis colors
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.2)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.2)')
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metadata for selected experiments
        if st.session_state.selected_experiments:
            st.markdown("#### Selected Experiments Metadata")
            
            if not experimental_dataset.overview_df.empty:
                mask = experimental_dataset.overview_df['Experiment'].isin(st.session_state.selected_experiments)
                filtered_df = experimental_dataset.overview_df[mask]
                
                if not filtered_df.empty:
                    st.dataframe(filtered_df, use_container_width=True)
                else:
                    st.info("No overview data available for selected experiments.")
            else:
                # Display basic info if no overview_df
                st.markdown("**Selected Experiments:**")
                for exp_name in st.session_state.selected_experiments:
                    exp_data = experimental_dataset.experiments[exp_name]
                    st.markdown(f"- **{exp_name}** (Group: {exp_data.group}, Color: {exp_data.color})")
        
    # Add explainer section at the bottom
    st.markdown("---")
    st.header("How to Use This Tool")
    
    with st.expander("📚 Detailed Explanation", expanded=False):
        st.markdown("""
        ### Understanding the Time-Series Data Visualization Tool
        
        This tool helps visualize time-series data from experiments organized by groups.
        
        #### 1. Group Selection (Left Panel)
        - Experiments are organized by their group attribute
        - Click on a group to expand and see all experiments in that group
        - For non-reference groups, relevant metadata values are shown in parentheses
        - Select checkboxes to add experiments to the visualization
        
        #### 2. Visualization Panel (Right Panel)
        - **Selected Experiments**: Shows and allows manual selection/deselection of experiments
        - **Select Data to Display**: Choose which data types to plot (based on available plotting instructions)
        
        #### 3. Understanding the Plot
        - Each experiment is shown in its designated color
        - Different data types (raw, smoothed, fitted) are shown with different styles
        - Hover over data points to see exact values
        - Click legend items once to hide/show that trace
        - Double-click legend items to show only that trace
        
        #### 4. Experiment Metadata
        The table at the bottom shows detailed metadata for all selected experiments.
        
        #### Tips
        - Use the multiselect box for quick bulk selection/deselection
        - Compare different processing methods by selecting multiple data types
        - Use the same colors across different views for easy experiment identification
        """)
    
    st.markdown("---")
    st.caption("pyKES Time-Series Visualization | Powered by Streamlit")

else:
    st.info("Please upload a HDF5 file on the Home page first.")
