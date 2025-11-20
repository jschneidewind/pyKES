import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
import numpy as np


def get_experiment_group(exp_name):
    # Remove the last number and any trailing dash
    return re.sub(r'-?\d+$', '', exp_name)

def select_reference_conditions(df):

    acol1, acol2, acol3, acol4= st.columns(4)

    with acol1:
        # Filter by irradiation intensity concentration
        intensities = sorted(df['Power output [W/m^2]'].unique())
        selected_intensity = st.selectbox(
            'Irradiation intensity [W/m^2]',
            intensities,
            index=list(intensities).index(6.637) if 6.637 in intensities else 0
        )
    
    with acol2:
        # Filter by Ru concentration
        ru_concentrations = sorted(df['c([Ru(bpy(3]Cl2) [M]'].unique())
        selected_ru = st.selectbox(
            '[Ru] Catalyst Concentration [M]',
            ru_concentrations,
            index=list(ru_concentrations).index(1e-5) if 1e-5 in ru_concentrations else 0
        )
    
    with acol3:
        # Filter by persulfate concentration
        persulfate_concentrations = sorted(df['c(Na2S2O8) [M]'].unique())
        selected_persulfate = st.selectbox(
            'Persulfate Concentration [M]',
            persulfate_concentrations,
            index=list(persulfate_concentrations).index(6e-3) if 6e-3 in persulfate_concentrations else 0
        )
    
    with acol4:
        # Filter by pH
        ph_values = sorted(df['pH [-]'].unique())
        selected_ph = st.selectbox(
            'pH Value',
            ph_values,
            index=list(ph_values).index(9.6) if 9.6 in ph_values else 0
        )

    reference_conditions = {
        'Power output [W/m^2]': selected_intensity,
        'c([Ru(bpy(3]Cl2) [M]': selected_ru,
        'c(Na2S2O8) [M]': selected_persulfate,
        'pH [-]':selected_ph,
        }
    
    return reference_conditions


def classify_experiments(df, reference_conditions, tol=1e-8):
    
    sets = []  # To store the assigned group for each experiment
    deviations_list = []  # To store which parameters deviate for reporting
    
    # Iterate over each row
    for _, row in df.iterrows():
        deviations = []

        for parameter, reference_value in reference_conditions.items():
            # Use np.isclose to compare float values
            if not np.isclose(row[parameter], reference_value, atol=tol):
                deviations.append(parameter)
            
        deviations_list.append(deviations)

        # Assign group based on how many parameters deviate
        if len(deviations) == 0:
            sets.append("Reference")
        elif len(deviations) == 1:
            sets.append(f"{deviations[0]}")
        else:
            sets.append("Other")
    
    df = df.copy()
    df['Deviation Parameters'] = deviations_list
    df['Set'] = sets
    return df

# Function to handle checkbox changes
def update_selection(exp_id, value):
    if value and exp_id not in st.session_state.selected_groups:
        st.session_state.selected_groups.append(exp_id)
    elif not value and exp_id in st.session_state.selected_groups:
        st.session_state.selected_groups.remove(exp_id)

# Function to handle multiselect changes
def sync_from_multiselect():
    # This gets called when the multiselect value changes
    selected_ids = st.session_state.multiselect_widget
    st.session_state.selected_groups = selected_ids


def create_visualization(data, selected_outcome, x_axis_column = 'group'):
    """
    Creates a visualization of experimental outcomes using Plotly.
    This function generates an interactive scatter plot where experiments are grouped 
    and displayed with vertical lines representing the range of values within each group.
    Each data point represents an individual experiment result, with hover information 
    showing the experiment name and value.
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing experimental data with at least the following columns:
        - 'Experiment': Name/ID of each experiment
        - Column matching selected_outcome: Values to be plotted
    selected_outcome : str
        Name of the column in data to be visualized
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure showing experiment results grouped by experiment type,
        with scatter points for individual values and vertical lines showing the range
        within each group.
    Notes
    -----
    - Groups are determined by the get_experiment_group function (not shown)
    - Error bars (vertical lines) are only displayed for groups with multiple data points
    - Hover information includes experiment name and exact value
    """
    
    # Create figure
    fig = go.Figure()
    
    # Add vertical lines for each group
    for i, (group, group_data) in enumerate(data.groupby('group')):
    
        y_values = group_data[selected_outcome]

        if x_axis_column == 'group':
            x_values = [i] * len(y_values)
            x_value = i
        else:
            x_values = group_data[x_axis_column]
            x_value = x_values.iloc[0]
            
        # Add vertical line
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers',
            name=group,
            text=group_data['Experiment'],  # Create array of experiment names
            hovertemplate=(
            "Experiment: %{text}<br>" +
            f"Value: %{{y:.3f}}<br>"
        )
        ))
        
        # Add error bars
        if len(y_values) > 1:
            fig.add_trace(go.Scatter(
                x=[x_value, x_value],
                y=[min(y_values), max(y_values)],
                mode='lines',
                showlegend=False,
                line=dict(color='gray', width=1),
                hoverinfo='skip'
            ))

    # Update layout
    fig.update_layout(
        title=f'{selected_outcome} by Experiment Group',
        xaxis_title=x_axis_column,
        yaxis_title=selected_outcome,
        xaxis={**dict()} if x_axis_column != 'group' else dict(
        tickmode='array',
        ticktext=list(data.groupby('group').groups.keys()),
        tickvals=list(range(len(data.groupby('group'))))
        ),
        showlegend=False,
        hovermode='closest'
    )
    
    return fig

st.set_page_config(
    page_title="HTE Data Visualization", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("Plotting Rates")

if st.session_state.experimental_dataset is not None:
    df = st.session_state.experimental_dataset.overview_df
    df['group'] = df['Experiment'].apply(get_experiment_group)

    if 'selected_groups' not in st.session_state:
        st.session_state.selected_groups = []
    if 'selected_x_axis' not in st.session_state:
        st.session_state.selected_x_axis = 'group'
    if 'selected_outcome_group' not in st.session_state:
        st.session_state.selected_outcome_group = 'max rate ydiff'

    st.header("Select Reference Conditions")

    reference_conditions = select_reference_conditions(df)

    experiment_sets = classify_experiments(df, reference_conditions)

    # Create columns for the layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Group Selection")

        for set_name, set_df in experiment_sets.groupby('Set'):

            group_counts = set_df.groupby('group').size().rename("Total Experiments")
            
            # Drop duplicate base groups so only one representative per group remains
            simple_df = set_df.drop_duplicates(subset='group')
            simple_df = simple_df.merge(group_counts, left_on='group', right_index=True, how='left')

            with st.expander(f"{set_name}", expanded=False):
                for index, row in simple_df.iterrows():

                    group = row['group']
                    deviations = row['Deviation Parameters']
                    total = row['Total Experiments']
                    
                    is_selected = group in st.session_state.selected_groups

                    if len(deviations) == 0:
                        # Reference experiments
                        checkbox_name = f"{group} (n = {total})"
                    elif len(deviations) == 1:
                        # Experiments where one parameter has been varied
                        parameter_value = row[deviations[0]]
                        checkbox_name = f"{group} ({parameter_value}, n = {total})"
                    else:
                        # Experiments where multiple parameters have been varied
                        checkbox_name = f"{group} ("
                        for deviation in deviations:
                            parameter_value = row[deviation]
                            checkbox_name += f"{deviation} = {parameter_value}, "
                        checkbox_name += f"n = {total})"

                    checkbox = st.checkbox(
                        checkbox_name,
                        value=is_selected,
                        key=f"checkbox_{group}_{np.random.randint(0, 100000)}",
                        on_change=update_selection,
                        args=(group, not is_selected))
                    
    with col2:
        st.header("Visualization Panel")

        gcol1, gcol2, gcol3 = st.columns(3)

        with gcol1:
            unique_groups = sorted(df['group'].unique())

            st.multiselect(
                "Selected Groups",
                options=unique_groups,
                default=st.session_state.selected_groups,
                key="multiselect_widget",
                on_change=sync_from_multiselect
                )
            
        with gcol2:
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            x_axis_options = ['group'] + numeric_columns
            
            st.session_state.selected_x_axis = st.selectbox(
                'Select X-Axis Column',
                x_axis_options,
                index=0  # Default to grouping by experiment
            )

        with gcol3:
            st.session_state.selected_outcome_group = st.radio(
                'Select analysis outcome',  # Label
                ['rate', 'max rate', 'max rate ydiff', 'rate constant'], # List of options
                key = 2,
                index = 2
            )

        group_mask = df['group'].isin(st.session_state.selected_groups)
        group_filtered_df = df[group_mask]

        if not group_filtered_df.empty:
            group_fig = create_visualization(group_filtered_df, st.session_state.selected_outcome_group, 
                                             x_axis_column=st.session_state.selected_x_axis)
            st.plotly_chart(group_fig)
        else:
            st.warning('No data available for the selected filters.')

        st.markdown("#### Displayed Experiments")

        st.dataframe(group_filtered_df, use_container_width=True)
        
    # Add explainer section at the bottom of the page
    st.markdown("---")
    st.header("How to Use This Tool")
    
    with st.expander("📚 Detailed Explanation", expanded=False):
        st.markdown("""
        ### Understanding the Plotting Rates Tool
        
        This tool helps analyze and visualize high-throughput experimental data. Here's how to use it:
        
        #### 1. Reference Conditions
        At the top of the page, select your reference conditions for:
        - Irradiation intensity (W/m²)
        - Ruthenium catalyst concentration [M]
        - Persulfate concentration [M]
        - pH value
        
        These values define your baseline against which other experiments are compared.
        
        #### 2. Group Selection
        The left panel organizes experiments into sets:
        - **Reference**: Experiments matching all reference conditions
        - **Power output**, **Ru concentration**, etc.: Experiments where only one parameter deviates from reference
        - **Other**: Experiments with multiple deviating parameters
        
        Click checkboxes to select experiment groups for visualization.
        
        #### 3. Visualization Panel
        - Use the dropdown to manually select/deselect groups
        - Choose which parameter to display on the X-axis
        - Select which analysis outcome to plot
        
        #### 4. Understanding the Plot
        - Each dot represents a single experiment
        - Vertical lines show the range of values within a group
        - Hover over points to see exact experiment details
        
        #### 5. Displayed Experiments
        The table at the bottom shows all data for your selected experiments.
        """)
    
    st.markdown("---")
    st.caption("Developed by the HTE Photocatalysis Team | Last updated: March 2025")

else:
    st.info("Please upload a HDF5 file on the home page first.")

