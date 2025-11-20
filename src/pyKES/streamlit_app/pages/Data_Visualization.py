import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Time-Series Data Visualization", 
    page_icon=":microscope:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("Time-Series Data Visualization")

if st.session_state.experimental_dataset is not None:
    
    experimental_dataset = st.session_state.experimental_dataset

    # Create two columns with more space between experiment and data selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Experiment Selection")
        # Multi-select for experiments
        selected_experiments = st.multiselect(
            "Choose Experiments",
            sorted(list(experimental_dataset.experiments.keys())),
            default=list()
        )

    with col2:
        st.header("Data Types")
        # Plot type selection
        plot_types = st.multiselect(
            "Select Data to Display",
            ["Reaction (baseline corrected)",
            "Reaction Fit", 
            "Baseline",
            "LBC Fit",
            "Full (baseline corrected)",
            "Full",
            "Rate (raw data)",
            "Rate (smoothed)",
            "Rate (reaction fit)"
            ],
            default=["Reaction (baseline corrected)", 
                    "Reaction Fit"]
        )
    
    # Create plot
    fig = go.Figure()
    
    # Add traces for selected experiments and plot types
    for exp_name in selected_experiments:
        exp_data = experimental_dataset.experiments[exp_name]
        
        # Get experiment-specific color
        exp_color = getattr(exp_data.experiment_metadata, 'color', "#0ab5cc")  # Fallback to random color
        
        hover_template = "<b>%{fullData.name}</b><br>" + "X: %{x}<br>" + "Y: %{y}<br>" +"<extra></extra>"


        # Plotting loop
        if "Reaction (baseline corrected)" in plot_types:
            # Dots for raw data
            fig.add_trace(go.Scatter(
                x=exp_data.time_series_data.time_reaction, 
                y=exp_data.time_series_data.data_reaction,
                name=f"{exp_name} - Reaction (baseline corrected)",
                mode='markers',
                marker=dict(color=exp_color, size=5, opacity=0.7),
                hovertemplate = hover_template,
                hoverlabel=dict(font_color=exp_color),
            ))
        
        if "Reaction Fit" in plot_types:
            # Solid line for reaction fit
            fig.add_trace(go.Scatter(
                x=exp_data.time_series_data.time_reaction, 
                y=exp_data.time_series_data.y_fit,
                name=f"{exp_name} - Reaction Fit",
                mode='lines',
                line=dict(color=exp_color, width=3),
                hovertemplate = hover_template,
                hoverlabel=dict(font_color=exp_color)
            ))
        
        if "Baseline" in plot_types:
            # Dashed line for baseline
            fig.add_trace(go.Scatter(
                x=exp_data.time_series_data.full_x_values, 
                y=exp_data.time_series_data.baseline_y,
                name=f"{exp_name} - Baseline",
                mode='lines',
                line=dict(color=exp_color, width=2, dash='dash'),
                hovertemplate = hover_template,
                hoverlabel=dict(font_color=exp_color)
            ))
        
        if "LBC Fit" in plot_types:
            # Dotted line for LBC fit
            fig.add_trace(go.Scatter(
                x=exp_data.time_series_data.full_x_values, 
                y=exp_data.time_series_data.lbc_fit_y,
                name=f"{exp_name} - LBC Fit",
                mode='lines',
                line=dict(color=exp_color, width=2, dash='dashdot'),
                hovertemplate = hover_template,
                hoverlabel=dict(font_color=exp_color)
            ))

        if "Full (baseline corrected)" in plot_types:
            # Markers for full data (baseline corrected)
            fig.add_trace(go.Scatter(
                x=exp_data.time_series_data.full_x_values, 
                y=exp_data.time_series_data.full_y_corrected,
                name=f"{exp_name} - Full (baseline corrected)",
                mode='markers',
                marker=dict(color=exp_color, size=5, opacity=0.7),
                hovertemplate = hover_template,
                hoverlabel=dict(font_color=exp_color)
            ))

        if "Full" in plot_types:
            # Markers for full data
            fig.add_trace(go.Scatter(
                x=exp_data.time_series_data.time_full, 
                y=exp_data.time_series_data.data_full,
                name=f"{exp_name} - Full",
                mode='markers',
                marker=dict(color=exp_color, size=5, opacity=0.7),
                hovertemplate = hover_template,
                hoverlabel=dict(font_color=exp_color)
            ))

        if "Rate (raw data)" in plot_types:
            # Dots for raw data
            fig.add_trace(go.Scatter(
                x=exp_data.time_series_data.x_diff, 
                y=exp_data.time_series_data.y_diff,
                name=f"{exp_name} - Rate (raw data)",
                mode='markers',
                marker=dict(color=exp_color, size=5, opacity=0.7),
                hovertemplate=hover_template,
                hoverlabel=dict(font_color=exp_color)
            ))

        if "Rate (smoothed)" in plot_types:
            # Solid line for reaction fit
            fig.add_trace(go.Scatter(
                x=exp_data.time_series_data.x_diff, 
                y=exp_data.time_series_data.y_diff_smoothed,
                name=f"{exp_name} - Rate (smoothed)",
                mode='lines',
                line=dict(color=exp_color, width=1.5),
                hovertemplate=hover_template,
                hoverlabel=dict(font_color=exp_color)
            ))
        
        if "Rate (reaction fit)" in plot_types:
            # Solid line for reaction fit
            fig.add_trace(go.Scatter(
                x=exp_data.time_series_data.x_diff, 
                y=exp_data.time_series_data.y_diff_fit,
                name=f"{exp_name} - Rate (reaction fit)",
                mode='lines',
                line=dict(color=exp_color, width=1.5, dash='dashdot'),
                hovertemplate=hover_template,
                hoverlabel=dict(font_color=exp_color)
            ))
    
    # Update layout with dark theme
    fig.update_layout(
        title="HTE Data Visualization",
        xaxis_title="Time (s)",
        yaxis_title="O2 Concentration",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        hoverlabel=dict(
                    bgcolor="white",  # white background for hover label
                    font_size=14,    # font size for hover label
                    font_color='black'  # font color for hover label
                ),
        showlegend=True,
        legend=dict(
            itemclick="toggleothers",  # Click one trace to show only that one
            itemdoubleclick="toggle"),
    )

    # Update axis colors
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.2)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.2)')
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Display metadata for selected experiments
    st.header("Experiment Metadata")
    mask = experimental_dataset.overview_df['Experiment'].isin(selected_experiments)
    filtered_df = experimental_dataset.overview_df[mask]

    st.dataframe(filtered_df, use_container_width=True)

else:
    st.info("Please upload a HDF5 file on the home page first.")

