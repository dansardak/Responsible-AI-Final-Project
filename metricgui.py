import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Set page configuration
st.set_page_config(page_title="Image Quality Metrics Analyzer", layout="wide")

# Title and description
st.title("Image Quality Metrics Analyzer")
st.markdown("This app helps analyze various image quality metrics across different regions and objects.")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('fixed_metrics.csv')

metrics_df = load_data()

# Define metric columns and other useful lists
cols = ['Image', 'Object', 'Region', 'NIQE', 'BRISQUE', 'PIQE', 'MetaIQA', 'RankIQA', 'HyperIQA', 'Contrique', 'CNNIQA', 'TReS', 'CLIPIQA']
metric_cols = ['NIQE', 'BRISQUE', 'PIQE', 'MetaIQA', 'RankIQA', 'HyperIQA', 'Contrique', 'CNNIQA', 'TReS', 'CLIPIQA']

# Calculate average metrics for each region
region_averages = metrics_df.groupby('Region').agg({
    metric: 'mean' for metric in metric_cols
}).reset_index()

regions = list(region_averages['Region'])

# Sidebar for filtering
st.sidebar.header("Filters")
selected_regions = st.sidebar.multiselect("Select Regions", regions, default=regions)
selected_metrics = st.sidebar.multiselect("Select Metrics to Display", metric_cols, default=metric_cols[:3])

# Filter data based on selection
filtered_data = metrics_df[metrics_df['Region'].isin(selected_regions)]
filtered_averages = region_averages[region_averages['Region'].isin(selected_regions)]

# Main content
st.header("Data Overview")
with st.expander("Show Raw Data"):
    st.dataframe(filtered_data)

# Metrics summary
st.header("Metrics Summary by Region")
st.dataframe(filtered_averages[['Region'] + selected_metrics])

# Visualization section
st.header("Visualizations")

# Visualization type selector
viz_type = st.radio("Select Visualization Type", 
                   ["Bar Chart", "Box Plot", "Heatmap", "Scatter Plot"])

if viz_type == "Bar Chart":
    fig = px.bar(
        filtered_averages, 
        x='Region', 
        y=selected_metrics,
        barmode='group',
        title="Average Metrics by Region"
    )
    st.plotly_chart(fig, use_container_width=True)

elif viz_type == "Box Plot":
    # Reshape data for box plot
    melted_data = pd.melt(
        filtered_data, 
        id_vars=['Region'], 
        value_vars=selected_metrics,
        var_name='Metric', 
        value_name='Value'
    )
    fig = px.box(
        melted_data, 
        x='Metric', 
        y='Value', 
        color='Region',
        title="Distribution of Metrics by Region"
    )
    st.plotly_chart(fig, use_container_width=True)

elif viz_type == "Heatmap":
    # Create correlation heatmap
    corr = filtered_data[selected_metrics].corr()
    fig = px.imshow(
        corr, 
        text_auto=True, 
        aspect="auto",
        title="Correlation Between Metrics"
    )
    st.plotly_chart(fig, use_container_width=True)

elif viz_type == "Scatter Plot":
    if len(selected_metrics) >= 2:
        x_metric = st.selectbox("X-axis Metric", selected_metrics, index=0)
        y_metric = st.selectbox("Y-axis Metric", selected_metrics, index=min(1, len(selected_metrics)-1))
        
        fig = px.scatter(
            filtered_data, 
            x=x_metric, 
            y=y_metric, 
            color='Region',
            hover_data=['Image', 'Object'],
            title=f"{x_metric} vs {y_metric} by Region"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least two metrics for scatter plot")

# Statistical analysis
st.header("Statistical Analysis")
if st.checkbox("Show Statistical Summary"):
    for metric in selected_metrics:
        st.subheader(f"{metric} Statistics")
        stats_df = filtered_data.groupby('Region')[metric].describe()
        st.dataframe(stats_df)

# Download section
st.header("Download Data")
csv = filtered_data.to_csv(index=False).encode('utf-8')
st.download_button(
    "Download Filtered Data as CSV",
    csv,
    "filtered_metrics.csv",
    "text/csv",
    key='download-csv'
)
