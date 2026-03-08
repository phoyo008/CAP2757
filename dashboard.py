"""
Step 6: Streamlit Interactive App - Lab 1 Data Exploration
Simple dashboard following the rubric requirements
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Biscayne Bay Data Explorer",
    page_icon="🌊",
    layout="wide"
)

# Title
st.title("Biscayne Bay Water Quality Explorer")
st.markdown("Lab 1: Data Exploration and Visualization")
st.markdown("---")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('oct25-2024.csv')
    return df

df = load_data()

# Sidebar controls
st.sidebar.header("Dashboard Controls")

# Checkbox to show/hide raw data
show_raw_data = st.sidebar.checkbox("Show Raw Data Preview", value=False)

# Checkbox to show statistics
show_stats = st.sidebar.checkbox("Show Summary Statistics", value=True)

# Checkbox to show correlation matrix
show_corr = st.sidebar.checkbox("Show Correlation Matrix", value=True)

# Checkbox to show visualizations
show_viz = st.sidebar.checkbox("Show Visualizations", value=True)

st.sidebar.markdown("---")
st.sidebar.info("""
**Dataset Info:**
- Samples: 678
- Collection Date: Oct 25, 2024
- Location: Biscayne Bay, Miami
- Variables: 24 columns
- Missing Values: 0
""")

# Display dataset info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Samples", len(df))
with col2:
    st.metric("Total Columns", len(df.columns))
with col3:
    st.metric("Missing Values", df.isnull().sum().sum())

st.markdown("---")


# SECTION 1: RAW DATA PREVIEW

if show_raw_data:
    st.subheader("Raw Dataset Preview")
    st.write(f"Showing first 10 rows of {len(df)} total samples:")
    st.dataframe(df.head(10), use_container_width=True)

    st.write("\n**Dataset Info:**")
    st.text(str(df.info()))

# SECTION 2: SUMMARY STATISTICS

if show_stats:
    st.subheader("Summary Statistics")
    st.write("Statistical summary of all numeric columns:")

    # Show summary statistics
    summary = df.describe()
    st.dataframe(summary.round(4), use_container_width=True)

    # Key observations
    with st.expander("Key Observations", expanded=True):
        st.write("""
        **Temperature (Temp °C):**
        - Mean: ~27.6°C, Range: 26.8 - 28.9°C
        - Very stable tropical conditions
        
        **Salinity (Sal psu):**
        - Mean: ~30.9 psu, Range: 0.21 - 34.89 psu
        - High variability indicates freshwater input mixing
        
        **pH:**
        - Mean: ~8.38, Range: 7.8 - 8.54
        - Typical alkaline seawater chemistry
        
        **Dissolved Oxygen (ODO mg/L):**
        - Mean: ~7.36 mg/L, Range: 2.73 - 13.09 mg/L
        - Generally adequate, some hypoxic zones
        
        **Turbidity (Turbidity FNU):**
        - Mean: ~548.5 FNU, Range: 0 - 1661.75 FNU
        - High variation indicates different water conditions
        
        **Altitude (Altitude m):**
        - Extreme positive outlier at 24.1 m suggests sensor error
        - Should be removed for clean analysis
        """)


# SECTION 3: CORRELATION ANALYSIS

if show_corr:
    st.subheader("🔗 Correlation Analysis")

    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()

    # Display correlation matrix
    st.write("**Correlation Matrix:**")
    st.dataframe(corr_matrix.round(3), use_container_width=True)

    # Find and display strongest correlations
    st.write("\n**Strongest Correlations:**")

    # Create list of correlations (excluding diagonal)
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            var1 = corr_matrix.columns[i]
            var2 = corr_matrix.columns[j]
            r = corr_matrix.iloc[i, j]
            corr_pairs.append((var1, var2, r))

    # Sort by absolute value
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # Display top 5
    for i, (var1, var2, r) in enumerate(corr_pairs[:5], 1):
        st.write(f"{i}. **{var1}** ↔ **{var2}**: r = {r:.4f}")


# SECTION 4: VISUALIZATIONS

if show_viz:
    st.subheader("Data Visualizations")

    # Create two columns for side-by-side plots
    col1, col2 = st.columns(2)

    # Scatter plot: Salinity vs. Temperature
    with col1:
        st.write("**Scatter Plot: Salinity vs. Temperature**")
        fig_scatter = px.scatter(
            df,
            x='Sal psu',
            y='Temp °C',
            title='Salinity vs. Temperature',
            labels={'Sal psu': 'Salinity (psu)', 'Temp °C': 'Temperature (°C)'},
            opacity=0.6
        )
        fig_scatter.update_layout(height=400, width=500)
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.write("""
        **Observation:** The scatter plot shows clustering around 30-35 psu salinity 
        and 27-28°C temperature, indicating stable marine conditions. 
        Lower salinity points suggest freshwater input zones.
        """)

    # Histogram: pH levels
    with col2:
        st.write("**Histogram: pH Distribution**")
        fig_hist_ph = px.histogram(
            df,
            x='pH',
            nbins=25,
            title='pH Distribution',
            labels={'pH': 'pH Level'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_hist_ph.update_layout(height=400, width=500, showlegend=False)
        st.plotly_chart(fig_hist_ph, use_container_width=True)

        st.write("""
        **Observation:** pH values cluster around 8.4, which is typical for seawater. 
        The distribution is relatively normal and concentrated, indicating stable 
        alkaline marine chemistry throughout the survey.
        """)

    # Temperature histogram
    st.write("**Histogram: Temperature Distribution**")
    fig_hist_temp = px.histogram(
        df,
        x='Temp °C',
        nbins=20,
        title='Temperature Distribution',
        labels={'Temp °C': 'Temperature (°C)'},
        color_discrete_sequence=['#ff7f0e']
    )
    fig_hist_temp.update_layout(height=400)
    st.plotly_chart(fig_hist_temp, use_container_width=True)


# SECTION 5: OUTLIER INFORMATION

st.markdown("---")
st.subheader("🧹 Outlier Detection Notes")

with st.expander("View Outlier Analysis", expanded=False):
    st.write("""
    **IQR Method Results for Key Variables:**
    
    **Temperature (Temp °C):**
    - Q1: 27.3°C, Q3: 27.8°C, IQR: 0.52°C
    - Bounds: 26.5°C to 28.6°C
    - 7 outliers detected (within expected range)
    
    **Salinity (Sal psu):**
    - Q1: 28.4 psu, Q3: 34.3 psu, IQR: 5.9 psu
    - Bounds: 19.5 to 43.1 psu
    - 10 outliers detected (freshwater input zones)
    
    **Altitude (Altitude m):**
    - Q1: -1.22 m, Q3: 1.01 m, IQR: 2.23 m
    - Bounds: -4.56 to 4.35 m
    - **Extreme outliers detected** (max 24.1 m)
    - **Decision: Remove these as sensor errors** (equipment likely lifted from water)
    
    **Recommendation:**
    Remove altitude outliers for clean analysis, but retain other outliers as they 
    represent real environmental variations (freshwater zones, different water masses).
    """)

# FOOTER

st.markdown("---")
st.markdown("""
**Lab 1: Data Exploration and Visualization**
- Course: CAP 2757 - Introduction to Data Science
- Institution: Florida International University
- Dataset: Biscayne Bay Water Quality (October 25, 2024)
- By: Pablo Hoyos 
- Panther ID: 6599555
""")
