import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Set the page configuration    
st.set_page_config(
    page_title="Data Visualization Dashboard",
    page_icon="📊",                                                 
    layout="wide"
)

# Load the dataset
df = pd.read_csv('StressLevelDataset.csv')

# Show description of the dataset
st.title("📊 Data Visualization Dashboard")
st.write("This dashboard provides visual insights into the Stress Level Dataset.")

# Display the first few rows of the dataset
st.subheader("Dataset Overview")
st.dataframe(df.head())

# --- Sidebar for Navigation ---
st.sidebar.title("🔍 Navigation")
options = st.sidebar.radio(
    "Go to:", 
    ["Dataset Info", "Statistics", "Visualizations", "Download"]
)

# =========================
# Dataset Info
# =========================
if options == "Dataset Info":
    st.header("📊 Dataset Information")
    st.write(f"**Number of Rows:** {df.shape[0]}")
    st.write(f"**Number of Columns:** {df.shape[1]}")
    st.write("**Column Names:**", list(df.columns))

    # Missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Data types
    st.subheader("Data Types")
    st.write(df.dtypes)

# =========================
# Statistics
# =========================
elif options == "Statistics":
    st.header("📈 Dataset Statistics")
    st.write(df.describe(include='all'))

    # Correlation matrix
    st.subheader("Correlation Matrix")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# =========================
# Visualizations
# =========================
elif options == "Visualizations":
    st.header("📊 Data Visualizations")

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # --- Histogram ---
    st.subheader("Histogram")
    selected_hist_col = st.selectbox("Select a column for histogram:", numeric_columns)
    fig = px.histogram(df, x=selected_hist_col, nbins=30, title=f"Histogram of {selected_hist_col}")
    st.plotly_chart(fig)

    # --- Box Plot ---
    st.subheader("Box Plot")
    selected_box_col = st.selectbox("Select a column for box plot:", numeric_columns)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(y=df[selected_box_col], ax=ax)
    ax.set_title(f"Boxplot of {selected_box_col}")
    st.pyplot(fig)

    # --- Scatter Plot ---
    st.subheader("Scatter Plot")
    col1 = st.selectbox("Select X-axis:", numeric_columns, key="scatter_x")
    col2 = st.selectbox("Select Y-axis:", numeric_columns, key="scatter_y")
    color_col = st.selectbox("Select a categorical column for color (optional):", ["None"] + categorical_columns)
    if color_col != "None":
        fig = px.scatter(df, x=col1, y=col2, color=df[color_col], title=f"Scatter Plot: {col1} vs {col2}")
    else:
        fig = px.scatter(df, x=col1, y=col2, title=f"Scatter Plot: {col1} vs {col2}")
    st.plotly_chart(fig)

    # --- Categorical Column Count ---
    if categorical_columns:
        st.subheader("Categorical Column Distribution")
        selected_cat_col = st.selectbox("Select a categorical column:", categorical_columns, key="cat_col")
        fig = px.bar(df[selected_cat_col].value_counts().reset_index(), 
                     x='index', y=selected_cat_col, 
                     title=f'Distribution of {selected_cat_col}')
        st.plotly_chart(fig)

# =========================
# Download
# =========================
elif options == "Download":
    st.header("📥 Download Dataset")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name='StressLevelDataset.csv',
        mime='text/csv'
    )

# --- Footer ---
st.markdown("---")
st.markdown("Created with ❤️ by [Susan Kihara]")
