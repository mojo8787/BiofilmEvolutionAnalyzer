import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.visualization import create_heatmap, generate_pca_plot, create_correlation_network

st.set_page_config(
    page_title="Data Exploration - Multi-Omics Platform",
    page_icon="🧬",
    layout="wide"
)

st.title("Multi-Omics Data Exploration")

st.markdown("""
This module helps you explore and visualize your imported multi-omics datasets:

1. **Statistical Summaries** - Basic statistics of your data
2. **Dimensionality Reduction** - PCA, t-SNE for pattern identification
3. **Correlation Analysis** - Find relationships between features
4. **Data Distribution** - Understand the distribution of your measurements
""")

# Check if data is available in session state
if not any(k in st.session_state for k in ['transcriptomics_data', 'tnseq_data', 'phenotype_data']):
    st.warning("No data found. Please import your data first in the 'Data Import' section.")
    st.stop()

# Sidebar for data selection
st.sidebar.header("Data Selection")
data_type = st.sidebar.selectbox(
    "Select Data Type",
    options=["Transcriptomics", "Tn-Seq", "Phenotype"],
    index=0
)

# Map selection to session state key
data_key_map = {
    "Transcriptomics": "transcriptomics_data",
    "Tn-Seq": "tnseq_data",
    "Phenotype": "phenotype_data"
}

data_key = data_key_map[data_type]

# Check if the selected data is available
if data_key not in st.session_state:
    st.warning(f"No {data_type} data found. Please import it first.")
    st.stop()

data = st.session_state[data_key]

# Create tabs for different exploration methods
tab1, tab2, tab3, tab4 = st.tabs(["Summary Statistics", "Dimensionality Reduction", "Correlation Analysis", "Data Distribution"])

with tab1:
    st.header("Summary Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Overview")
        st.write(f"Number of rows: {data.shape[0]}")
        st.write(f"Number of columns: {data.shape[1]}")
        st.write(f"Missing values: {data.isna().sum().sum()}")
        
        if data.shape[1] > 1:
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                st.subheader("Descriptive Statistics")
                st.dataframe(data[numeric_cols].describe())
    
    with col2:
        st.subheader("Data Preview")
        st.dataframe(data.head())
        
        # Display column types
        st.subheader("Column Data Types")
        col_types = pd.DataFrame({
            'Column': data.columns,
            'Type': data.dtypes.values
        })
        st.dataframe(col_types)

with tab2:
    st.header("Dimensionality Reduction")
    
    # Filter only numeric columns for dimensionality reduction
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    
    if numeric_data.shape[1] < 2:
        st.warning("Not enough numeric columns for dimensionality reduction.")
    else:
        # Sidebar options for PCA
        st.sidebar.subheader("PCA Options")
        n_components = st.sidebar.slider("Number of components", 2, min(10, numeric_data.shape[1]), 2)
        scale_data = st.sidebar.checkbox("Scale data before PCA", True)
        
        # Try to find a potential group column (non-numeric)
        categorical_cols = data.select_dtypes(exclude=['float64', 'int64']).columns.tolist()
        group_col = None
        
        if categorical_cols:
            group_col = st.sidebar.selectbox(
                "Color by category (optional)",
                options=[None] + categorical_cols,
                index=0
            )
        
        # Generate PCA plot
        fig = generate_pca_plot(numeric_data, n_components, scale_data, group_col, data if group_col else None)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display explained variance
        pca = PCA(n_components=n_components)
        if scale_data:
            scaled_data = StandardScaler().fit_transform(numeric_data)
            pca.fit(scaled_data)
        else:
            pca.fit(numeric_data)
            
        explained_variance = pca.explained_variance_ratio_
        
        # Create explained variance plot
        fig = px.bar(
            x=[f'PC{i+1}' for i in range(n_components)],
            y=explained_variance,
            labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'},
            title='Explained Variance by Principal Component'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance in PCA
        if st.checkbox("Show feature contributions to principal components"):
            loadings = pca.components_
            loading_df = pd.DataFrame(
                loadings.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=numeric_data.columns
            )
            
            st.dataframe(loading_df)
            
            # Visualize feature contributions for PC1 and PC2
            fig = px.scatter(
                x=loading_df['PC1'],
                y=loading_df['PC2'],
                text=loading_df.index,
                labels={'x': 'PC1', 'y': 'PC2'},
                title='Feature Contributions to PC1 and PC2'
            )
            fig.update_traces(textposition='top center')
            fig.update_layout(height=600, width=800)
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Correlation Analysis")
    
    # Filter only numeric columns for correlation
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    
    if numeric_data.shape[1] < 2:
        st.warning("Not enough numeric columns for correlation analysis.")
    else:
        correlation_method = st.selectbox(
            "Correlation Method",
            options=["Pearson", "Spearman", "Kendall"],
            index=0
        )
        
        # Calculate correlation matrix
        corr_method_map = {
            "Pearson": "pearson",
            "Spearman": "spearman",
            "Kendall": "kendall"
        }
        
        corr_matrix = numeric_data.corr(method=corr_method_map[correlation_method])
        
        # Create heatmap
        heatmap_fig = create_heatmap(corr_matrix, f"{correlation_method} Correlation Matrix")
        st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Option to show network visualization
        if st.checkbox("Show correlation network"):
            threshold = st.slider(
                "Correlation threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05
            )
            
            network_fig = create_correlation_network(corr_matrix, threshold)
            st.plotly_chart(network_fig, use_container_width=True)
        
        # Allow downloading the correlation matrix
        csv = corr_matrix.to_csv()
        st.download_button(
            label="Download Correlation Matrix",
            data=csv,
            file_name=f"{data_type.lower()}_correlation_{correlation_method.lower()}.csv",
            mime="text/csv",
        )

with tab4:
    st.header("Data Distribution")
    
    # Filter only numeric columns for distribution analysis
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    
    if numeric_data.shape[1] < 1:
        st.warning("No numeric columns for distribution analysis.")
    else:
        # Select columns for visualization
        selected_columns = st.multiselect(
            "Select columns to visualize",
            options=numeric_data.columns.tolist(),
            default=numeric_data.columns.tolist()[:min(3, numeric_data.shape[1])]
        )
        
        if not selected_columns:
            st.warning("Please select at least one column.")
        else:
            # Distribution visualization type
            viz_type = st.radio(
                "Visualization type",
                options=["Histogram", "Box Plot", "Violin Plot", "Density Plot"],
                index=0
            )
            
            if viz_type == "Histogram":
                # Create histograms
                if len(selected_columns) <= 2:
                    # Single row of histograms if 1-2 columns
                    fig = make_subplots(rows=1, cols=len(selected_columns))
                    for i, col in enumerate(selected_columns):
                        fig.add_trace(
                            go.Histogram(x=numeric_data[col], name=col),
                            row=1, col=i+1
                        )
                    fig.update_layout(height=400, title_text="Histograms of Selected Features")
                else:
                    # Grid of histograms if more than 2 columns
                    n_cols = min(3, len(selected_columns))
                    n_rows = (len(selected_columns) + n_cols - 1) // n_cols
                    fig = make_subplots(rows=n_rows, cols=n_cols)
                    
                    for i, col in enumerate(selected_columns):
                        row = i // n_cols + 1
                        col_idx = i % n_cols + 1
                        fig.add_trace(
                            go.Histogram(x=numeric_data[col], name=col),
                            row=row, col=col_idx
                        )
                    fig.update_layout(height=300 * n_rows, title_text="Histograms of Selected Features")
                
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Box Plot":
                fig = go.Figure()
                for col in selected_columns:
                    fig.add_trace(go.Box(y=numeric_data[col], name=col))
                fig.update_layout(title_text="Box Plots of Selected Features")
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Violin Plot":
                fig = go.Figure()
                for col in selected_columns:
                    fig.add_trace(go.Violin(y=numeric_data[col], name=col, box_visible=True, meanline_visible=True))
                fig.update_layout(title_text="Violin Plots of Selected Features")
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Density Plot":
                # Using matplotlib for density plots through seaborn
                fig, ax = plt.subplots(figsize=(10, 6))
                for col in selected_columns:
                    sns.kdeplot(numeric_data[col], label=col, ax=ax)
                plt.title("Density Plots of Selected Features")
                plt.legend()
                st.pyplot(fig)
                
            # Add descriptive statistics for the selected columns
            st.subheader("Descriptive Statistics for Selected Features")
            st.dataframe(numeric_data[selected_columns].describe())
