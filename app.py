import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Multi-Omics Integration Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page content
st.title("Multi-Omics Integration Platform for Bacterial Phenotype Prediction")
st.caption("Developed by Almotasem Bellah Younis (Motasem)")

st.markdown("""
## Welcome to the Bacterial Multi-Omics Analysis Platform

This computational platform integrates multi-omics data to predict bacterial phenotypes and 
elucidate regulatory mechanisms, with a focus on biofilm versus motile phenotypes.

### Key Features:

1. **Data Integration** - Import and process transcriptomics, Tn-Seq, and phenotype data
2. **Machine Learning Models** - Predict bacterial phenotypes from multi-omics data
3. **Regulatory Mechanism Discovery** - Identify key regulators using feature importance
4. **Metabolic Modeling** - Explore resource allocation trade-offs using genome-scale models
5. **In Silico Evolution** - Simulate bacterial adaptation under different conditions
6. **Experimental Validation** - Compare computational predictions with experimental results

### Getting Started:
- Use the sidebar to navigate to different modules
- Start by importing your data in the Data Import section
- Follow the workflow from data exploration to machine learning and simulations
""")

# Display a sample visualization in the main page
st.subheader("Example: Bacterial Phenotype Space")

# Sample data for visualization
np.random.seed(42)
num_samples = 100
sample_data = pd.DataFrame({
    'Biofilm Formation': np.random.normal(0.5, 0.15, num_samples),
    'Motility': np.random.normal(0.5, 0.15, num_samples),
    'Antibiotic Resistance': np.random.normal(0.5, 0.15, num_samples),
    'Group': np.random.choice(['Group A', 'Group B', 'Group C'], num_samples)
})

# Constrain values to be between 0 and 1
for col in ['Biofilm Formation', 'Motility', 'Antibiotic Resistance']:
    sample_data[col] = sample_data[col].clip(0, 1)

# Create an inverse relationship between biofilm and motility
sample_data['Motility'] = 1 - sample_data['Biofilm Formation'] + np.random.normal(0, 0.1, num_samples)
sample_data['Motility'] = sample_data['Motility'].clip(0, 1)

# Create a 3D scatter plot
fig = px.scatter_3d(
    sample_data, 
    x='Biofilm Formation', 
    y='Motility', 
    z='Antibiotic Resistance',
    color='Group',
    opacity=0.7,
    title="Example: Bacterial Phenotype Space"
)

fig.update_layout(
    scene=dict(
        xaxis_title='Biofilm Formation',
        yaxis_title='Motility',
        zaxis_title='Antibiotic Resistance'
    ),
    width=800,
    height=600
)

st.plotly_chart(fig)

st.info("""
**Note:** The visualization above is a conceptual representation of the phenotype space. 
Your actual data will replace this example once imported and processed.
""")

st.markdown("""
### Next Steps

Navigate to the **Data Import** page from the sidebar to get started with your analysis.

For more information about the methods and workflows, visit the **Documentation** section.
""")
