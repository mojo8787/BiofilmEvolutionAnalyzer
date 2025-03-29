import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from utils.data_processing import preprocess_transcriptomics, preprocess_tnseq, preprocess_phenotype

st.set_page_config(
    page_title="Data Import - Multi-Omics Platform",
    page_icon="🧬",
    layout="wide"
)

st.title("Data Import and Processing")

st.markdown("""
This module allows you to import various types of omics data for bacterial phenotype analysis:

1. **Transcriptomics Data** - Gene expression profiles (RNA-seq, microarray)
2. **Tn-Seq Data** - Transposon insertion sequencing for gene essentiality
3. **Phenotype Data** - Measured bacterial phenotypes (biofilm formation, motility, etc.)
""")

# Function to create a download link for a dataframe
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Create tabs for different data types
tab1, tab2, tab3, tab4 = st.tabs(["Transcriptomics", "Tn-Seq", "Phenotype", "Integrated Data"])

with tab1:
    st.header("Transcriptomics Data Import")
    
    st.markdown("""
    Import your transcriptomics data in CSV, TSV, or Excel format.
    - Rows should represent genes/features
    - Columns should represent samples/conditions
    - The first column should contain gene identifiers
    """)
    
    transcriptomics_file = st.file_uploader("Upload Transcriptomics Data", type=["csv", "tsv", "txt", "xlsx"])
    
    if transcriptomics_file is not None:
        try:
            # Determine the file type and read accordingly
            if transcriptomics_file.name.endswith(('.csv', '.txt')):
                df_transcriptomics = pd.read_csv(transcriptomics_file)
            elif transcriptomics_file.name.endswith('.tsv'):
                df_transcriptomics = pd.read_csv(transcriptomics_file, sep='\t')
            elif transcriptomics_file.name.endswith(('.xls', '.xlsx')):
                df_transcriptomics = pd.read_excel(transcriptomics_file)
            
            st.success(f"Transcriptomics data loaded successfully! Shape: {df_transcriptomics.shape}")
            
            # Display a preview of the data
            st.subheader("Data Preview")
            st.dataframe(df_transcriptomics.head())
            
            # Process the data
            if st.button("Process Transcriptomics Data"):
                processed_transcriptomics = preprocess_transcriptomics(df_transcriptomics)
                
                st.subheader("Processed Data Preview")
                st.dataframe(processed_transcriptomics.head())
                
                # Store in session state for later use
                st.session_state['transcriptomics_data'] = processed_transcriptomics
                
                # Provide a download link for processed data
                st.markdown(get_download_link(processed_transcriptomics, 
                                             "processed_transcriptomics.csv", 
                                             "Download Processed Transcriptomics Data"), 
                           unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error processing transcriptomics data: {e}")

with tab2:
    st.header("Tn-Seq Data Import")
    
    st.markdown("""
    Import your Tn-Seq data in CSV, TSV, or Excel format.
    - The data should include gene IDs and their corresponding fitness/essentiality scores
    - The file should include sufficient metadata about experimental conditions
    """)
    
    tnseq_file = st.file_uploader("Upload Tn-Seq Data", type=["csv", "tsv", "txt", "xlsx"])
    
    if tnseq_file is not None:
        try:
            # Determine the file type and read accordingly
            if tnseq_file.name.endswith(('.csv', '.txt')):
                df_tnseq = pd.read_csv(tnseq_file)
            elif tnseq_file.name.endswith('.tsv'):
                df_tnseq = pd.read_csv(tnseq_file, sep='\t')
            elif tnseq_file.name.endswith(('.xls', '.xlsx')):
                df_tnseq = pd.read_excel(tnseq_file)
            
            st.success(f"Tn-Seq data loaded successfully! Shape: {df_tnseq.shape}")
            
            # Display a preview of the data
            st.subheader("Data Preview")
            st.dataframe(df_tnseq.head())
            
            # Process the data
            if st.button("Process Tn-Seq Data"):
                processed_tnseq = preprocess_tnseq(df_tnseq)
                
                st.subheader("Processed Data Preview")
                st.dataframe(processed_tnseq.head())
                
                # Store in session state for later use
                st.session_state['tnseq_data'] = processed_tnseq
                
                # Provide a download link for processed data
                st.markdown(get_download_link(processed_tnseq, 
                                             "processed_tnseq.csv", 
                                             "Download Processed Tn-Seq Data"), 
                           unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error processing Tn-Seq data: {e}")

with tab3:
    st.header("Phenotype Data Import")
    
    st.markdown("""
    Import your phenotype data in CSV, TSV, or Excel format.
    - The data should include sample IDs and their corresponding phenotype measurements
    - Example phenotypes: biofilm formation, motility, antibiotic resistance, etc.
    """)
    
    phenotype_file = st.file_uploader("Upload Phenotype Data", type=["csv", "tsv", "txt", "xlsx"])
    
    if phenotype_file is not None:
        try:
            # Determine the file type and read accordingly
            if phenotype_file.name.endswith(('.csv', '.txt')):
                df_phenotype = pd.read_csv(phenotype_file)
            elif phenotype_file.name.endswith('.tsv'):
                df_phenotype = pd.read_csv(phenotype_file, sep='\t')
            elif phenotype_file.name.endswith(('.xls', '.xlsx')):
                df_phenotype = pd.read_excel(phenotype_file)
            
            st.success(f"Phenotype data loaded successfully! Shape: {df_phenotype.shape}")
            
            # Display a preview of the data
            st.subheader("Data Preview")
            st.dataframe(df_phenotype.head())
            
            # Process the data
            if st.button("Process Phenotype Data"):
                processed_phenotype = preprocess_phenotype(df_phenotype)
                
                st.subheader("Processed Data Preview")
                st.dataframe(processed_phenotype.head())
                
                # Store in session state for later use
                st.session_state['phenotype_data'] = processed_phenotype
                
                # Provide a download link for processed data
                st.markdown(get_download_link(processed_phenotype, 
                                             "processed_phenotype.csv", 
                                             "Download Processed Phenotype Data"), 
                           unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error processing phenotype data: {e}")

with tab4:
    st.header("Integrated Data View")
    
    if ('transcriptomics_data' in st.session_state or 
        'tnseq_data' in st.session_state or 
        'phenotype_data' in st.session_state):
        
        st.subheader("Imported Datasets Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'transcriptomics_data' in st.session_state:
                st.metric("Transcriptomics Data", 
                          f"{st.session_state['transcriptomics_data'].shape[0]} genes", 
                          f"{st.session_state['transcriptomics_data'].shape[1]} samples")
            else:
                st.warning("No transcriptomics data imported")
                
        with col2:
            if 'tnseq_data' in st.session_state:
                st.metric("Tn-Seq Data", 
                          f"{st.session_state['tnseq_data'].shape[0]} genes", 
                          f"{st.session_state['tnseq_data'].shape[1]} conditions")
            else:
                st.warning("No Tn-Seq data imported")
                
        with col3:
            if 'phenotype_data' in st.session_state:
                st.metric("Phenotype Data", 
                          f"{st.session_state['phenotype_data'].shape[0]} samples", 
                          f"{st.session_state['phenotype_data'].shape[1]} phenotypes")
            else:
                st.warning("No phenotype data imported")
        
        st.markdown("---")
        
        # Provide guidance on proceeding with the workflow
        st.subheader("Next Steps")
        st.markdown("""
        Now that you have imported your data, you can proceed to:
        
        1. **Data Exploration** - Explore your data with visualization tools
        2. **Machine Learning** - Build models to predict phenotypes from omics data
        3. **Metabolic Modeling** - Integrate omics data with genome-scale metabolic models
        
        Navigate to these sections using the sidebar.
        """)
        
    else:
        st.info("Import at least one dataset to see the integrated view.")
