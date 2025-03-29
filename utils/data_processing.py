import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

def preprocess_transcriptomics(data):
    """
    Preprocess transcriptomics data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw transcriptomics data with genes as rows and samples as columns.
        First column should contain gene identifiers.
    
    Returns:
    --------
    pandas.DataFrame
        Processed transcriptomics data.
    """
    # Make a copy to avoid modifying the original data
    processed_data = data.copy()
    
    # Extract gene IDs
    gene_ids = processed_data.iloc[:, 0]
    
    # Get expression data (exclude gene ID column)
    expression_data = processed_data.iloc[:, 1:].astype(float)
    
    # Check for missing values
    missing_values = expression_data.isnull().sum().sum()
    if missing_values > 0:
        # Impute missing values (per gene)
        imputer = SimpleImputer(strategy='mean', axis=1)
        expression_data = pd.DataFrame(
            imputer.fit_transform(expression_data),
            columns=expression_data.columns,
            index=expression_data.index
        )
    
    # Log transform if data is not already log-transformed
    # Check if data appears to be log-transformed already (approximate heuristic)
    if expression_data.mean().mean() > 50 or expression_data.max().max() > 1000:
        # Add small value to avoid log(0)
        min_nonzero = expression_data[expression_data > 0].min().min()
        expression_data = np.log2(expression_data + min_nonzero * 0.1)
    
    # Normalize data (per sample)
    # First transpose to have genes as columns for normalization across samples
    expression_data_t = expression_data.T
    scaler = StandardScaler()
    normalized_data_t = pd.DataFrame(
        scaler.fit_transform(expression_data_t),
        columns=expression_data_t.columns,
        index=expression_data_t.index
    )
    # Transpose back to original orientation
    normalized_data = normalized_data_t.T
    
    # Reconstruct the dataframe with gene IDs
    processed_data = pd.DataFrame(normalized_data)
    processed_data.insert(0, data.columns[0], gene_ids)
    processed_data.columns = data.columns
    
    return processed_data

def preprocess_tnseq(data):
    """
    Preprocess Tn-Seq data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw Tn-Seq data with genes as rows and conditions as columns.
        First column should contain gene identifiers.
    
    Returns:
    --------
    pandas.DataFrame
        Processed Tn-Seq data.
    """
    # Make a copy to avoid modifying the original data
    processed_data = data.copy()
    
    # Extract gene IDs
    gene_ids = processed_data.iloc[:, 0]
    
    # Get Tn-Seq data (exclude gene ID column)
    tnseq_data = processed_data.iloc[:, 1:].astype(float)
    
    # Check for missing values
    missing_values = tnseq_data.isnull().sum().sum()
    if missing_values > 0:
        # For Tn-Seq, missing values often indicate no insertions
        # Use a different imputation strategy for Tn-Seq data
        imputer = SimpleImputer(strategy='constant', fill_value=-10)  # Assuming negative values indicate essentiality
        tnseq_data = pd.DataFrame(
            imputer.fit_transform(tnseq_data),
            columns=tnseq_data.columns,
            index=tnseq_data.index
        )
    
    # Normalize data if needed
    # For Tn-Seq, often normalization is specific to the protocol used
    # Here we implement a basic normalization to make conditions comparable
    if tnseq_data.std().std() > 0:
        scaler = MinMaxScaler(feature_range=(-10, 2))  # Typical range for fitness scores
        tnseq_data = pd.DataFrame(
            scaler.fit_transform(tnseq_data),
            columns=tnseq_data.columns,
            index=tnseq_data.index
        )
    
    # Reconstruct the dataframe with gene IDs
    processed_data = pd.DataFrame(tnseq_data)
    processed_data.insert(0, data.columns[0], gene_ids)
    processed_data.columns = data.columns
    
    return processed_data

def preprocess_phenotype(data):
    """
    Preprocess phenotype data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw phenotype data with samples as rows and phenotypes as columns.
        First column should contain sample identifiers.
    
    Returns:
    --------
    pandas.DataFrame
        Processed phenotype data.
    """
    # Make a copy to avoid modifying the original data
    processed_data = data.copy()
    
    # Extract sample IDs
    sample_ids = processed_data.iloc[:, 0]
    
    # Get phenotype data (exclude sample ID column)
    phenotype_data = processed_data.iloc[:, 1:].astype(float)
    
    # Check for missing values
    missing_values = phenotype_data.isnull().sum().sum()
    if missing_values > 0:
        # Impute missing values per phenotype
        imputer = SimpleImputer(strategy='median')
        phenotype_data = pd.DataFrame(
            imputer.fit_transform(phenotype_data),
            columns=phenotype_data.columns,
            index=phenotype_data.index
        )
    
    # Normalize each phenotype to 0-1 range for easier comparison
    scaler = MinMaxScaler()
    phenotype_data = pd.DataFrame(
        scaler.fit_transform(phenotype_data),
        columns=phenotype_data.columns,
        index=phenotype_data.index
    )
    
    # Reconstruct the dataframe with sample IDs
    processed_data = pd.DataFrame(phenotype_data)
    processed_data.insert(0, data.columns[0], sample_ids)
    processed_data.columns = data.columns
    
    return processed_data

def integrate_omics_data(transcriptomics_data, phenotype_data, tnseq_data=None):
    """
    Integrate multiple types of omics data for downstream analysis.
    
    Parameters:
    -----------
    transcriptomics_data : pandas.DataFrame
        Processed transcriptomics data with genes as rows and samples as columns.
    phenotype_data : pandas.DataFrame
        Processed phenotype data with samples as rows and phenotypes as columns.
    tnseq_data : pandas.DataFrame, optional
        Processed Tn-Seq data with genes as rows and conditions as columns.
    
    Returns:
    --------
    dict
        Dictionary with integrated data matrices ready for analysis.
    """
    # Extract sample IDs from each dataset
    transcriptomics_samples = transcriptomics_data.columns[1:].tolist()
    phenotype_samples = phenotype_data.iloc[:, 0].tolist()
    
    # Find common samples
    common_samples = list(set(transcriptomics_samples) & set(phenotype_samples))
    
    if not common_samples:
        raise ValueError("No common samples found between transcriptomics and phenotype data.")
    
    # Create feature matrix X from transcriptomics data
    # First transpose to have samples as rows
    X_transcriptomics = transcriptomics_data.iloc[:, 1:].loc[:, common_samples].T
    
    # Match phenotype values with transcriptomics samples
    y = pd.DataFrame(index=common_samples)
    for phenotype in phenotype_data.columns[1:]:
        y[phenotype] = [phenotype_data.loc[phenotype_data.iloc[:, 0] == sample, phenotype].values[0] 
                        for sample in common_samples]
    
    # Add Tn-Seq data if available
    if tnseq_data is not None:
        tnseq_samples = tnseq_data.columns[1:].tolist()
        common_samples_tnseq = list(set(common_samples) & set(tnseq_samples))
        
        if common_samples_tnseq:
            X_tnseq = tnseq_data.iloc[:, 1:].loc[:, common_samples_tnseq].T
            
            # We need to align samples between transcriptomics and Tn-Seq
            # Only use samples present in both datasets
            X_transcriptomics = X_transcriptomics.loc[common_samples_tnseq]
            y = y.loc[common_samples_tnseq]
            
            # Combine features
            X = pd.concat([X_transcriptomics, X_tnseq], axis=1)
        else:
            X = X_transcriptomics
            print("Warning: No common samples between Tn-Seq and other datasets. Using only transcriptomics data.")
    else:
        X = X_transcriptomics
    
    return {
        'X': X,
        'y': y,
        'common_samples': common_samples,
        'gene_ids': transcriptomics_data.iloc[:, 0].tolist(),
        'feature_names': X.columns.tolist(),
        'phenotype_names': y.columns.tolist()
    }

def filter_features_by_variance(data, n_top=500):
    """
    Filter features by variance to select the most variable genes.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Processed omics data.
    n_top : int
        Number of top features to select.
    
    Returns:
    --------
    pandas.DataFrame
        Data with only the top most variable features.
    """
    # Calculate variance for each feature
    variances = data.var(axis=0)
    
    # Select top N features
    top_features = variances.nlargest(n_top).index
    
    # Return filtered data
    return data[top_features]

def filter_features_by_correlation(X, y, n_top=500):
    """
    Filter features by correlation with a target variable.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target variable.
    n_top : int
        Number of top features to select.
    
    Returns:
    --------
    pandas.DataFrame
        Data with only the top most correlated features.
    """
    correlations = []
    
    # Calculate correlation for each feature
    for col in X.columns:
        corr = abs(np.corrcoef(X[col], y)[0, 1])
        if not np.isnan(corr):
            correlations.append((col, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    # Get top feature names
    top_features = [feat for feat, corr in correlations[:n_top]]
    
    # Return filtered data
    return X[top_features]
