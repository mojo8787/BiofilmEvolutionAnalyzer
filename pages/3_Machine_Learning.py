import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
# import shap - temporarily disabled
import io
from contextlib import redirect_stdout
from utils.ml_models import (
    build_classifier, build_regressor, feature_importance_plot, 
    plot_confusion_matrix, plot_shap_summary, get_neural_network_feature_importance
)

st.set_page_config(
    page_title="Machine Learning - Multi-Omics Platform",
    page_icon="🧬",
    layout="wide"
)

st.title("Machine Learning for Bacterial Phenotype Prediction")

st.markdown("""
This module allows you to build machine learning models to predict bacterial phenotypes from omics data:

1. **Data Integration** - Combine transcriptomics, Tn-Seq, and other data
2. **Feature Selection** - Identify the most relevant features for prediction
3. **Model Training** - Train and validate machine learning models
4. **Model Interpretation** - Use SHAP values to understand feature importance
""")

# Check if data is available in session state
required_data = ['transcriptomics_data', 'phenotype_data']
missing_data = [data for data in required_data if data not in st.session_state]

if missing_data:
    st.warning(f"Missing required data: {', '.join(missing_data)}. Please import your data first in the 'Data Import' section.")
    st.stop()

# Data Integration Section
st.header("1. Data Integration")

st.markdown("""
To build an effective ML model, we need to integrate the omics data with phenotypes.
Select the features and target variables to use in your model.
""")

# Get available data
transcriptomics_data = st.session_state['transcriptomics_data']
phenotype_data = st.session_state['phenotype_data']

# Show info about available datasets
col1, col2 = st.columns(2)
with col1:
    st.subheader("Transcriptomics Data")
    st.write(f"Shape: {transcriptomics_data.shape}")
    st.write(f"Features: {transcriptomics_data.shape[1]} genes")
    
with col2:
    st.subheader("Phenotype Data")
    st.write(f"Shape: {phenotype_data.shape}")
    st.write(f"Available phenotypes: {', '.join(phenotype_data.columns[1:])}")

# Add Tn-Seq data if available
if 'tnseq_data' in st.session_state:
    tnseq_data = st.session_state['tnseq_data']
    st.subheader("Tn-Seq Data")
    st.write(f"Shape: {tnseq_data.shape}")
    st.write(f"Features: {tnseq_data.shape[1]} measurements")
    include_tnseq = st.checkbox("Include Tn-Seq data in model", value=True)
else:
    include_tnseq = False

# Select target phenotype
target_phenotype = st.selectbox(
    "Select target phenotype to predict",
    options=phenotype_data.columns[1:].tolist(),
    index=0
)

# Determine problem type (classification or regression)
problem_type = st.radio(
    "Problem type",
    options=["Classification", "Regression"],
    index=0,
    help="Classification for categorical outcomes (e.g., biofilm vs. motile), regression for continuous measurements"
)

if problem_type == "Classification":
    # For classification, we need to check if the target is categorical
    unique_values = phenotype_data[target_phenotype].nunique()
    if unique_values > 10:  # Assuming more than 10 unique values indicates a continuous variable
        st.warning(f"The selected phenotype has {unique_values} unique values. Consider using regression instead.")
    
    # Let user set threshold for binary classification if needed
    if 2 < unique_values <= 10:
        st.info(f"The selected phenotype has {unique_values} unique values. You can proceed with multi-class classification or convert to binary.")
        binary_classification = st.checkbox("Convert to binary classification")
        
        if binary_classification:
            threshold_method = st.radio(
                "Threshold method",
                options=["Value", "Median", "Mean"],
                index=1
            )
            
            if threshold_method == "Value":
                threshold_value = st.number_input(
                    "Threshold value",
                    value=float(phenotype_data[target_phenotype].median())
                )
            elif threshold_method == "Median":
                threshold_value = float(phenotype_data[target_phenotype].median())
                st.write(f"Median value: {threshold_value}")
            else:  # Mean
                threshold_value = float(phenotype_data[target_phenotype].mean())
                st.write(f"Mean value: {threshold_value}")
            
            # Preview the binary classification
            binary_target = (phenotype_data[target_phenotype] > threshold_value).astype(int)
            st.write(f"Class distribution: {binary_target.value_counts().to_dict()}")
        else:
            # Multi-class classification
            st.write(f"Class distribution: {phenotype_data[target_phenotype].value_counts().to_dict()}")

# Feature selection
st.header("2. Feature Selection")

# Get sample IDs which are present in both datasets
transcriptomics_samples = transcriptomics_data.columns[1:].tolist()
phenotype_samples = phenotype_data.iloc[:, 0].tolist()
common_samples = list(set(transcriptomics_samples) & set(phenotype_samples))

if not common_samples:
    st.error("No common samples found between transcriptomics and phenotype data. Check your sample IDs.")
    st.stop()

st.write(f"Found {len(common_samples)} common samples between datasets.")

# Feature selection method
feature_selection_method = st.selectbox(
    "Feature selection method",
    options=["All features", "Top genes by variance", "Top genes by correlation with phenotype"],
    index=1
)

if feature_selection_method == "Top genes by variance":
    n_features = st.slider("Number of top genes by variance", 10, 500, 100, 10)
    
    # Get top genes by variance
    gene_variances = transcriptomics_data.iloc[:, 1:].var(axis=1)
    top_genes_idx = gene_variances.sort_values(ascending=False).head(n_features).index
    selected_genes = transcriptomics_data.loc[top_genes_idx]
    
    st.success(f"Selected {len(selected_genes)} genes based on variance.")
    
elif feature_selection_method == "Top genes by correlation with phenotype":
    n_features = st.slider("Number of top genes by correlation", 10, 500, 100, 10)
    
    # Calculate correlation between each gene and the phenotype
    correlations = []
    for idx, row in transcriptomics_data.iterrows():
        gene_id = row.iloc[0]
        gene_values = row.iloc[1:].values
        phenotype_values = []
        
        # Match phenotype values with gene samples
        for sample in transcriptomics_data.columns[1:]:
            if sample in phenotype_samples:
                phenotype_idx = phenotype_samples.index(sample)
                phenotype_values.append(phenotype_data.loc[phenotype_idx, target_phenotype])
        
        # Calculate correlation if we have enough matching samples
        if len(phenotype_values) > 5:
            gene_values_matched = [gene_values[i] for i, sample in enumerate(transcriptomics_data.columns[1:]) if sample in phenotype_samples]
            corr = np.corrcoef(gene_values_matched, phenotype_values)[0, 1]
            correlations.append((gene_id, abs(corr)))
    
    # Sort by absolute correlation and select top genes
    correlations.sort(key=lambda x: x[1], reverse=True)
    top_genes = [gene for gene, corr in correlations[:n_features]]
    
    # Filter transcriptomics data to include only top genes
    selected_genes = transcriptomics_data[transcriptomics_data.iloc[:, 0].isin(top_genes)]
    
    st.success(f"Selected {len(selected_genes)} genes based on correlation with {target_phenotype}.")
    
else:  # All features
    selected_genes = transcriptomics_data
    st.write(f"Using all {len(selected_genes)} genes as features.")

# Model Training Section
st.header("3. Model Training")

# Prepare features and target
X = selected_genes.iloc[:, 1:].loc[:, common_samples].T  # Transpose so samples are rows
y = pd.Series(dtype='float64')

# Match phenotype values with transcriptomics samples
for sample in common_samples:
    sample_idx = phenotype_samples.index(sample)
    y[sample] = phenotype_data.loc[sample_idx, target_phenotype]

# For classification, convert to binary if requested
if problem_type == "Classification" and 'binary_classification' in locals() and binary_classification:
    y = (y > threshold_value).astype(int)

# Add Tn-Seq data if available and selected
if include_tnseq and 'tnseq_data' in st.session_state:
    tnseq_data = st.session_state['tnseq_data']
    tnseq_genes = tnseq_data.iloc[:, 0].tolist()
    tnseq_features = tnseq_data.iloc[:, 1:].loc[:, tnseq_data.columns[1:].isin(common_samples)].T
    
    # Combine with transcriptomics features
    X = pd.concat([X, tnseq_features], axis=1)
    
    st.write(f"Added {tnseq_features.shape[1]} Tn-Seq features to the model.")

# Display the final feature matrix shape
st.write(f"Final feature matrix shape: {X.shape}")
st.write(f"Target vector shape: {y.shape}")

# Split data into train and test sets
test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
random_state = st.number_input("Random seed", 0, 1000, 42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

st.write(f"Training set: {X_train.shape[0]} samples")
st.write(f"Testing set: {X_test.shape[0]} samples")

# Select ML algorithm
if problem_type == "Classification":
    algorithm = st.selectbox(
        "Select classification algorithm",
        options=["Random Forest", "Gradient Boosting", "Support Vector Machine", "Logistic Regression", "Neural Network"],
        index=0
    )
else:  # Regression
    algorithm = st.selectbox(
        "Select regression algorithm",
        options=["Random Forest", "Gradient Boosting", "Support Vector Machine", "Linear Regression", "Neural Network"],
        index=0
    )

# Neural Network specific options
nn_params = {}
if algorithm == "Neural Network":
    st.subheader("Neural Network Configuration")
    nn_params['input_dim'] = X.shape[1]  # Number of features
    
    if problem_type == "Classification":
        # For classification, determine if binary or multi-class
        unique_classes = np.unique(y)
        if len(unique_classes) == 2:
            nn_params['binary'] = True
            st.info("Binary classification detected. Using a binary classification neural network.")
        else:
            nn_params['binary'] = False
            nn_params['num_classes'] = len(unique_classes)
            st.info(f"Multi-class classification detected with {len(unique_classes)} classes.")
    
    # Additional neural network hyperparameters could be added here
    st.info("Neural network will use early stopping to prevent overfitting.")

# Hyperparameter tuning option
tune_hyperparams = st.checkbox("Tune hyperparameters (may take time)", value=False)

# Model training button
if st.button("Train Model"):
    with st.spinner("Training model..."):
        # Create pipeline with preprocessing and model
        if problem_type == "Classification":
            if algorithm == "Neural Network":
                model_pipeline = build_classifier(algorithm, tune_hyperparams, **nn_params)
            else:
                model_pipeline = build_classifier(algorithm, tune_hyperparams)
        else:  # Regression
            if algorithm == "Neural Network":
                model_pipeline = build_regressor(algorithm, tune_hyperparams, **nn_params)
            else:
                model_pipeline = build_regressor(algorithm, tune_hyperparams)
        
        # Fit the model
        model_pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model_pipeline.predict(X_test)
        
        # Store model in session state
        st.session_state['trained_model'] = model_pipeline
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = y_pred
        st.session_state['feature_names'] = X.columns.tolist()
        st.session_state['problem_type'] = problem_type
        st.session_state['algorithm'] = algorithm
        
        st.success("Model training complete!")

# Model Evaluation Section
if 'trained_model' in st.session_state:
    st.header("4. Model Evaluation")
    
    model = st.session_state['trained_model']
    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']
    y_train = st.session_state['y_train']
    y_test = st.session_state['y_test']
    y_pred = st.session_state['y_pred']
    problem_type = st.session_state['problem_type']
    
    # Display evaluation metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        
        if problem_type == "Classification":
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Display metrics
            st.metric("Accuracy", f"{accuracy:.4f}")
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            # Confusion matrix
            cm_fig = plot_confusion_matrix(y_test, y_pred)
            st.pyplot(cm_fig)
            
        else:  # Regression
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Display metrics
            st.metric("Mean Squared Error", f"{mse:.4f}")
            st.metric("R² Score", f"{r2:.4f}")
            
            # Actual vs Predicted plot
            fig = px.scatter(
                x=y_test, y=y_pred,
                labels={'x': 'Actual', 'y': 'Predicted'},
                title='Actual vs Predicted Values'
            )
            
            # Add diagonal line for perfect predictions
            fig.add_trace(
                go.Scatter(
                    x=[min(y_test), max(y_test)],
                    y=[min(y_test), max(y_test)],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Perfect Prediction'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Interpretation")
        
        # Feature importance
        feature_names = st.session_state['feature_names']
        algorithm = st.session_state['algorithm']
        
        if algorithm == "Neural Network":
            st.info("""
            Neural networks don't provide direct feature importance metrics like tree-based models.
            For neural networks, we compute feature importance using permutation importance,
            which measures how model performance decreases when a feature is randomly shuffled.
            """)
            
            # For neural networks, we could calculate permutation importance, but it can be time-consuming
            # So we'll provide a button to compute it on demand
            if st.button("Calculate Neural Network Feature Importance (may take time)"):
                with st.spinner("Computing permutation importance for neural network..."):
                    try:
                        # Get the trained model component
                        if hasattr(model, 'named_steps'):
                            model_name = list(model.named_steps.keys())[-1]
                            model_component = model.named_steps[model_name]
                            
                            # Get feature importance for neural network
                            X_test_np = X_test.values
                            y_test_np = y_test.values
                            
                            # Use our custom function for neural network feature importance
                            if problem_type == "Classification":
                                importances = get_neural_network_feature_importance(model, X_test_np, y_test_np, feature_names)
                                st.session_state['nn_importances'] = importances
                                
                                # Create a DataFrame for visualization
                                imp_df = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Importance': importances
                                }).sort_values('Importance', ascending=False).head(20)
                                
                                # Plot the importance
                                fig, ax = plt.subplots(figsize=(10, 8))
                                ax.barh(imp_df['Feature'], imp_df['Importance'])
                                ax.set_title('Neural Network Feature Importance (Permutation Method)')
                                st.pyplot(fig)
                            else:
                                st.warning("Permutation importance for regression neural networks is not implemented yet.")
                    except Exception as e:
                        st.error(f"Error computing feature importance: {str(e)}")
        else:
            importance_fig = feature_importance_plot(model, feature_names)
            st.pyplot(importance_fig)
        
        # Cross-validation
        st.subheader("Cross-Validation")
        cv_folds = st.slider("Number of CV folds", 3, 10, 5)
        
        with st.spinner("Running cross-validation..."):
            if problem_type == "Classification":
                cv_scores = cross_val_score(model, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), cv=cv_folds, scoring='accuracy')
                st.write(f"Cross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            else:  # Regression
                cv_scores = cross_val_score(model, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), cv=cv_folds, scoring='r2')
                st.write(f"Cross-validation R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # SHAP Analysis Section
    st.header("5. SHAP Analysis for Model Interpretability")
    
    st.info("""
    SHAP (SHapley Additive exPlanations) analysis is temporarily disabled in this deployment.
    
    When enabled, this feature would help you:
    - Understand how each feature contributes to model predictions
    - Identify the most important features across your dataset
    - Visualize complex interactions between features
    """)
    
    if st.checkbox("Run SHAP analysis (currently disabled)", value=False):
        st.warning("SHAP analysis is currently disabled due to package dependencies. Please install the SHAP package to enable this feature.")
        
        # Show example of what SHAP would provide
        st.subheader("Example: Potential Regulatory Insights")
        st.markdown("""
        SHAP analysis would help identify the features with the highest impact on predictions that may represent 
        important regulatory elements in the biofilm vs. motile phenotype transition.
        
        These could include:
        - Transcription factors controlling biofilm formation
        - Genes involved in c-di-GMP signaling
        - Two-component systems responding to environmental cues
        
        With SHAP, you could visualize how these features contribute to specific predictions and identify patterns
        that may not be obvious from traditional feature importance plots.
        """)

    # Model export option
    st.header("6. Export Model and Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export model information
        if st.button("Export Model Summary"):
            # Capture model description and parameters
            model_info = io.StringIO()
            with redirect_stdout(model_info):
                print(f"Model Type: {problem_type} - {algorithm}")
                print(f"Feature Count: {len(feature_names)}")
                print("\nModel Parameters:")
                print(model)
                
                print("\nFeature Importance:")
                if hasattr(model, 'named_steps'):
                    model_name = list(model.named_steps.keys())[-1]
                    model_component = model.named_steps[model_name]
                    if hasattr(model_component, 'feature_importances_'):
                        for feature, importance in zip(feature_names, model_component.feature_importances_):
                            print(f"{feature}: {importance:.4f}")
                
                print("\nPerformance Metrics:")
                if problem_type == "Classification":
                    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
                    print("\nClassification Report:")
                    print(classification_report(y_test, y_pred))
                else:
                    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
                    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
            
            # Provide download button
            st.download_button(
                label="Download Model Summary",
                data=model_info.getvalue(),
                file_name="model_summary.txt",
                mime="text/plain"
            )
    
    with col2:
        # Export predictions
        if st.button("Export Predictions"):
            # Create dataframe with predictions
            predictions_df = pd.DataFrame({
                'Sample_ID': X_test.index,
                'Actual': y_test.values,
                'Predicted': y_pred
            })
            
            # Add prediction probabilities for classification
            if problem_type == "Classification" and hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X_test)
                if probas.shape[1] == 2:  # Binary classification
                    predictions_df['Probability_Class_1'] = probas[:, 1]
                else:  # Multi-class
                    for i in range(probas.shape[1]):
                        predictions_df[f'Probability_Class_{i}'] = probas[:, i]
            
            # Convert to CSV
            csv = predictions_df.to_csv(index=False)
            
            # Provide download button
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="model_predictions.csv",
                mime="text/csv"
            )
