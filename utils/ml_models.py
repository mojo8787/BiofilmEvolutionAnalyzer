import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import shap - temporarily disabled

def build_classifier(algorithm, tune_hyperparams=False):
    """
    Build a classification pipeline with preprocessing and model.
    
    Parameters:
    -----------
    algorithm : str
        The classification algorithm to use.
    tune_hyperparams : bool
        Whether to perform hyperparameter tuning.
    
    Returns:
    --------
    sklearn.pipeline.Pipeline
        The classification pipeline.
    """
    # Define preprocessing steps
    steps = [
        ('scaler', StandardScaler())
    ]
    
    # Add model based on algorithm choice
    if algorithm == "Random Forest":
        if tune_hyperparams:
            model = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid={
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                },
                cv=5,
                n_jobs=-1
            )
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                random_state=42
            )
        
    elif algorithm == "Gradient Boosting":
        if tune_hyperparams:
            model = GridSearchCV(
                GradientBoostingClassifier(random_state=42),
                param_grid={
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                cv=5,
                n_jobs=-1
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        
    elif algorithm == "Support Vector Machine":
        if tune_hyperparams:
            model = GridSearchCV(
                SVC(random_state=42, probability=True),
                param_grid={
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto', 0.1, 0.01]
                },
                cv=5,
                n_jobs=-1
            )
        else:
            model = SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42
            )
        
    elif algorithm == "Logistic Regression":
        if tune_hyperparams:
            model = GridSearchCV(
                LogisticRegression(random_state=42, max_iter=1000),
                param_grid={
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet', None],
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                },
                cv=5,
                n_jobs=-1
            )
        else:
            model = LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            )
    
    else:
        raise ValueError(f"Unsupported classification algorithm: {algorithm}")
    
    steps.append(('model', model))
    
    return Pipeline(steps)

def build_regressor(algorithm, tune_hyperparams=False):
    """
    Build a regression pipeline with preprocessing and model.
    
    Parameters:
    -----------
    algorithm : str
        The regression algorithm to use.
    tune_hyperparams : bool
        Whether to perform hyperparameter tuning.
    
    Returns:
    --------
    sklearn.pipeline.Pipeline
        The regression pipeline.
    """
    # Define preprocessing steps
    steps = [
        ('scaler', StandardScaler())
    ]
    
    # Add model based on algorithm choice
    if algorithm == "Random Forest":
        if tune_hyperparams:
            model = GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid={
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                },
                cv=5,
                n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                random_state=42
            )
        
    elif algorithm == "Gradient Boosting":
        if tune_hyperparams:
            model = GridSearchCV(
                GradientBoostingRegressor(random_state=42),
                param_grid={
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                cv=5,
                n_jobs=-1
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        
    elif algorithm == "Support Vector Machine":
        if tune_hyperparams:
            model = GridSearchCV(
                SVR(),
                param_grid={
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto', 0.1, 0.01]
                },
                cv=5,
                n_jobs=-1
            )
        else:
            model = SVR(
                C=1.0,
                kernel='rbf',
                gamma='scale'
            )
        
    elif algorithm == "Linear Regression":
        # Linear regression has few hyperparameters to tune
        model = LinearRegression()
    
    else:
        raise ValueError(f"Unsupported regression algorithm: {algorithm}")
    
    steps.append(('model', model))
    
    return Pipeline(steps)

def feature_importance_plot(model, feature_names):
    """
    Create a feature importance plot for a trained model.
    
    Parameters:
    -----------
    model : sklearn.pipeline.Pipeline
        Trained model pipeline.
    feature_names : list
        List of feature names.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The feature importance plot.
    """
    # Extract the model from the pipeline
    if hasattr(model, 'named_steps'):
        model_name = list(model.named_steps.keys())[-1]
        model_component = model.named_steps[model_name]
    else:
        model_component = model
    
    # Check if the model has feature_importances_ attribute
    if hasattr(model_component, 'feature_importances_'):
        importances = model_component.feature_importances_
    elif hasattr(model_component, 'coef_'):
        importances = np.abs(model_component.coef_)
        if importances.ndim > 1:  # For multi-class classifiers
            importances = importances.mean(axis=0)
    else:
        # If model doesn't have built-in feature importance
        # Return a placeholder figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Feature importance not available for this model type",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Create sorted indices
    sorted_idx = np.argsort(importances)[::-1]
    
    # Get top 20 features (or all if less than 20)
    n_features = min(20, len(feature_names))
    top_indices = sorted_idx[:n_features]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(n_features)
    
    # Plot horizontal bars
    ax.barh(y_pos, importances[top_indices], align='center')
    ax.set_yticks(y_pos)
    feature_labels = [feature_names[i] if i < len(feature_names) else f"Feature {i}" for i in top_indices]
    ax.set_yticklabels(feature_labels)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top Features by Importance')
    
    plt.tight_layout()
    return fig

def plot_shap_summary(explainer, shap_values, X, feature_names):
    """
    Create a SHAP summary plot for model interpretability.
    This function is temporarily modified to work without SHAP dependency.
    
    Parameters:
    -----------
    explainer : shap.Explainer
        SHAP explainer object.
    shap_values : numpy.ndarray
        SHAP values for each sample and feature.
    X : pandas.DataFrame
        Feature matrix used to compute SHAP values.
    feature_names : list
        List of feature names.
    
    Returns:
    --------
    matplotlib.figure.Figure
        A placeholder figure when SHAP is not available.
    """
    # Create a placeholder figure for now
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.text(0.5, 0.5, "SHAP analysis is temporarily disabled due to package dependencies.",
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=14)
    ax.set_axis_off()
    
    return fig

def plot_confusion_matrix(y_true, y_pred):
    """
    Create a confusion matrix plot for classification models.
    
    Parameters:
    -----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The confusion matrix plot.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    return fig

def get_regulatory_insights(model, feature_names, top_n=20):
    """
    Extract potential regulatory insights from the trained model.
    
    Parameters:
    -----------
    model : sklearn.pipeline.Pipeline
        Trained model pipeline.
    feature_names : list
        List of feature names.
    top_n : int
        Number of top features to analyze.
    
    Returns:
    --------
    dict
        Dictionary with regulatory insights.
    """
    # Extract feature importance
    if hasattr(model, 'named_steps'):
        model_name = list(model.named_steps.keys())[-1]
        model_component = model.named_steps[model_name]
    else:
        model_component = model
    
    # Get feature importance
    if hasattr(model_component, 'feature_importances_'):
        importances = model_component.feature_importances_
    elif hasattr(model_component, 'coef_'):
        importances = np.abs(model_component.coef_)
        if importances.ndim > 1:  # For multi-class classifiers
            importances = importances.mean(axis=0)
    else:
        return {"error": "Model does not provide feature importance information"}
    
    # Get top features
    sorted_idx = np.argsort(importances)[::-1]
    top_indices = sorted_idx[:top_n]
    
    # Create insights dictionary
    insights = {
        "top_regulators": [
            {
                "feature": feature_names[i] if i < len(feature_names) else f"Feature {i}",
                "importance": float(importances[i]),
                "rank": idx + 1
            }
            for idx, i in enumerate(top_indices)
        ],
        "summary": {
            "model_type": type(model_component).__name__,
            "total_features": len(feature_names),
            "importance_concentration": float(sum(importances[top_indices]) / sum(importances))
        }
    }
    
    return insights
