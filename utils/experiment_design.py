"""
Utility functions for experiment design and optimization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


def create_factorial_design(factors, levels):
    """
    Create a full factorial experimental design.
    
    Parameters:
    -----------
    factors : list
        List of factor names.
    levels : dict
        Dictionary with factor names as keys and lists of factor levels as values.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the full factorial design.
    """
    import itertools
    
    # Create all combinations of factor levels
    level_values = [levels[factor] for factor in factors]
    combinations = list(itertools.product(*level_values))
    
    # Create a table of experimental conditions
    experiments = pd.DataFrame(combinations, columns=factors)
    
    return experiments


def create_response_surface_design(factors, centers, ranges, face_centered=False):
    """
    Create a central composite design for response surface methodology.
    
    Parameters:
    -----------
    factors : list
        List of factor names.
    centers : list
        Center point values for each factor.
    ranges : list
        Range of values for each factor.
    face_centered : bool, optional
        Whether to use a face-centered design (alpha = 1).
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the central composite design.
    """
    n_factors = len(factors)
    
    # Set alpha based on design type
    if face_centered:
        alpha = 1.0
    else:
        # Typical value for rotatability
        alpha = (2**n_factors)**(1/4)
    
    # Create factorial points
    factorial_points = []
    for i in range(2**n_factors):
        # Convert i to binary and pad with zeros
        binary = format(i, f'0{n_factors}b')
        # Convert binary to -1/+1 levels
        point = [2*int(bit) - 1 for bit in binary]
        factorial_points.append(point)
    
    # Create axial points
    axial_points = []
    for i in range(n_factors):
        point_plus = [0] * n_factors
        point_minus = [0] * n_factors
        point_plus[i] = alpha
        point_minus[i] = -alpha
        axial_points.append(point_plus)
        axial_points.append(point_minus)
    
    # Create center points
    center_points = [[0] * n_factors] * 5
    
    # Combine all points
    all_points = factorial_points + axial_points + center_points
    
    # Convert to actual factor values
    design = pd.DataFrame(all_points, columns=factors)
    for i, factor in enumerate(factors):
        design[factor] = centers[i] + design[factor] * ranges[i]
    
    return design


def calculate_optimal_designs(X, y, n_designs=10, method='information_gain', model=None):
    """
    Calculate optimal experimental designs based on a model or dataset.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix with existing experimental data.
    y : pandas.Series or numpy.ndarray
        Target variable with existing experimental outcomes.
    n_designs : int, optional
        Number of optimal designs to generate.
    method : str, optional
        Method to use for design optimization ('information_gain', 'active_learning', 'bayesian').
    model : object, optional
        Pre-trained model. If None, a new model will be trained.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with the optimal experimental designs.
    """
    # If no model is provided, train a default model
    if model is None:
        if method == 'bayesian':
            # Use Gaussian Process for Bayesian optimization
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            model = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True)
        else:
            # Use Random Forest for other methods
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Train the model
        model.fit(X, y)
    
    # Generate a grid of potential experiments
    # For simplicity, we'll use a uniform grid over the feature space
    n_features = X.shape[1]
    grid_points = min(5, int(np.power(1000, 1/n_features)))  # Limit the number of points per dimension
    
    # Define feature ranges based on existing data
    feature_min = X.min(axis=0)
    feature_max = X.max(axis=0)
    
    # Create grid points for each feature
    grid_values = [np.linspace(feature_min[i], feature_max[i], grid_points) for i in range(n_features)]
    
    # Create all combinations of grid points
    from itertools import product
    grid_combinations = list(product(*grid_values))
    
    # Convert to DataFrame
    feature_names = X.columns if isinstance(X, pd.DataFrame) else [f'Feature_{i+1}' for i in range(n_features)]
    grid_df = pd.DataFrame(grid_combinations, columns=feature_names)
    
    # Calculate design criterion based on method
    if method == 'information_gain':
        # Use variance of predictions as a measure of information gain
        if hasattr(model, 'estimators_'):
            # For random forest, use variance of tree predictions
            tree_preds = np.array([tree.predict(grid_df) for tree in model.estimators_])
            grid_df['criterion'] = np.var(tree_preds, axis=0)
        else:
            # For other models, use a random value (should be replaced with proper implementation)
            grid_df['criterion'] = np.random.uniform(0, 1, len(grid_df))
        
        # Sort by criterion (higher is better)
        grid_df = grid_df.sort_values('criterion', ascending=False)
        
    elif method == 'active_learning':
        # Use variance of predictions as a measure of uncertainty
        if hasattr(model, 'estimators_'):
            # For random forest, use variance of tree predictions
            tree_preds = np.array([tree.predict(grid_df) for tree in model.estimators_])
            grid_df['criterion'] = np.var(tree_preds, axis=0)
        else:
            # For other models, use a random value (should be replaced with proper implementation)
            grid_df['criterion'] = np.random.uniform(0, 1, len(grid_df))
        
        # Sort by criterion (higher is better)
        grid_df = grid_df.sort_values('criterion', ascending=False)
        
    elif method == 'bayesian':
        # For Gaussian Process, use upper confidence bound as acquisition function
        if hasattr(model, 'predict'):
            # Get mean and std predictions
            mean_pred, std_pred = model.predict(grid_df, return_std=True)
            
            # Calculate Upper Confidence Bound (UCB)
            kappa = 2.0  # Exploration parameter
            grid_df['criterion'] = mean_pred + kappa * std_pred
        else:
            # For other models, use a random value (should be replaced with proper implementation)
            grid_df['criterion'] = np.random.uniform(0, 1, len(grid_df))
        
        # Sort by criterion (higher is better)
        grid_df = grid_df.sort_values('criterion', ascending=False)
    
    # Select top designs
    optimal_designs = grid_df.head(n_designs)
    
    return optimal_designs


def identify_knowledge_gaps(X, target_phenotype, min_samples_per_bin=3):
    """
    Identify knowledge gaps in experimental data.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix with existing experimental data.
    target_phenotype : pandas.Series
        Target variable with existing experimental outcomes.
    min_samples_per_bin : int, optional
        Minimum number of samples required in each bin.
    
    Returns:
    --------
    dict
        Dictionary containing knowledge gap information.
    """
    # Simple approach: divide the range of target_phenotype into bins
    # and identify bins with fewer than min_samples_per_bin samples
    n_bins = min(20, len(target_phenotype) // min_samples_per_bin)
    if n_bins < 2:
        n_bins = 2  # Ensure at least 2 bins
    
    # Create histogram
    hist, bin_edges = np.histogram(target_phenotype, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Identify bins with few samples
    sparse_bins = [i for i, count in enumerate(hist) if count < min_samples_per_bin]
    sparse_ranges = [(bin_edges[i], bin_edges[i+1]) for i in sparse_bins]
    
    # Calculate the largest gap in the data
    sorted_values = np.sort(target_phenotype)
    gaps = sorted_values[1:] - sorted_values[:-1]
    largest_gap_idx = np.argmax(gaps)
    largest_gap = gaps[largest_gap_idx]
    largest_gap_range = (sorted_values[largest_gap_idx], sorted_values[largest_gap_idx+1])
    
    # Calculate data density in different regions
    q1, q3 = np.percentile(target_phenotype, [25, 75])
    below_q1 = np.sum(target_phenotype < q1)
    between_q1_q3 = np.sum((target_phenotype >= q1) & (target_phenotype <= q3))
    above_q3 = np.sum(target_phenotype > q3)
    
    # Determine where data is sparsest
    regions = ['low', 'mid', 'high']
    counts = [below_q1, between_q1_q3, above_q3]
    sparsest_region = regions[np.argmin(counts)]
    
    # Return knowledge gap information
    return {
        'histogram': {
            'counts': hist,
            'bin_edges': bin_edges,
            'bin_centers': bin_centers
        },
        'sparse_bins': sparse_bins,
        'sparse_ranges': sparse_ranges,
        'largest_gap': largest_gap,
        'largest_gap_range': largest_gap_range,
        'quartiles': {
            'q1': q1,
            'q3': q3
        },
        'region_counts': {
            'below_q1': below_q1,
            'between_q1_q3': between_q1_q3,
            'above_q3': above_q3
        },
        'sparsest_region': sparsest_region
    }


def optimize_experiment_sequence(X, y, n_total_experiments, n_initial_experiments=5, method='bayesian'):
    """
    Optimize a sequence of experiments for efficient exploration of the design space.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix with potential experimental conditions.
    y : pandas.Series or numpy.ndarray
        Target variable with expected outcomes (can be empty for initial experiments).
    n_total_experiments : int
        Total number of experiments to plan.
    n_initial_experiments : int, optional
        Number of initial experiments to select randomly.
    method : str, optional
        Method to use for optimization ('bayesian', 'active_learning').
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with the optimal sequence of experiments.
    """
    n_features = X.shape[1]
    
    # If we have fewer rows than n_initial_experiments, use all available data
    if len(X) <= n_initial_experiments:
        sequence = X.copy()
        sequence['experiment_order'] = range(1, len(X) + 1)
        return sequence
    
    # Select initial experiments using Latin Hypercube Sampling for good coverage
    from sklearn.model_selection import KFold
    
    # Normalize features to [0, 1] range for sampling
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    # Use k-fold to get a stratified sample
    kf = KFold(n_splits=n_initial_experiments, shuffle=True, random_state=42)
    
    # Get indices for initial experiments
    initial_indices = []
    for _, test_idx in kf.split(X_norm):
        # Select one sample from each fold
        initial_indices.append(test_idx[0])
    
    # Create a copy of X for the sequence
    sequence = X.iloc[initial_indices].copy()
    sequence['experiment_order'] = range(1, n_initial_experiments + 1)
    
    # If we're done, return the sequence
    if n_total_experiments <= n_initial_experiments:
        return sequence
    
    # Simulate the optimization process
    remaining_indices = list(set(range(len(X))) - set(initial_indices))
    remaining_X = X.iloc[remaining_indices]
    
    # For demonstration purposes, we'll use random selection for the remaining experiments
    # In a real implementation, this would use the specified method to select optimal experiments
    n_remaining = n_total_experiments - n_initial_experiments
    selected_remaining = np.random.choice(len(remaining_X), size=n_remaining, replace=False)
    
    # Add the remaining experiments to the sequence
    remaining_sequence = remaining_X.iloc[selected_remaining].copy()
    remaining_sequence['experiment_order'] = range(n_initial_experiments + 1, n_total_experiments + 1)
    
    # Combine initial and remaining experiments
    sequence = pd.concat([sequence, remaining_sequence])
    
    return sequence


def get_experiment_protocol_template(research_goal, target_phenotype, design_strategy, constraints=None):
    """
    Generate a template for an experimental protocol.
    
    Parameters:
    -----------
    research_goal : str
        The primary research goal (e.g., "Maximize biofilm formation").
    target_phenotype : str
        The target phenotype to measure or optimize.
    design_strategy : str
        The experimental design strategy being used.
    constraints : dict, optional
        Dictionary with experimental constraints.
    
    Returns:
    --------
    str
        Template for an experimental protocol.
    """
    if constraints is None:
        constraints = {}
    
    template = f"""# Experimental Protocol for {research_goal}

## Objective
{research_goal}

## Target Phenotype
{target_phenotype}

## Design Strategy
{design_strategy}

## Materials Required
- Bacterial strain(s)
- Growth media: {constraints.get('media', 'Standard laboratory media')}
- Standard laboratory equipment
- Phenotyping assay reagents for {target_phenotype}

## Procedure
1. Prepare bacterial cultures according to standard laboratory procedures
2. Set up experimental conditions as specified in the design table
3. Measure {target_phenotype} using established protocols
4. Record data and import into the Multi-Omics Platform for analysis

## Data Collection
- Record all experimental parameters
- Measure {target_phenotype} in triplicate
- Include appropriate controls

## Analysis Plan
1. Import data into the Multi-Omics Platform
2. Perform exploratory data analysis
3. Build predictive models using the Machine Learning module
4. Validate findings with follow-up experiments as needed

## Technical Notes
- Resource constraint: {constraints.get('n_experiments', 'Not specified')} experiments
- Technical constraints: {', '.join(constraints.get('tech_constraints', ['None specified']))}
"""
    
    return template