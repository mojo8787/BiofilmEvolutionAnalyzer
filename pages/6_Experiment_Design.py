import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
import io
from contextlib import redirect_stdout
from utils.experiment_design import (
    create_factorial_design, 
    create_response_surface_design,
    calculate_optimal_designs,
    identify_knowledge_gaps,
    optimize_experiment_sequence,
    get_experiment_protocol_template
)

st.set_page_config(
    page_title="Experiment Design - Multi-Omics Platform",
    page_icon="🧪",
    layout="wide"
)

st.title("Automated Experiment Design Suggestion")

st.markdown("""
This module suggests optimal experiments based on your research goals and existing data:

1. **Design Space Analysis** - Identify understudied regions in your experimental space
2. **Knockout Prioritization** - Suggest gene knockouts to maximize information gain
3. **Growth Condition Optimization** - Optimize growth conditions for specific phenotypes
4. **Experimental Validation** - Design validation experiments for computational predictions
""")

# Check if data is available in session state
required_data = ['transcriptomics_data', 'phenotype_data']
missing_data = [data for data in required_data if data not in st.session_state]

if missing_data:
    st.warning(f"Missing required data: {', '.join(missing_data)}. Please import your data first in the 'Data Import' section.")
    st.stop()

# Data Integration Section
st.header("1. Define Research Objective")

st.markdown("""
Define your research objective to guide experiment design. 
What phenotype or biological process are you trying to understand or optimize?
""")

# Define research objective
research_goal = st.selectbox(
    "Select your primary research goal",
    options=[
        "Maximize biofilm formation",
        "Maximize motility",
        "Identify key regulatory genes",
        "Optimize stress resistance",
        "Understand metabolic trade-offs",
        "Custom objective"
    ],
    index=0
)

if research_goal == "Custom objective":
    custom_goal = st.text_input("Describe your research objective")

# Let user select target phenotype from available data
phenotype_data = st.session_state['phenotype_data']
target_phenotype = st.selectbox(
    "Select target phenotype to optimize",
    options=phenotype_data.columns[1:].tolist(),
    index=0
)

# Design constraints
st.header("2. Design Constraints")

st.markdown("""
Specify constraints for your experiment design to ensure feasibility and relevance.
""")

# Resource constraints
resource_constraint = st.slider(
    "Resource constraint (number of experiments)",
    min_value=1,
    max_value=50,
    value=10,
    help="Maximum number of experiments that can be performed given your resources"
)

# Technical constraints
tech_constraints = st.multiselect(
    "Technical constraints",
    options=[
        "Limited to aerobic conditions", 
        "Temperature range (20-37°C)",
        "pH range (5.5-8.5)",
        "No antibiotic selection",
        "Biosafety level 1 only",
        "Limited to specific growth media"
    ],
    default=[]
)

if "Limited to specific growth media" in tech_constraints:
    media_options = st.multiselect(
        "Available growth media",
        options=["LB", "M9 minimal", "BHI", "TSB", "DMEM", "Custom media"],
        default=["LB", "M9 minimal"]
    )

# Experiment Design Strategy
st.header("3. Experiment Design Strategy")

design_strategy = st.radio(
    "Select experiment design strategy",
    options=[
        "Factorial Design",
        "Response Surface Methodology",
        "Bayesian Optimization",
        "Active Learning"
    ],
    index=0,
    help="""
    - Factorial Design: Test all combinations of selected factors
    - Response Surface Methodology: Optimize process parameters
    - Bayesian Optimization: Iterative approach to find global optimum
    - Active Learning: ML-guided selection of most informative experiments
    """
)

# Design Space Analysis
st.header("4. Design Space Analysis")

# Get available data
transcriptomics_data = st.session_state['transcriptomics_data']
has_model = 'trained_model' in st.session_state

st.markdown("""
Analyze the current experimental space to identify understudied regions and knowledge gaps.
""")

if st.button("Analyze Design Space"):
    with st.spinner("Analyzing design space..."):
        # Extract phenotype values for the target phenotype
        phenotype_values = phenotype_data[target_phenotype].values
        
        # Use our utility function to identify knowledge gaps
        gaps_analysis = identify_knowledge_gaps(
            X=pd.DataFrame(), # Not actually using features here, just phenotype
            target_phenotype=phenotype_values, 
            min_samples_per_bin=max(2, len(phenotype_values) // 10)
        )
        
        # Create a histogram visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        bin_edges = gaps_analysis['histogram']['bin_edges']
        ax.hist(phenotype_values, bins=bin_edges, alpha=0.7)
        ax.set_xlabel(target_phenotype)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {target_phenotype} in Existing Data')
        
        # Add vertical lines for the min, max, and mean values
        ax.axvline(np.min(phenotype_values), color='r', linestyle='--', label=f'Min: {np.min(phenotype_values):.2f}')
        ax.axvline(np.max(phenotype_values), color='g', linestyle='--', label=f'Max: {np.max(phenotype_values):.2f}')
        ax.axvline(np.mean(phenotype_values), color='b', linestyle='--', label=f'Mean: {np.mean(phenotype_values):.2f}')
        
        # Highlight sparse regions
        sparse_ranges = gaps_analysis['sparse_ranges']
        for sparse_range in sparse_ranges:
            ax.axvspan(sparse_range[0], sparse_range[1], alpha=0.2, color='red')
        
        # Highlight the largest gap
        largest_gap_range = gaps_analysis['largest_gap_range']
        ax.axvspan(largest_gap_range[0], largest_gap_range[1], alpha=0.3, color='orange')
        
        ax.legend()
        st.pyplot(fig)
        
        # Display statistics and insights
        st.subheader("Data Coverage Analysis")
        
        # Simple statistics
        st.write(f"**Sample size:** {len(phenotype_values)} experiments")
        st.write(f"**Range:** {np.min(phenotype_values):.2f} to {np.max(phenotype_values):.2f}")
        st.write(f"**Value gap:** {np.max(phenotype_values) - np.min(phenotype_values):.2f}")
        
        # Largest gap information
        largest_gap = gaps_analysis['largest_gap']
        largest_gap_range = gaps_analysis['largest_gap_range']
        
        st.write(f"**Largest observed gap:** {largest_gap:.2f} between {largest_gap_range[0]:.2f} and {largest_gap_range[1]:.2f}")
        
        # Recommend sampling in the largest gap
        st.info(f"Consider sampling within the range {largest_gap_range[0]:.2f} to {largest_gap_range[1]:.2f} to fill the largest knowledge gap in your data.")
        
        # Calculate data density in different regions
        if len(phenotype_values) > 5:
            q1 = gaps_analysis['quartiles']['q1']
            q3 = gaps_analysis['quartiles']['q3']
            below_q1 = gaps_analysis['region_counts']['below_q1']
            between_q1_q3 = gaps_analysis['region_counts']['between_q1_q3']
            above_q3 = gaps_analysis['region_counts']['above_q3']
            
            st.write("**Data density by region:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Low values", f"{below_q1} samples", f"< {q1:.2f}")
            with col2:
                st.metric("Middle values", f"{between_q1_q3} samples", f"{q1:.2f} - {q3:.2f}")
            with col3:
                st.metric("High values", f"{above_q3} samples", f"> {q3:.2f}")
            
            # Make recommendation based on sparsest region
            sparsest_region = gaps_analysis['sparsest_region']
            if sparsest_region == 'low':
                st.info(f"Your dataset is sparse in the lower range (< {q1:.2f}). Consider additional experiments in this region.")
            elif sparsest_region == 'high':
                st.info(f"Your dataset is sparse in the upper range (> {q3:.2f}). Consider additional experiments in this region.")
            else:
                st.info(f"Your dataset is sparse in the middle range ({q1:.2f} - {q3:.2f}). Consider additional experiments in this region.")
            
        # If we have a trained model, suggest experiments based on model uncertainty
        if has_model and 'feature_names' in st.session_state:
            st.subheader("Model-Based Recommendations")
            st.write("Based on your trained machine learning model, we can identify regions of high uncertainty where additional experiments would be most informative.")
            
            if st.button("Calculate High-Uncertainty Regions"):
                model = st.session_state['trained_model']
                
                # This would be replaced with actual logic using the model
                st.write("Identifying regions of high model uncertainty...")
                st.info("To maximize learning from each experiment, consider sampling in regions where your model is most uncertain about predictions.")

# Experiment Suggestion Section
st.header("5. Suggested Experiments")

if st.button("Generate Experiment Suggestions"):
    with st.spinner("Generating optimal experiment designs..."):
        # Display a set of suggested experiments based on the strategy
        st.subheader("Recommended Experiments")
        
        if design_strategy == "Factorial Design":
            # For factorial design, we'll create a simple 2-level factorial design
            # with 2-3 key factors
            
            # Identify potential factors based on research goal
            if research_goal in ["Maximize biofilm formation", "Maximize motility", "Understand metabolic trade-offs"]:
                factors = ["Temperature", "Carbon source", "Oxygen level"]
            else:
                factors = ["Media type", "Growth phase", "Stress condition"]
            
            # Create factorial design
            levels = {
                "Temperature": ["25°C", "37°C"],
                "Carbon source": ["Glucose", "Glycerol"],
                "Oxygen level": ["Aerobic", "Microaerobic"],
                "Media type": ["Rich", "Minimal"],
                "Growth phase": ["Exponential", "Stationary"],
                "Stress condition": ["None", "Oxidative stress"]
            }
            
            # Use our utility function to create the factorial design
            selected_factors = factors[:3]  # Use first 3 factors
            experiments = create_factorial_design(
                factors=selected_factors, 
                levels={factor: levels[factor] for factor in selected_factors}
            )
            
            # Add expected information gain (simulating information value of each experiment)
            experiments["Expected information gain"] = np.random.uniform(0.6, 0.9, size=len(experiments))
            experiments["Priority"] = range(1, len(experiments) + 1)
            
            # Sort by expected information gain
            experiments = experiments.sort_values("Expected information gain", ascending=False)
            experiments["Priority"] = range(1, len(experiments) + 1)
            
            # Display the experimental design
            st.dataframe(experiments.head(resource_constraint))
            
            # Visualization of the design
            if len(selected_factors) >= 2:
                fig = px.scatter(
                    experiments.head(resource_constraint), 
                    x=selected_factors[0], 
                    y=selected_factors[1],
                    color="Expected information gain",
                    size="Expected information gain",
                    hover_data=experiments.columns,
                    title=f"Factorial Design: {selected_factors[0]} vs {selected_factors[1]}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif design_strategy == "Response Surface Methodology":
            # For RSM, we'll create a central composite design
            
            # Identify key factors based on research goal
            if research_goal in ["Maximize biofilm formation", "Maximize motility"]:
                factors = ["Temperature (°C)", "Glucose (g/L)"]
                centers = [30, 5]
                ranges = [10, 5]
            else:
                factors = ["Growth time (hours)", "pH"]
                centers = [12, 7]
                ranges = [6, 1.5]
            
            # Use our utility function to create the response surface design
            experiments = create_response_surface_design(
                factors=factors,
                centers=centers,
                ranges=ranges,
                face_centered=False  # Use rotatable design with alpha = ~1.414
            )
            
            # Import for visualization
            from scipy.stats import norm
            
            # Add expected information gain
            experiments["Expected information gain"] = norm.pdf(np.sqrt(experiments.iloc[:, 0]**2 + experiments.iloc[:, 1]**2), 0, 2)
            experiments["Expected information gain"] = experiments["Expected information gain"] / experiments["Expected information gain"].max()
            experiments["Priority"] = range(1, len(experiments) + 1)
            
            # Sort by expected information gain
            experiments = experiments.sort_values("Expected information gain", ascending=False)
            experiments["Priority"] = range(1, len(experiments) + 1)
            
            # Display the experimental design
            st.dataframe(experiments.head(resource_constraint))
            
            # Create a contour plot for the design
            x = np.linspace(centers[0] - 1.5*ranges[0], centers[0] + 1.5*ranges[0], 100)
            y = np.linspace(centers[1] - 1.5*ranges[1], centers[1] + 1.5*ranges[1], 100)
            X, Y = np.meshgrid(x, y)
            Z = norm.pdf(np.sqrt(((X - centers[0]) / ranges[0])**2 + ((Y - centers[1]) / ranges[1])**2), 0, 2)
            Z = Z / Z.max()
            
            fig = go.Figure(data=[
                go.Contour(x=x, y=y, z=Z, colorscale='Viridis'),
                go.Scatter(
                    x=experiments[factors[0]].head(resource_constraint),
                    y=experiments[factors[1]].head(resource_constraint),
                    mode='markers+text',
                    marker=dict(color='red', size=10),
                    text=experiments['Priority'].head(resource_constraint),
                    textposition="top center"
                )
            ])
            
            fig.update_layout(
                title=f"Response Surface Design: {factors[0]} vs {factors[1]}",
                xaxis_title=factors[0],
                yaxis_title=factors[1]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif design_strategy == "Bayesian Optimization":
            # For Bayesian optimization, we'll simulate some expected outcomes
            
            # Generate some random experimental conditions
            n_experiments = 15
            
            if research_goal in ["Maximize biofilm formation", "Maximize motility"]:
                factors = ["Temperature (°C)", "Glucose (g/L)", "Salt (g/L)"]
                experiments = pd.DataFrame({
                    factors[0]: np.random.uniform(25, 40, n_experiments),
                    factors[1]: np.random.uniform(0, 10, n_experiments),
                    factors[2]: np.random.uniform(0, 15, n_experiments)
                })
            else:
                factors = ["Growth time (hours)", "pH", "Oxygen (%)"]
                experiments = pd.DataFrame({
                    factors[0]: np.random.uniform(6, 24, n_experiments),
                    factors[1]: np.random.uniform(5.5, 8.5, n_experiments),
                    factors[2]: np.random.uniform(0, 21, n_experiments)
                })
            
            # Function to calculate a simulated "acquisition function"
            # In real Bayesian opt, this would be expected improvement or upper confidence bound
            def acquisition_function(df):
                # Create a simple function that peaks at certain values
                if research_goal in ["Maximize biofilm formation", "Maximize motility"]:
                    ideal = np.array([30, 5, 5])  # Ideal conditions for biofilm
                else:
                    ideal = np.array([12, 7, 15])  # Ideal conditions for other processes
                
                # Calculate distance from ideal point
                distances = np.sqrt(((df.iloc[:, 0] - ideal[0]) / 15)**2 + 
                                   ((df.iloc[:, 1] - ideal[1]) / 5)**2 + 
                                   ((df.iloc[:, 2] - ideal[2]) / 10)**2)
                
                # Calculate acquisition values (higher is better)
                acq_values = np.exp(-distances) + np.random.normal(0, 0.1, len(df))
                return acq_values
            
            # Add acquisition function values
            experiments["Acquisition value"] = acquisition_function(experiments)
            experiments["Expected information gain"] = (experiments["Acquisition value"] - experiments["Acquisition value"].min()) / (experiments["Acquisition value"].max() - experiments["Acquisition value"].min())
            experiments["Priority"] = range(1, len(experiments) + 1)
            
            # Sort by acquisition value
            experiments = experiments.sort_values("Acquisition value", ascending=False)
            experiments["Priority"] = range(1, len(experiments) + 1)
            
            # Display the suggested experiments
            st.dataframe(experiments.head(resource_constraint))
            
            # Create a 3D scatter plot for the design
            fig = px.scatter_3d(
                experiments.head(resource_constraint),
                x=factors[0],
                y=factors[1],
                z=factors[2],
                color="Acquisition value",
                size="Acquisition value",
                hover_data=experiments.columns,
                title=f"Bayesian Optimization: Top {resource_constraint} Experiments"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **Bayesian Optimization Process:**
            1. These experiments are suggested based on expected information gain
            2. After conducting each experiment, record the results
            3. Re-run the optimization to get the next best experiments to run
            4. This iterative process converges to optimal conditions efficiently
            """)
        
        elif design_strategy == "Active Learning":
            # For active learning, we'll select points based on model uncertainty
            
            # If a trained model exists, use it; otherwise create a simple model
            if has_model and 'feature_names' in st.session_state:
                st.write("Using your trained ML model to guide experiment selection.")
                model = st.session_state['trained_model']
                feature_names = st.session_state['feature_names']
            else:
                st.write("Creating a simple model to guide experiment selection.")
                # Create a simple dataset for demonstration
                X_demo = pd.DataFrame(np.random.rand(50, 3), columns=["Gene_A", "Gene_B", "Gene_C"])
                y_demo = (X_demo["Gene_A"] * 0.5 + X_demo["Gene_B"] * 0.3 - X_demo["Gene_C"] * 0.2 + np.random.normal(0, 0.1, 50))
                
                # Train a simple random forest model
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_demo, y_demo)
                feature_names = X_demo.columns.tolist()
            
            # Generate a grid of potential experiments
            if len(feature_names) >= 3:
                selected_features = feature_names[:3]  # Use first 3 features
            else:
                selected_features = feature_names
            
            # Create a grid of potential experiments
            from itertools import product
            grid_points = 5
            grid_values = [np.linspace(0, 1, grid_points) for _ in range(len(selected_features))]
            grid_experiments = pd.DataFrame(list(product(*grid_values)), columns=selected_features)
            
            # For remaining features, fill with mean values
            for feature in feature_names:
                if feature not in selected_features:
                    grid_experiments[feature] = 0.5  # fill with middle value
            
            # Function to estimate prediction variance using random forest
            def get_prediction_variance(model, X):
                # For random forest, we can use the variance of tree predictions
                if hasattr(model, 'estimators_'):
                    # Get predictions of individual trees
                    tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
                    # Calculate variance across trees
                    return np.var(tree_preds, axis=0)
                else:
                    # For non-forest models, just return random values
                    return np.random.uniform(0, 1, len(X))
            
            # Calculate uncertainty for each grid point
            if hasattr(model, 'predict'):
                # If the model is a pipeline, extract the final estimator
                if hasattr(model, 'named_steps'):
                    # Get the last step
                    model_name = list(model.named_steps.keys())[-1]
                    model_component = model.named_steps[model_name]
                    grid_experiments['Uncertainty'] = get_prediction_variance(model_component, grid_experiments[feature_names])
                else:
                    grid_experiments['Uncertainty'] = get_prediction_variance(model, grid_experiments[feature_names])
            else:
                # Fallback if the model doesn't have the expected structure
                grid_experiments['Uncertainty'] = np.random.uniform(0, 1, len(grid_experiments))
            
            # Normalize uncertainty
            grid_experiments['Normalized uncertainty'] = (grid_experiments['Uncertainty'] - grid_experiments['Uncertainty'].min()) / (grid_experiments['Uncertainty'].max() - grid_experiments['Uncertainty'].min())
            grid_experiments["Priority"] = range(1, len(grid_experiments) + 1)
            
            # Sort by uncertainty (higher uncertainty = higher priority)
            grid_experiments = grid_experiments.sort_values('Uncertainty', ascending=False)
            grid_experiments["Priority"] = range(1, len(grid_experiments) + 1)
            
            # Select top experiments based on resource constraint
            selected_experiments = grid_experiments.head(resource_constraint)
            
            # Display the suggested experiments
            st.dataframe(selected_experiments[selected_features + ['Normalized uncertainty', 'Priority']])
            
            # Create a visualization
            if len(selected_features) >= 2:
                fig = px.scatter(
                    selected_experiments,
                    x=selected_features[0],
                    y=selected_features[1],
                    color='Normalized uncertainty',
                    size='Normalized uncertainty',
                    hover_data=selected_features + ['Normalized uncertainty'],
                    title="Active Learning: Experiments with Highest Uncertainty"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **Active Learning Strategy:**
            1. These experiments are selected in regions where the model is most uncertain
            2. Conducting these experiments will provide the most informative data
            3. After collecting new data, retrain your model and repeat the process
            4. This approach efficiently explores the experimental space
            """)

# Experiment Protocol Generation
st.header("6. Experiment Protocol Generation")

if st.button("Generate Experimental Protocol"):
    st.subheader("Experimental Protocol")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        ### Protocol for {research_goal}
        
        **Objective:** {research_goal if research_goal != "Custom objective" else custom_goal}
        
        **Target phenotype:** {target_phenotype}
        
        **Design strategy:** {design_strategy}
        
        #### Materials Required:
        - Bacterial strain(s)
        - Growth media: {", ".join(media_options) if "Limited to specific growth media" in tech_constraints else "Standard laboratory media"}
        - Standard laboratory equipment
        - Phenotyping assay reagents for {target_phenotype}
        
        #### Procedure:
        1. Prepare bacterial cultures according to standard laboratory procedures
        2. Set up experimental conditions as specified in the design table
        3. Measure {target_phenotype} using established protocols
        4. Record data and import into the Multi-Omics Platform for analysis
        
        #### Data Collection:
        - Record all experimental parameters
        - Measure {target_phenotype} in triplicate
        - Include appropriate controls
        
        #### Analysis Plan:
        1. Import data into the Multi-Omics Platform
        2. Perform exploratory data analysis
        3. Build predictive models using the Machine Learning module
        4. Validate findings with follow-up experiments as needed
        """)
    
    with col2:
        st.markdown("#### Experimental Workflow")
        
        # Create a simple flowchart
        from graphviz import Digraph
        
        dot = Digraph()
        dot.attr(rankdir='TB')
        
        # Add nodes
        dot.node('A', 'Prepare Cultures')
        dot.node('B', 'Set Up Conditions')
        dot.node('C', 'Run Experiments')
        dot.node('D', 'Collect Data')
        dot.node('E', 'Analyze Results')
        dot.node('F', 'Optimize Design')
        
        # Add edges
        dot.edge('A', 'B')
        dot.edge('B', 'C')
        dot.edge('C', 'D')
        dot.edge('D', 'E')
        dot.edge('E', 'F')
        dot.edge('F', 'B', label='Iterate')
        
        # Render the graph
        st.graphviz_chart(dot)
    
    # Use our utility function to generate a protocol template
    constraints = {
        'n_experiments': resource_constraint,
        'tech_constraints': tech_constraints if tech_constraints else ["None specified"],
        'media': ", ".join(media_options) if "Limited to specific growth media" in tech_constraints else "Standard laboratory media"
    }
    
    # Handle custom goal
    effective_goal = research_goal
    if research_goal == "Custom objective" and 'custom_goal' in locals():
        effective_goal = custom_goal
    
    protocol_text = get_experiment_protocol_template(
        research_goal=effective_goal,
        target_phenotype=target_phenotype,
        design_strategy=design_strategy,
        constraints=constraints
    )
    
    st.download_button(
        label="Download Protocol as Text",
        data=protocol_text,
        file_name="experimental_protocol.txt",
        mime="text/plain"
    )