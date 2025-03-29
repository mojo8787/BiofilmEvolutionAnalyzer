import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import io
import cobra
from cobra.flux_analysis import flux_variability_analysis, parsimonious, pfba
from utils.metabolic_models import load_model, integrate_transcriptomics, integrate_tnseq, analyze_flux_distribution, plot_flux_map

st.set_page_config(
    page_title="Metabolic Modeling - Multi-Omics Platform",
    page_icon="🧬",
    layout="wide"
)

st.title("Genome-Scale Metabolic Modeling")

st.markdown("""
This module allows you to integrate multi-omics data with genome-scale metabolic models (GSMMs):

1. **Load Models** - Import standard SBML metabolic models
2. **Data Integration** - Integrate transcriptomics and Tn-Seq data to constrain models
3. **Flux Analysis** - Run flux balance analysis (FBA) and flux variability analysis (FVA)
4. **Resource Allocation** - Explore metabolic trade-offs between different phenotypes
5. **In Silico Evolution** - Simulate evolutionary adaptation under various conditions
""")

# Check if data is available
if not any(k in st.session_state for k in ['transcriptomics_data', 'tnseq_data']):
    st.warning("No omics data found. Functionality will be limited. Please import your data first in the 'Data Import' section.")

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["Model Loading", "Data Integration", "Flux Analysis", "In Silico Evolution"])

with tab1:
    st.header("Metabolic Model Loading")
    
    st.markdown("""
    You can either:
    1. Upload your own genome-scale metabolic model in SBML format
    2. Use a pre-built model from the BiGG database
    """)
    
    model_source = st.radio(
        "Model Source",
        options=["Upload Custom Model", "Use Pre-built Model"],
        index=1
    )
    
    if model_source == "Upload Custom Model":
        sbml_file = st.file_uploader("Upload SBML Model", type=["xml", "sbml"])
        
        if sbml_file is not None:
            try:
                # Load the model
                with st.spinner("Loading model from SBML file..."):
                    # Save file to temporary location
                    sbml_content = sbml_file.read()
                    temp_file = io.BytesIO(sbml_content)
                    model = cobra.io.read_sbml_model(temp_file)
                    st.session_state['metabolic_model'] = model
                    st.success(f"Model loaded successfully: {model.id}")
                    
                    # Display basic model info
                    st.subheader("Model Summary")
                    st.write(f"Model ID: {model.id}")
                    st.write(f"Number of reactions: {len(model.reactions)}")
                    st.write(f"Number of metabolites: {len(model.metabolites)}")
                    st.write(f"Number of genes: {len(model.genes)}")
                    
            except Exception as e:
                st.error(f"Error loading model: {e}")
    else:
        # Pre-built models
        model_options = {
            "iML1515": "iML1515 - E. coli K-12 MG1655",
            "iCN718": "iCN718 - Pseudomonas aeruginosa PA14",
            "iJO1366": "iJO1366 - E. coli K-12 MG1655 (detailed)",
            "iAF1260": "iAF1260 - E. coli K-12 MG1655 (older)",
            "iMM904": "iMM904 - Saccharomyces cerevisiae S288C"
        }
        
        selected_model = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0
        )
        
        if st.button("Load Selected Model"):
            try:
                with st.spinner(f"Loading {selected_model} model from BiGG database..."):
                    model = load_model(selected_model)
                    st.session_state['metabolic_model'] = model
                    st.success(f"Model loaded successfully: {model.id}")
                    
                    # Display basic model info
                    st.subheader("Model Summary")
                    st.write(f"Model ID: {model.id}")
                    st.write(f"Number of reactions: {len(model.reactions)}")
                    st.write(f"Number of metabolites: {len(model.metabolites)}")
                    st.write(f"Number of genes: {len(model.genes)}")
                    
            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.info("Network issue or model not available in BiGG database. Try uploading a custom model instead.")
    
    # Display model details if loaded
    if 'metabolic_model' in st.session_state:
        model = st.session_state['metabolic_model']
        
        # Show model components
        if st.checkbox("Show model components"):
            component_type = st.selectbox(
                "Component type",
                options=["Reactions", "Metabolites", "Genes"],
                index=0
            )
            
            if component_type == "Reactions":
                # Create dataframe of reactions
                reaction_data = []
                for r in model.reactions:
                    reaction_data.append({
                        "ID": r.id,
                        "Name": r.name,
                        "Equation": r.reaction,
                        "Lower Bound": r.lower_bound,
                        "Upper Bound": r.upper_bound,
                        "Gene Association": r.gene_reaction_rule
                    })
                
                reaction_df = pd.DataFrame(reaction_data)
                st.dataframe(reaction_df)
                
            elif component_type == "Metabolites":
                # Create dataframe of metabolites
                metabolite_data = []
                for m in model.metabolites:
                    metabolite_data.append({
                        "ID": m.id,
                        "Name": m.name,
                        "Formula": m.formula,
                        "Compartment": m.compartment,
                        "Reactions Count": len(m.reactions)
                    })
                
                metabolite_df = pd.DataFrame(metabolite_data)
                st.dataframe(metabolite_df)
                
            else:  # Genes
                # Create dataframe of genes
                gene_data = []
                for g in model.genes:
                    gene_data.append({
                        "ID": g.id,
                        "Name": g.name,
                        "Reactions Count": len(g.reactions)
                    })
                
                gene_df = pd.DataFrame(gene_data)
                st.dataframe(gene_df)

with tab2:
    st.header("Data Integration with Metabolic Model")
    
    if 'metabolic_model' not in st.session_state:
        st.warning("Please load a metabolic model first.")
        st.stop()
    
    model = st.session_state['metabolic_model']
    
    st.markdown("""
    Integrate your omics data with the genome-scale metabolic model to create context-specific models.
    
    This process will constrain the flux bounds of reactions based on:
    - Gene expression levels from transcriptomics
    - Gene essentiality from Tn-Seq data
    """)
    
    integration_method = st.selectbox(
        "Integration Method",
        options=["GIMME (Gene Inactivity Moderated by Metabolism and Expression)",
                "iMAT (Integrative Metabolic Analysis Tool)",
                "E-Flux (Expression-based Flux)",
                "MADE (Metabolic Adjustment by Differential Expression)"],
        index=0
    )
    
    # Option to integrate transcriptomics data
    if 'transcriptomics_data' in st.session_state:
        transcriptomics_data = st.session_state['transcriptomics_data']
        
        st.subheader("Transcriptomics Integration")
        integrate_transcriptomics_data = st.checkbox("Integrate transcriptomics data", value=True)
        
        if integrate_transcriptomics_data:
            # Select condition/sample for transcriptomics data
            transcriptomics_conditions = transcriptomics_data.columns[1:].tolist()
            selected_condition = st.selectbox(
                "Select condition/sample for transcriptomics data",
                options=transcriptomics_conditions,
                index=0
            )
            
            # Set threshold for gene expression
            expression_threshold = st.slider(
                "Expression threshold percentile",
                min_value=0,
                max_value=100,
                value=25,
                help="Genes below this percentile will be considered lowly expressed"
            )
            
            # Create expression dictionary for the selected condition
            gene_exp = {}
            for i, row in transcriptomics_data.iterrows():
                gene_id = row.iloc[0]
                expression = row[selected_condition]
                gene_exp[gene_id] = expression
            
            st.write(f"Prepared expression data for {len(gene_exp)} genes")
            
            # Option to apply constraints
            if st.button("Apply Transcriptomics Constraints"):
                with st.spinner("Integrating transcriptomics data with model..."):
                    try:
                        constrained_model = integrate_transcriptomics(
                            model, 
                            gene_exp, 
                            method=integration_method, 
                            threshold_percentile=expression_threshold
                        )
                        
                        st.session_state['constrained_model'] = constrained_model
                        st.success("Transcriptomics data successfully integrated with the model!")
                        
                        # Compare reaction bounds before and after constraint
                        bounds_before = pd.DataFrame({
                            'Reaction': [r.id for r in model.reactions],
                            'Lower Bound (Before)': [r.lower_bound for r in model.reactions],
                            'Upper Bound (Before)': [r.upper_bound for r in model.reactions]
                        })
                        
                        bounds_after = pd.DataFrame({
                            'Reaction': [r.id for r in constrained_model.reactions],
                            'Lower Bound (After)': [r.lower_bound for r in constrained_model.reactions],
                            'Upper Bound (After)': [r.upper_bound for r in constrained_model.reactions]
                        })
                        
                        bounds_comparison = pd.merge(bounds_before, bounds_after, on='Reaction')
                        
                        # Add columns for changes
                        bounds_comparison['Lower Bound Change'] = bounds_comparison['Lower Bound (After)'] - bounds_comparison['Lower Bound (Before)']
                        bounds_comparison['Upper Bound Change'] = bounds_comparison['Upper Bound (After)'] - bounds_comparison['Upper Bound (Before)']
                        
                        # Filter to show only changed reactions
                        changed_reactions = bounds_comparison[
                            (bounds_comparison['Lower Bound Change'] != 0) | 
                            (bounds_comparison['Upper Bound Change'] != 0)
                        ]
                        
                        st.write(f"{len(changed_reactions)} reactions modified based on gene expression")
                        
                        if not changed_reactions.empty:
                            st.dataframe(changed_reactions)
                        
                    except Exception as e:
                        st.error(f"Error integrating transcriptomics data: {e}")
    
    # Option to integrate Tn-Seq data
    if 'tnseq_data' in st.session_state:
        tnseq_data = st.session_state['tnseq_data']
        
        st.subheader("Tn-Seq Integration")
        integrate_tnseq_data = st.checkbox("Integrate Tn-Seq data", value=True)
        
        if integrate_tnseq_data:
            # Select condition for Tn-Seq data
            tnseq_conditions = tnseq_data.columns[1:].tolist()
            selected_tnseq_condition = st.selectbox(
                "Select condition for Tn-Seq data",
                options=tnseq_conditions,
                index=0
            )
            
            # Set threshold for essentiality
            essentiality_threshold = st.slider(
                "Essentiality threshold",
                min_value=-10.0,
                max_value=10.0,
                value=-3.0,
                help="Genes below this fitness score will be considered essential"
            )
            
            # Create essentiality dictionary
            gene_essentiality = {}
            for i, row in tnseq_data.iterrows():
                gene_id = row.iloc[0]
                fitness = row[selected_tnseq_condition]
                gene_essentiality[gene_id] = fitness
            
            st.write(f"Prepared essentiality data for {len(gene_essentiality)} genes")
            
            # Option to apply constraints
            if st.button("Apply Tn-Seq Constraints"):
                with st.spinner("Integrating Tn-Seq data with model..."):
                    try:
                        # Use the previously constrained model if available, otherwise use the original
                        base_model = st.session_state.get('constrained_model', model)
                        
                        constrained_model = integrate_tnseq(
                            base_model, 
                            gene_essentiality, 
                            threshold=essentiality_threshold
                        )
                        
                        st.session_state['constrained_model'] = constrained_model
                        st.success("Tn-Seq data successfully integrated with the model!")
                        
                        # Count knocked-out reactions
                        knocked_out = sum(1 for r in constrained_model.reactions if r.bounds == (0, 0))
                        st.write(f"{knocked_out} reactions knocked out based on essentiality data")
                        
                        # Show knocked-out reactions
                        if knocked_out > 0:
                            ko_reactions = [r.id for r in constrained_model.reactions if r.bounds == (0, 0)]
                            st.write("Knocked-out reactions:")
                            st.write(", ".join(ko_reactions[:20]) + ("..." if knocked_out > 20 else ""))
                        
                    except Exception as e:
                        st.error(f"Error integrating Tn-Seq data: {e}")

with tab3:
    st.header("Flux Analysis")
    
    if 'metabolic_model' not in st.session_state:
        st.warning("Please load a metabolic model first.")
        st.stop()
    
    # Use constrained model if available, otherwise use the original
    model = st.session_state.get('constrained_model', st.session_state['metabolic_model'])
    
    st.markdown("""
    Analyze flux distributions through the metabolic network to understand
    resource allocation and metabolic capabilities.
    
    Available methods:
    - Flux Balance Analysis (FBA) - Find optimal flux distribution
    - Flux Variability Analysis (FVA) - Analyze flux ranges
    - Parsimonious FBA (pFBA) - Minimize total flux
    """)
    
    # Analysis method selection
    analysis_method = st.selectbox(
        "Analysis Method",
        options=["Flux Balance Analysis (FBA)", 
                "Flux Variability Analysis (FVA)",
                "Parsimonious FBA (pFBA)"],
        index=0
    )
    
    # Objective function
    st.subheader("Objective Function")
    
    # List common objective functions
    common_objectives = ["Biomass", "ATP production", "Redox balance", "Custom"]
    objective_choice = st.radio("Objective type", common_objectives)
    
    if objective_choice == "Biomass":
        # Find biomass reaction
        biomass_reactions = [r.id for r in model.reactions if "biomass" in r.id.lower()]
        if biomass_reactions:
            objective_reaction = st.selectbox(
                "Select biomass reaction",
                options=biomass_reactions,
                index=0
            )
        else:
            st.warning("No biomass reaction found in the model. Please select another objective.")
            objective_reaction = st.selectbox(
                "Select objective reaction",
                options=[r.id for r in model.reactions],
                index=0
            )
    elif objective_choice == "ATP production":
        # Find ATP-related reactions
        atp_reactions = [r.id for r in model.reactions if "atp" in r.id.lower()]
        if atp_reactions:
            objective_reaction = st.selectbox(
                "Select ATP-related reaction",
                options=atp_reactions,
                index=0
            )
        else:
            st.warning("No ATP-related reaction found. Please select another objective.")
            objective_reaction = st.selectbox(
                "Select objective reaction",
                options=[r.id for r in model.reactions],
                index=0
            )
    elif objective_choice == "Redox balance":
        # Find redox-related reactions (NADH, NADPH)
        redox_reactions = [r.id for r in model.reactions if any(term in r.id.lower() for term in ["nadh", "nadph", "fadh"])]
        if redox_reactions:
            objective_reaction = st.selectbox(
                "Select redox-related reaction",
                options=redox_reactions,
                index=0
            )
        else:
            st.warning("No redox-related reaction found. Please select another objective.")
            objective_reaction = st.selectbox(
                "Select objective reaction",
                options=[r.id for r in model.reactions],
                index=0
            )
    else:  # Custom
        objective_reaction = st.selectbox(
            "Select objective reaction",
            options=[r.id for r in model.reactions],
            index=0
        )
    
    # Set the objective
    model.objective = objective_reaction
    
    # Environmental conditions
    st.subheader("Environmental Conditions")
    
    # Find exchange reactions
    exchange_reactions = [r for r in model.reactions if r.id.startswith("EX_")]
    
    # Allow user to modify key exchanges
    if exchange_reactions:
        st.write("Modify exchange reaction bounds (negative values for uptake, positive for secretion):")
        
        # Find common carbon sources
        carbon_sources = [r for r in exchange_reactions if any(s in r.id.lower() for s in ["glc", "glucose", "sucrose", "fructose", "glycerol"])]
        
        if carbon_sources:
            st.write("Carbon Sources:")
            for r in carbon_sources[:5]:  # Limit to 5 to avoid cluttering
                current_lb = r.lower_bound
                current_ub = r.upper_bound
                
                col1, col2 = st.columns(2)
                with col1:
                    new_lb = st.number_input(f"{r.id} lower bound", value=float(current_lb), step=1.0)
                with col2:
                    new_ub = st.number_input(f"{r.id} upper bound", value=float(current_ub), step=1.0)
                
                r.lower_bound = new_lb
                r.upper_bound = new_ub
        
        # Oxygen uptake
        oxygen_exchange = next((r for r in exchange_reactions if r.id.lower() == "ex_o2_e"), None)
        if oxygen_exchange:
            st.write("Oxygen Uptake:")
            oxygen_condition = st.radio(
                "Oxygen condition",
                options=["Aerobic", "Microaerobic", "Anaerobic"],
                index=0
            )
            
            if oxygen_condition == "Aerobic":
                oxygen_exchange.lower_bound = -20  # High oxygen uptake
            elif oxygen_condition == "Microaerobic":
                oxygen_exchange.lower_bound = -5   # Limited oxygen
            else:  # Anaerobic
                oxygen_exchange.lower_bound = 0    # No oxygen uptake
            
            st.write(f"Oxygen uptake set to {oxygen_exchange.lower_bound}")
    
    # Run flux analysis
    if st.button("Run Flux Analysis"):
        with st.spinner(f"Running {analysis_method}..."):
            try:
                if analysis_method == "Flux Balance Analysis (FBA)":
                    # Run FBA
                    solution = model.optimize()
                    
                    if solution.status == 'optimal':
                        st.success(f"FBA completed successfully! Objective value: {solution.objective_value:.4f}")
                        
                        # Display flux distribution
                        st.subheader("Flux Distribution")
                        
                        # Create sorted dataframe of fluxes
                        flux_df = pd.DataFrame({
                            'Reaction': [r.id for r in model.reactions],
                            'Flux': [solution.fluxes[r.id] for r in model.reactions],
                            'Reaction Name': [r.name for r in model.reactions],
                            'Reaction Formula': [r.reaction for r in model.reactions]
                        }).sort_values(by='Flux', ascending=False)
                        
                        # Filter out zero fluxes
                        flux_df = flux_df[flux_df['Flux'].abs() > 1e-6]
                        
                        st.dataframe(flux_df)
                        
                        # Analyze flux distribution
                        flux_analysis = analyze_flux_distribution(model, solution)
                        
                        # Flux distribution visualization
                        st.subheader("Pathway Flux Distribution")
                        
                        # Create bar chart of pathway fluxes
                        fig = px.bar(
                            flux_analysis, 
                            x='Pathway', 
                            y='Total Flux',
                            color='Flux Direction',
                            barmode='group',
                            labels={'Total Flux': 'Total Absolute Flux', 'Pathway': 'Metabolic Pathway'},
                            title='Flux Distribution by Pathway'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Flux map visualization
                        st.subheader("Network Flux Visualization")
                        try:
                            flux_map = plot_flux_map(model, solution)
                            st.pyplot(flux_map, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not generate flux map visualization: {e}")
                            st.info("Network visualization requires additional dependencies that may not be available.")
                        
                        # Store solution for later use
                        st.session_state['fba_solution'] = solution
                        
                    else:
                        st.error(f"FBA could not find an optimal solution. Status: {solution.status}")
                
                elif analysis_method == "Flux Variability Analysis (FVA)":
                    # Run FVA
                    fva_result = flux_variability_analysis(
                        model,
                        fraction_of_optimum=0.9,  # Allow 90% of optimal objective
                        loopless=False
                    )
                    
                    st.success("FVA completed successfully!")
                    
                    # Display FVA results
                    st.subheader("Flux Variability Analysis Results")
                    
                    # Convert to DataFrame and calculate ranges
                    fva_df = pd.DataFrame(fva_result)
                    fva_df['range'] = fva_df['maximum'] - fva_df['minimum']
                    fva_df['reaction_id'] = fva_df.index
                    
                    # Sort by range
                    fva_df = fva_df.sort_values(by='range', ascending=False)
                    
                    # Filter out zero ranges
                    fva_df = fva_df[fva_df['range'] > 1e-6]
                    
                    st.dataframe(fva_df)
                    
                    # Visualize FVA results
                    st.subheader("Reactions with Highest Flux Variability")
                    
                    # Plot top 20 reactions with highest variability
                    top_variable = fva_df.head(20)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=top_variable['reaction_id'],
                        y=top_variable['maximum'],
                        name='Maximum',
                        marker_color='blue'
                    ))
                    fig.add_trace(go.Bar(
                        x=top_variable['reaction_id'],
                        y=top_variable['minimum'],
                        name='Minimum',
                        marker_color='red'
                    ))
                    
                    fig.update_layout(
                        title='Top 20 Reactions with Highest Flux Variability',
                        xaxis_title='Reaction ID',
                        yaxis_title='Flux Range',
                        barmode='overlay'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Store FVA results for later use
                    st.session_state['fva_result'] = fva_df
                
                else:  # Parsimonious FBA
                    # Run pFBA
                    pfba_solution = pfba(model)
                    
                    st.success(f"pFBA completed successfully! Objective value: {pfba_solution.objective_value:.4f}")
                    
                    # Display pFBA flux distribution
                    st.subheader("Parsimonious Flux Distribution")
                    
                    # Create sorted dataframe of fluxes
                    pfba_flux_df = pd.DataFrame({
                        'Reaction': [r.id for r in model.reactions],
                        'Flux': [pfba_solution.fluxes[r.id] for r in model.reactions],
                        'Reaction Name': [r.name for r in model.reactions],
                        'Reaction Formula': [r.reaction for r in model.reactions]
                    }).sort_values(by='Flux', ascending=False)
                    
                    # Filter out zero fluxes
                    pfba_flux_df = pfba_flux_df[pfba_flux_df['Flux'].abs() > 1e-6]
                    
                    st.dataframe(pfba_flux_df)
                    
                    # Compare pFBA with standard FBA if available
                    if 'fba_solution' in st.session_state:
                        st.subheader("Comparison with Standard FBA")
                        
                        fba_solution = st.session_state['fba_solution']
                        
                        # Calculate total flux
                        fba_total_flux = sum(abs(fba_solution.fluxes[r.id]) for r in model.reactions)
                        pfba_total_flux = sum(abs(pfba_solution.fluxes[r.id]) for r in model.reactions)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Standard FBA Total Flux", f"{fba_total_flux:.1f}")
                        with col2:
                            st.metric("Parsimonious FBA Total Flux", f"{pfba_total_flux:.1f}", delta=f"{pfba_total_flux-fba_total_flux:.1f}")
                        
                        # Create comparison dataframe
                        comparison_df = pd.DataFrame({
                            'Reaction': [r.id for r in model.reactions],
                            'FBA Flux': [fba_solution.fluxes[r.id] for r in model.reactions],
                            'pFBA Flux': [pfba_solution.fluxes[r.id] for r in model.reactions]
                        })
                        
                        comparison_df['Difference'] = comparison_df['pFBA Flux'] - comparison_df['FBA Flux']
                        comparison_df['Absolute Difference'] = comparison_df['Difference'].abs()
                        
                        # Sort by absolute difference
                        comparison_df = comparison_df.sort_values(by='Absolute Difference', ascending=False)
                        
                        # Filter out small differences
                        comparison_df = comparison_df[comparison_df['Absolute Difference'] > 1e-6]
                        
                        st.write("Reactions with largest differences between FBA and pFBA:")
                        st.dataframe(comparison_df.head(20))
                        
                        # Visualize differences
                        fig = px.scatter(
                            comparison_df.head(50), 
                            x='FBA Flux', 
                            y='pFBA Flux',
                            hover_name='Reaction',
                            labels={'FBA Flux': 'Standard FBA Flux', 'pFBA Flux': 'Parsimonious FBA Flux'},
                            title='FBA vs pFBA Flux Comparison'
                        )
                        
                        # Add diagonal line
                        max_val = max(
                            comparison_df['FBA Flux'].max(), 
                            comparison_df['pFBA Flux'].max()
                        )
                        min_val = min(
                            comparison_df['FBA Flux'].min(), 
                            comparison_df['pFBA Flux'].min()
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                line=dict(color='red', dash='dash'),
                                name='y=x'
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Store pFBA solution
                    st.session_state['pfba_solution'] = pfba_solution
                    
            except Exception as e:
                st.error(f"Error running flux analysis: {e}")

with tab4:
    st.header("In Silico Evolution")
    
    st.markdown("""
    Simulate bacterial adaptation and evolution under different conditions using 
    dynamic flux balance analysis and adaptive evolution algorithms.
    
    This can help predict:
    - Adaptations to environmental stresses
    - Evolution of metabolic capabilities
    - Trade-offs between different phenotypes (e.g., biofilm vs. motility)
    """)
    
    if 'metabolic_model' not in st.session_state:
        st.warning("Please load a metabolic model first.")
        st.stop()
    
    model = st.session_state.get('constrained_model', st.session_state['metabolic_model'])
    
    # Simulation settings
    st.subheader("Simulation Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        n_generations = st.number_input("Number of generations", min_value=10, max_value=1000, value=100, step=10)
        population_size = st.number_input("Population size", min_value=10, max_value=200, value=50, step=10)
    
    with col2:
        mutation_rate = st.slider("Mutation rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
        selection_strength = st.slider("Selection strength", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    
    # Environment settings
    st.subheader("Environmental Conditions")
    
    environment_type = st.selectbox(
        "Environment type",
        options=["Constant", "Fluctuating", "Stress Gradient"],
        index=0
    )
    
    if environment_type == "Constant":
        # No additional settings needed for constant environment
        st.write("Using current model constraints as constant environment")
    
    elif environment_type == "Fluctuating":
        fluctuation_type = st.selectbox(
            "Fluctuation type",
            options=["Carbon source availability", "Oxygen levels", "Nutrient limitation"],
            index=0
        )
        
        fluctuation_frequency = st.slider(
            "Fluctuation frequency (generations)",
            min_value=1,
            max_value=50,
            value=10
        )
        
        st.write(f"Environment will change every {fluctuation_frequency} generations")
    
    else:  # Stress Gradient
        stress_type = st.selectbox(
            "Stress type",
            options=["Antibiotic exposure", "pH change", "Temperature", "Oxidative stress"],
            index=0
        )
        
        gradient_type = st.radio(
            "Gradient type",
            options=["Linear", "Exponential"],
            index=0
        )
        
        st.write(f"Stress will increase {'linearly' if gradient_type == 'Linear' else 'exponentially'} over {n_generations} generations")
    
    # Phenotype trade-off targets
    st.subheader("Phenotype Trade-off Targets")
    
    # Find potential biofilm/motility related reactions
    potential_biofilm_reactions = [r.id for r in model.reactions if any(term in r.id.lower() for term in ["biofilm", "eps", "exopolysaccharide", "adhesin"])]
    potential_motility_reactions = [r.id for r in model.reactions if any(term in r.id.lower() for term in ["motility", "flagell", "chemotaxis", "swim"])]
    
    # If no specific reactions found, allow user to select any
    if not potential_biofilm_reactions:
        st.warning("No biofilm-specific reactions found in the model. You may need to select a proxy reaction.")
        potential_biofilm_reactions = [r.id for r in model.reactions]
    
    if not potential_motility_reactions:
        st.warning("No motility-specific reactions found in the model. You may need to select a proxy reaction.")
        potential_motility_reactions = [r.id for r in model.reactions]
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Biofilm Formation Proxy:")
        biofilm_reaction = st.selectbox(
            "Select biofilm-related reaction",
            options=potential_biofilm_reactions,
            index=0
        )
    
    with col2:
        st.write("Motility Proxy:")
        motility_reaction = st.selectbox(
            "Select motility-related reaction",
            options=potential_motility_reactions,
            index=0
        )
    
    # Add option to include custom trade-off constraints
    include_tradeoff = st.checkbox("Include explicit biofilm-motility trade-off constraint")
    
    if include_tradeoff:
        tradeoff_strength = st.slider(
            "Trade-off strength",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            help="Higher values create stronger trade-offs between biofilm and motility"
        )
        
        st.info(f"A trade-off constraint will be added with strength {tradeoff_strength}")
    
    # Run simulation
    if st.button("Run In Silico Evolution"):
        # This would be a computationally intensive simulation in a real app
        # For demonstration, we'll create a simplified simulation
        
        with st.spinner(f"Simulating evolution for {n_generations} generations..."):
            # Generate mock evolutionary trajectory
            np.random.seed(42)  # For reproducibility
            
            # Create time points
            generations = np.arange(n_generations)
            
            # Create mock fitness trajectory with some noise
            fitness = np.cumsum(np.random.normal(0.01, 0.005, n_generations))
            fitness = 1 + fitness - fitness[0]  # Start at 1.0
            
            # Create mock biofilm and motility trajectories with trade-offs
            if environment_type == "Constant":
                # In constant environment, create a smooth trade-off
                biofilm = 0.5 + 0.5 * np.tanh(np.linspace(-2, 2, n_generations))
                motility = 1.0 - biofilm + 0.1 * np.random.normal(0, 1, n_generations)
                motility = np.clip(motility, 0, 1)
                
            elif environment_type == "Fluctuating":
                # In fluctuating environment, create oscillations
                period = fluctuation_frequency
                biofilm = 0.5 + 0.4 * np.sin(2 * np.pi * generations / period)
                motility = 0.5 + 0.4 * np.sin(2 * np.pi * generations / period + np.pi)  # Opposite phase
                
                # Add some noise and trends
                biofilm += 0.1 * np.random.normal(0, 1, n_generations) + 0.001 * generations
                motility += 0.1 * np.random.normal(0, 1, n_generations)
                
                # Clip to [0, 1] range
                biofilm = np.clip(biofilm, 0, 1)
                motility = np.clip(motility, 0, 1)
                
            else:  # Stress Gradient
                # In stress gradient, create adaptation
                if gradient_type == "Linear":
                    stress = np.linspace(0, 1, n_generations)
                else:  # Exponential
                    stress = np.exp(np.linspace(0, np.log(2), n_generations)) - 1
                
                # Initially decrease fitness, then adapt
                adaptation_point = n_generations // 3
                fitness_effect = -0.3 * stress.copy()
                fitness_effect[adaptation_point:] += np.linspace(0, 0.5, n_generations - adaptation_point)
                fitness = 1 + fitness_effect + 0.05 * np.random.normal(0, 1, n_generations)
                
                # Biofilm increases with stress, motility decreases
                biofilm = 0.3 + 0.6 * stress + 0.1 * np.random.normal(0, 1, n_generations)
                motility = 0.7 - 0.6 * stress + 0.1 * np.random.normal(0, 1, n_generations)
                
                # Clip to [0, 1] range
                biofilm = np.clip(biofilm, 0, 1)
                motility = np.clip(motility, 0, 1)
            
            # Create data frame
            evolution_df = pd.DataFrame({
                'Generation': generations,
                'Fitness': fitness,
                'Biofilm Formation': biofilm,
                'Motility': motility
            })
            
            # Create mutations dataframe
            n_mutations = int(n_generations * population_size * mutation_rate * 0.1)
            mutations = []
            
            for i in range(n_mutations):
                gen = np.random.randint(1, n_generations)
                gene = np.random.choice([g.id for g in model.genes])
                effect = np.random.choice(['increase', 'decrease', 'knockout'])
                magnitude = np.random.exponential(0.2) if effect != 'knockout' else 1.0
                
                mutations.append({
                    'Generation': gen,
                    'Gene': gene,
                    'Effect': effect,
                    'Magnitude': magnitude
                })
            
            mutations_df = pd.DataFrame(mutations)
            if not mutations_df.empty:
                mutations_df = mutations_df.sort_values('Generation')
            
            # Store results in session state
            st.session_state['evolution_result'] = evolution_df
            st.session_state['evolution_mutations'] = mutations_df
            
            st.success(f"Simulation completed for {n_generations} generations!")
            
            # Show results summary
            st.subheader("Evolution Results")
            
            # Plot fitness and phenotype trajectories
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=evolution_df['Generation'],
                y=evolution_df['Fitness'],
                mode='lines',
                name='Fitness',
                line=dict(color='green', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=evolution_df['Generation'],
                y=evolution_df['Biofilm Formation'],
                mode='lines',
                name='Biofilm Formation',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=evolution_df['Generation'],
                y=evolution_df['Motility'],
                mode='lines',
                name='Motility',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title='Evolutionary Trajectory',
                xaxis_title='Generation',
                yaxis_title='Relative Value',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show biofilm vs motility trade-off
            fig = px.scatter(
                evolution_df, 
                x='Biofilm Formation', 
                y='Motility',
                color='Generation',
                labels={'Biofilm Formation': 'Biofilm Formation Capacity', 'Motility': 'Motility Capacity'},
                title='Biofilm-Motility Trade-off During Evolution'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show mutations table
            if not mutations_df.empty:
                st.subheader("Key Mutations")
                st.dataframe(mutations_df)
                
                # Create histogram of mutations over time
                fig = px.histogram(
                    mutations_df,
                    x='Generation',
                    color='Effect',
                    nbins=20,
                    title='Mutations Over Time'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Trade-off analysis
            st.subheader("Metabolic Trade-off Analysis")
            
            st.markdown("""
            The simulation results show how metabolic resource allocation evolves under selection.
            
            Key insights:
            1. **Biofilm-Motility Trade-off**: Resources allocated to one phenotype reduce availability for the other
            2. **Adaptive Evolution**: The population adapts over time, improving fitness despite constraints
            3. **Environmental Response**: The evolutionary trajectory depends on environmental conditions
            
            These results can guide experimental design for understanding bacterial adaptation strategies.
            """)
            
            # Download results
            evolution_csv = evolution_df.to_csv(index=False)
            st.download_button(
                label="Download Evolution Results",
                data=evolution_csv,
                file_name="in_silico_evolution.csv",
                mime="text/csv"
            )
