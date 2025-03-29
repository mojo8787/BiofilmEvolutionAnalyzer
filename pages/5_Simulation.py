import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from utils.simulation import BacterialEvolutionSimulator, BiofilmMotilePredictionModel

st.set_page_config(
    page_title="Simulation - Multi-Omics Platform",
    page_icon="🧬",
    layout="wide"
)

st.title("Bacterial Evolution Simulation")

st.markdown("""
This module simulates bacterial evolution under various environmental conditions to explore:

1. **Phenotype Trade-offs** - Understand the relationship between biofilm formation and motility
2. **Adaptive Evolution** - Observe how bacteria adapt to fluctuating environments
3. **Regulatory Mechanisms** - Identify key genes involved in phenotype transitions
4. **Experimental Design Guidance** - Generate testable hypotheses for laboratory studies
""")

# Create tabs for different simulation functionalities
tab1, tab2, tab3 = st.tabs(["Evolution Simulation", "Regulatory Analysis", "Hypothesis Testing"])

with tab1:
    st.header("Bacterial Evolution Simulation")
    
    st.markdown("""
    This simulation models the evolution of a bacterial population under different environmental conditions,
    tracking the trade-off between biofilm formation and motility phenotypes.
    """)
    
    # Simulation parameters
    st.subheader("Simulation Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_genes = st.slider("Number of genes", 100, 5000, 1000, step=100)
        n_samples = st.slider("Population size", 10, 200, 50, step=10)
    
    with col2:
        n_generations = st.slider("Number of generations", 50, 500, 300, step=50)
        mutation_rate = st.slider("Mutation rate", 0.001, 0.05, 0.01, step=0.001, format="%f")
    
    with col3:
        initial_phenotype = st.selectbox(
            "Initial phenotype bias",
            options=["None", "Biofilm", "Motile"],
            index=0
        )
        phenotype_bias = None if initial_phenotype == "None" else initial_phenotype.lower()
    
    # Environment selection
    st.subheader("Environmental Conditions")
    
    environment_type = st.selectbox(
        "Environment type",
        options=["Constant", "Fluctuating", "Stress gradient"],
        index=1
    )
    
    if environment_type == "Constant":
        environment = st.selectbox(
            "Select constant environment",
            options=["Biofilm-favoring", "Motility-favoring", "Neutral"],
            index=0
        )
        # Create environment sequence with the same condition
        environment_sequence = [environment.lower()] * n_generations
        
    elif environment_type == "Fluctuating":
        fluctuation_period = st.slider(
            "Environmental fluctuation period (generations)",
            min_value=10,
            max_value=100,
            value=50,
            step=10
        )
        
        environment_a = st.selectbox(
            "Environment A",
            options=["Biofilm-favoring", "Motility-favoring"],
            index=0
        )
        
        environment_b = "Motility-favoring" if environment_a == "Biofilm-favoring" else "Biofilm-favoring"
        st.write(f"Environment B: {environment_b}")
        
        # Create alternating environment sequence
        n_cycles = n_generations // fluctuation_period + 1
        cycle = ([environment_a.lower()] * (fluctuation_period // 2)) + ([environment_b.lower()] * (fluctuation_period // 2))
        environment_sequence = (cycle * n_cycles)[:n_generations]
        
    else:  # Stress gradient
        stress_type = st.selectbox(
            "Stress type",
            options=["Antibiotic", "Nutrient limitation", "pH change"],
            index=0
        )
        
        adaptation_type = st.selectbox(
            "Expected adaptation",
            options=["Increased biofilm formation", "Increased motility"],
            index=0
        )
        
        # Create gradient environment sequence
        if stress_type == "Antibiotic":
            environment_sequence = ["antibiotic-stress"] * n_generations
        else:
            # For other stresses, start with neutral then gradually add stress
            neutral_period = n_generations // 3
            stress_period = n_generations - neutral_period
            environment_sequence = (["neutral"] * neutral_period) + (["stress-gradient"] * stress_period)
    
    # Run simulation button
    if st.button("Run Evolution Simulation"):
        with st.spinner(f"Simulating bacterial evolution for {n_generations} generations..."):
            # Initialize simulator
            simulator = BacterialEvolutionSimulator(
                n_genes=n_genes,
                n_samples=n_samples,
                n_generations=n_generations
            )
            
            # Initialize population
            simulator.initialize_population(phenotype_bias=phenotype_bias)
            
            # Run evolution
            evolution_history = simulator.evolve(
                environment_sequence=environment_sequence,
                mutation_rate=mutation_rate
            )
            
            # Store simulation results in session state
            st.session_state['evolution_simulator'] = simulator
            st.session_state['evolution_history'] = evolution_history
            
            st.success("Simulation completed successfully!")
    
    # Display results if simulation has been run
    if 'evolution_history' in st.session_state:
        st.subheader("Evolution Results")
        
        # Create visualization
        fig = st.session_state['evolution_simulator'].visualize_evolution()
        st.pyplot(fig)
        
        # Analysis of results
        st.subheader("Simulation Analysis")
        
        analysis = st.session_state['evolution_simulator'].analyze_evolution()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Fitness Improvement", 
                f"{analysis['fitness_improvement']:.4f}",
                f"{(analysis['relative_improvement'] - 1) * 100:.1f}%"
            )
            
            st.metric(
                "Biofilm-Motility Correlation", 
                f"{analysis['biofilm_motility_correlation']:.4f}"
            )
            
        with col2:
            # Environment-specific adaptation
            if 'environment_adaptation' in analysis:
                for env, data in analysis['environment_adaptation'].items():
                    st.write(f"**Adaptation in {env} environment:**")
                    st.write(f"- Fitness change: {data['improvement']:.4f}")
                    st.write(f"- Biofilm shift: {data['phenotype_shift']['biofilm']:.4f}")
                    st.write(f"- Motility shift: {data['phenotype_shift']['motility']:.4f}")
        
        # Top genes that changed during evolution
        st.subheader("Key Genetic Changes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top upregulated genes:**")
            increased_genes = pd.DataFrame(analysis['top_increased_genes'])
            st.dataframe(increased_genes)
            
        with col2:
            st.write("**Top downregulated genes:**")
            decreased_genes = pd.DataFrame(analysis['top_decreased_genes'])
            st.dataframe(decreased_genes)
        
        # Download results
        st.subheader("Download Simulation Results")
        
        csv = st.session_state['evolution_history'].to_csv(index=False)
        st.download_button(
            label="Download Evolution History",
            data=csv,
            file_name="bacterial_evolution_simulation.csv",
            mime="text/csv"
        )

with tab2:
    st.header("Regulatory Network Analysis")
    
    st.markdown("""
    This analysis identifies potential regulatory networks involved in the biofilm-motility transition,
    based on simulated gene expression patterns and phenotype correlations.
    """)
    
    if 'evolution_simulator' not in st.session_state:
        st.warning("Please run a simulation first in the 'Evolution Simulation' tab.")
        st.stop()
    
    simulator = st.session_state['evolution_simulator']
    
    # Simulate gene expression data for regulatory analysis
    st.subheader("Gene Expression Patterns")
    
    # Display gene expression snapshots if available
    if hasattr(simulator, 'gene_expression_snapshots'):
        # Select generation for visualization
        available_generations = list(simulator.gene_expression_snapshots.keys())
        selected_generation = st.selectbox(
            "Select generation to visualize",
            options=available_generations,
            index=len(available_generations)-1
        )
        
        # Get gene expression data for selected generation
        gene_expression = simulator.gene_expression_snapshots[selected_generation]
        
        # Create heatmap of top genes
        st.write("Gene expression heatmap for selected genes:")
        
        # Select top varying genes
        gene_variance = np.var(gene_expression, axis=0)
        top_varying_idx = np.argsort(gene_variance)[::-1][:50]
        
        # Create heatmap data
        heatmap_data = gene_expression[:20, top_varying_idx]
        
        # Get gene names
        gene_names = [simulator.gene_names[i] for i in top_varying_idx]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(heatmap_data, cmap='viridis')
        ax.set_xticks(np.arange(len(gene_names)))
        ax.set_xticklabels(gene_names, rotation=90)
        ax.set_yticks(np.arange(20))
        ax.set_yticklabels([f"Sample {i+1}" for i in range(20)])
        ax.set_title(f"Gene Expression at Generation {selected_generation}")
        fig.colorbar(im, ax=ax, label="Expression Level")
        plt.tight_layout()
        
        st.pyplot(fig)
    
    # Build predictive model
    st.subheader("Phenotype Prediction Model")
    
    if st.button("Build Regulatory Network Model"):
        with st.spinner("Building model to identify key regulatory genes..."):
            # Get final gene expression and phenotypes
            if hasattr(simulator, 'gene_expression_snapshots'):
                # Use final generation data
                final_gen = max(simulator.gene_expression_snapshots.keys())
                gene_expression = simulator.gene_expression_snapshots[final_gen]
                phenotypes = simulator.phenotypes
                
                # Build model
                model = BiofilmMotilePredictionModel(model_type='random_forest')
                model.train(gene_expression, phenotypes)
                
                # Store model in session state
                st.session_state['regulatory_model'] = model
                
                st.success("Regulatory network model built successfully!")
    
    # Display model results if available
    if 'regulatory_model' in st.session_state:
        model = st.session_state['regulatory_model']
        
        # Feature importance plot
        st.write("Important genes for phenotype prediction:")
        importance_fig = model.plot_important_genes(simulator.gene_names)
        st.pyplot(importance_fig)
        
        # Get important genes
        important_genes = model.get_important_genes(simulator.gene_names)
        
        if important_genes:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top biofilm-associated genes:**")
                biofilm_genes = pd.DataFrame(important_genes['biofilm'])
                st.dataframe(biofilm_genes)
            
            with col2:
                st.write("**Top motility-associated genes:**")
                motility_genes = pd.DataFrame(important_genes['motility'])
                st.dataframe(motility_genes)
            
            # Identify regulatory genes among important genes
            biofilm_reg_genes = [g for g in important_genes['biofilm'] 
                               if g['gene'] in simulator.regulatory_genes]
            
            motility_reg_genes = [g for g in important_genes['motility'] 
                                if g['gene'] in simulator.regulatory_genes]
            
            st.subheader("Key Regulatory Genes")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Biofilm regulators:**")
                if biofilm_reg_genes:
                    st.dataframe(pd.DataFrame(biofilm_reg_genes))
                else:
                    st.write("No regulatory genes identified.")
            
            with col2:
                st.write("**Motility regulators:**")
                if motility_reg_genes:
                    st.dataframe(pd.DataFrame(motility_reg_genes))
                else:
                    st.write("No regulatory genes identified.")
            
            # Insights
            st.subheader("Regulatory Insights")
            
            st.markdown("""
            **Potential regulatory mechanisms:**
            
            1. **Direct regulation** - Transcription factors controlling biofilm or motility genes
            2. **Indirect regulation** - Regulators affecting metabolic pathways that influence phenotype
            3. **Feedback mechanisms** - Sensing environmental conditions and adjusting phenotype accordingly
            
            The simulation suggests that the biofilm-motility transition is controlled by a network of
            regulators that respond to environmental signals, likely involving second messengers like c-di-GMP.
            """)

with tab3:
    st.header("Hypothesis Testing and Experimental Design")
    
    st.markdown("""
    Based on simulation results, generate testable hypotheses and experimental designs
    to validate predictions about bacterial phenotype regulation.
    """)
    
    # Check if simulation has been run
    if 'evolution_simulator' not in st.session_state:
        st.warning("Please run a simulation first in the 'Evolution Simulation' tab.")
        st.stop()
    
    # Hypothesis generation
    st.subheader("Hypothesis Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hypothesis_focus = st.selectbox(
            "Hypothesis focus",
            options=["Regulatory mechanisms", "Metabolic trade-offs", "Environmental adaptation"],
            index=0
        )
    
    with col2:
        phenotype_focus = st.selectbox(
            "Phenotype focus",
            options=["Biofilm formation", "Motility", "Biofilm-motility switch"],
            index=2
        )
    
    # Generate hypotheses based on selections
    if st.button("Generate Hypotheses"):
        st.write("### Generated Hypotheses")
        
        if hypothesis_focus == "Regulatory mechanisms":
            if phenotype_focus == "Biofilm formation":
                st.markdown("""
                **Hypothesis 1:** Biofilm formation is primarily regulated by a set of master regulators that respond to environmental signals by modulating c-di-GMP levels.
                
                **Hypothesis 2:** The transition to biofilm phenotype requires coordinated upregulation of exopolysaccharide synthesis genes and downregulation of motility apparatus genes.
                
                **Experimental approach:**
                1. Construct knockout mutants of predicted biofilm regulator genes
                2. Measure biofilm formation capacity under various conditions
                3. Quantify c-di-GMP levels in wild-type vs. mutant strains
                4. Perform RNA-seq to identify affected downstream targets
                """)
                
            elif phenotype_focus == "Motility":
                st.markdown("""
                **Hypothesis 1:** Motility is regulated by a two-component system that senses nutrient availability and modulates flagellar gene expression.
                
                **Hypothesis 2:** The activation of motility requires not only expression of flagellar genes but also metabolic reprogramming to allocate energy resources.
                
                **Experimental approach:**
                1. Construct knockout mutants of predicted motility regulator genes
                2. Perform swimming and swarming assays under varying nutrient conditions
                3. Measure ATP consumption in motile vs. non-motile states
                4. Use ChIP-seq to identify binding sites of key regulators
                """)
                
            else:  # Biofilm-motility switch
                st.markdown("""
                **Hypothesis 1:** The biofilm-motility switch is controlled by a bistable regulatory network centered around c-di-GMP metabolism.
                
                **Hypothesis 2:** Environmental signals are integrated by multiple sensor kinases that converge on a few master regulators controlling the phenotypic switch.
                
                **Experimental approach:**
                1. Engineer reporter strains with fluorescent proteins under biofilm and motility gene promoters
                2. Observe single-cell phenotype switching in microfluidic devices
                3. Manipulate c-di-GMP levels through inducible phosphodiesterases and synthases
                4. Perform time-series transcriptomics during phenotype transitions
                """)
                
        elif hypothesis_focus == "Metabolic trade-offs":
            if phenotype_focus == "Biofilm formation":
                st.markdown("""
                **Hypothesis 1:** Biofilm formation requires significant resource allocation to exopolysaccharide production, creating a metabolic burden.
                
                **Hypothesis 2:** Bacteria in biofilms adopt a different metabolic state optimized for slow growth and stress resistance.
                
                **Experimental approach:**
                1. Perform metabolic flux analysis using 13C-labeled substrates in biofilm vs. planktonic cells
                2. Measure growth rates and biomass yields in strains with varying biofilm capacity
                3. Profile metabolites in different regions of the biofilm structure
                4. Test if supplementation with key metabolites can overcome trade-offs
                """)
                
            elif phenotype_focus == "Motility":
                st.markdown("""
                **Hypothesis 1:** Motility represents a significant energy investment that diverts resources from growth and division.
                
                **Hypothesis 2:** The energetic cost of motility is offset by improved nutrient acquisition in heterogeneous environments.
                
                **Experimental approach:**
                1. Measure ATP consumption rates in motile vs. non-motile isogenic strains
                2. Compare growth yields in homogeneous vs. heterogeneous nutrient environments
                3. Use metabolomics to identify key pathways upregulated during motile growth
                4. Test competitive fitness of motile vs. non-motile strains under various conditions
                """)
                
            else:  # Biofilm-motility switch
                st.markdown("""
                **Hypothesis 1:** The biofilm-motility switch represents a fundamental resource allocation trade-off that optimizes fitness in changing environments.
                
                **Hypothesis 2:** Bacteria maintain a bet-hedging strategy with subpopulations in each phenotypic state to manage uncertain environmental conditions.
                
                **Experimental approach:**
                1. Develop dual reporter system to simultaneously track both phenotypes
                2. Measure metabolic costs of maintaining dual capacity vs. specialized states
                3. Use microfluidics to observe single-cell transitions and correlate with growth rates
                4. Perform competitive fitness assays in fluctuating environments
                """)
                
        else:  # Environmental adaptation
            if phenotype_focus == "Biofilm formation":
                st.markdown("""
                **Hypothesis 1:** Repeated exposure to antibiotics selects for increased biofilm formation capacity.
                
                **Hypothesis 2:** Biofilm adaptation involves not only increased production but structural modifications for enhanced protection.
                
                **Experimental approach:**
                1. Conduct experimental evolution under sub-MIC antibiotic pressure
                2. Compare biofilm architecture before and after adaptation using confocal microscopy
                3. Test evolved strains for cross-protection against different stressors
                4. Sequence evolved lines to identify genetic adaptations
                """)
                
            elif phenotype_focus == "Motility":
                st.markdown("""
                **Hypothesis 1:** Nutrient limitation drives adaptation toward enhanced motility to locate resources.
                
                **Hypothesis 2:** Motility adaptation involves changes in chemotaxis sensitivity as well as flagellar efficiency.
                
                **Experimental approach:**
                1. Conduct experimental evolution in structured environments with nutrient gradients
                2. Analyze swimming patterns and chemotaxis efficiency in evolved strains
                3. Measure expression of motility and chemotaxis genes under different conditions
                4. Test competitive fitness in structured vs. well-mixed environments
                """)
                
            else:  # Biofilm-motility switch
                st.markdown("""
                **Hypothesis 1:** Fluctuating environments select for rapid and efficient switching between biofilm and motile states.
                
                **Hypothesis 2:** The rate of environmental fluctuation determines the optimal balance between switching speed and phenotypic stability.
                
                **Experimental approach:**
                1. Design experimental evolution system with programmable environmental fluctuations
                2. Measure phenotype transition rates before and after adaptation
                3. Analyze genetic changes that influence switching efficiency
                4. Test fitness of evolved strains under different fluctuation frequencies
                """)
    
    # Experimental design section
    st.subheader("Experimental Design Assistant")
    
    st.markdown("""
    Design an experiment to test hypotheses generated from simulation results.
    """)
    
    exp_type = st.selectbox(
        "Experiment type",
        options=[
            "Knockout/mutation study",
            "Gene expression analysis",
            "Experimental evolution",
            "Phenotype characterization",
            "Metabolic analysis"
        ],
        index=2
    )
    
    if exp_type == "Knockout/mutation study":
        # Knockout study form
        st.write("**Knockout/Mutation Study Design**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_genes = st.text_area(
                "Target genes (one per line)",
                height=150,
                help="Enter genes to target for knockout or mutation"
            )
            
            mutation_type = st.selectbox(
                "Modification type",
                options=["Complete knockout", "Point mutation", "Overexpression", "CRISPR interference"],
                index=0
            )
        
        with col2:
            phenotypes = st.multiselect(
                "Phenotypes to measure",
                options=["Biofilm formation", "Swimming motility", "Swarming motility", 
                        "Growth rate", "Antibiotic resistance", "Stress tolerance"],
                default=["Biofilm formation", "Swimming motility"]
            )
            
            conditions = st.multiselect(
                "Test conditions",
                options=["Standard media", "Minimal media", "With antibiotics", 
                        "Nutrient limitation", "Anaerobic", "High osmolarity"],
                default=["Standard media", "With antibiotics"]
            )
        
        st.markdown("### Experimental Protocol")
        
        if target_genes and phenotypes:
            genes_list = [g.strip() for g in target_genes.split("\n") if g.strip()]
            
            st.markdown(f"""
            **Objective:** Determine the role of {len(genes_list)} selected genes in bacterial phenotype regulation
            
            **Strains:**
            - Wild-type control
            - {mutation_type} mutants of: {", ".join(genes_list[:5])}{" and others" if len(genes_list) > 5 else ""}
            
            **Measurements:**
            {", ".join(phenotypes)}
            
            **Conditions:**
            {", ".join(conditions)}
            
            **Expected outcome:**
            Changes in phenotype in mutant strains will reveal the regulatory role of each gene in the biofilm-motility transition.
            
            **Validation:**
            Complement mutant strains to confirm phenotype restoration.
            """)
            
    elif exp_type == "Gene expression analysis":
        # Gene expression analysis form
        st.write("**Gene Expression Analysis Design**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            analysis_type = st.selectbox(
                "Analysis method",
                options=["RNA-seq", "RT-qPCR", "Microarray", "Reporter fusion assays"],
                index=0
            )
            
            conditions = st.multiselect(
                "Experimental conditions",
                options=["Planktonic growth", "Biofilm growth", "Nutrient starvation", 
                        "Subinhibitory antibiotics", "High cell density", "Low cell density"],
                default=["Planktonic growth", "Biofilm growth"]
            )
        
        with col2:
            time_points = st.text_input(
                "Time points (hours, comma separated)",
                value="0, 6, 12, 24, 48",
                help="When to collect samples"
            )
            
            replicates = st.number_input(
                "Number of biological replicates",
                min_value=3,
                max_value=10,
                value=3
            )
        
        st.markdown("### Experimental Protocol")
        
        if conditions and time_points:
            time_points_list = [t.strip() for t in time_points.split(",")]
            
            st.markdown(f"""
            **Objective:** Analyze gene expression patterns during biofilm formation and motility
            
            **Method:** {analysis_type}
            
            **Conditions:**
            {", ".join(conditions)}
            
            **Time points:** {", ".join(time_points_list)} hours
            
            **Replicates:** {replicates} biological replicates
            
            **Data analysis approach:**
            1. Differential expression analysis between conditions
            2. Time-series clustering of expression patterns
            3. Regulatory network inference
            4. Pathway enrichment analysis
            
            **Expected outcome:**
            Identification of co-regulated gene modules and putative regulatory networks controlling phenotype transitions.
            """)
            
    elif exp_type == "Experimental evolution":
        # Experimental evolution form
        st.write("**Experimental Evolution Design**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selection_pressure = st.multiselect(
                "Selection pressure",
                options=["Antibiotic exposure", "Nutrient limitation", "Biofilm selection", 
                        "Motility selection", "Fluctuating environment", "Spatial structure"],
                default=["Fluctuating environment"]
            )
            
            n_generations = st.number_input(
                "Number of generations",
                min_value=100,
                max_value=1000,
                value=300,
                step=50
            )
        
        with col2:
            n_populations = st.number_input(
                "Number of replicate populations",
                min_value=3,
                max_value=20,
                value=6
            )
            
            sequencing = st.checkbox("Whole genome sequencing of evolved strains", value=True)
            phenotyping = st.checkbox("Comprehensive phenotyping", value=True)
        
        st.markdown("### Experimental Protocol")
        
        if selection_pressure:
            st.markdown(f"""
            **Objective:** Evolve bacterial populations to adapt to {" and ".join(selection_pressure)} conditions
            
            **Setup:**
            - {n_populations} independent populations
            - {n_generations} generations of selection
            - Daily transfers with selection regime
            
            **Measurements during evolution:**
            - Fitness assays every 50 generations
            - Biofilm and motility phenotyping
            - Preservation of isolates for later analysis
            
            **Post-evolution analysis:**
            {("- Whole genome sequencing to identify adaptive mutations" if sequencing else "")}
            {("- Comprehensive phenotypic characterization of evolved strains" if phenotyping else "")}
            - Competition assays between ancestral and evolved strains
            - Transcriptomic profiling of evolved populations
            
            **Expected outcome:**
            Identification of genetic adaptations that optimize the biofilm-motility balance under the selected conditions.
            """)
            
    elif exp_type == "Phenotype characterization":
        # Phenotype characterization form
        st.write("**Phenotype Characterization Design**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            phenotypes = st.multiselect(
                "Phenotypes to characterize",
                options=["Biofilm biomass", "Biofilm architecture", "Swimming motility", 
                        "Swarming motility", "Adhesion strength", "EPS production", 
                        "Flagellar expression", "c-di-GMP levels"],
                default=["Biofilm biomass", "Swimming motility"]
            )
            
            strains = st.text_area(
                "Strains to test (one per line)",
                height=100,
                help="Enter strains to characterize"
            )
        
        with col2:
            conditions = st.multiselect(
                "Test conditions",
                options=["Standard media", "Minimal media", "With antibiotics", 
                        "Nutrient limitation", "Anaerobic", "High osmolarity", 
                        "Different temperatures", "Different pH"],
                default=["Standard media", "With antibiotics"]
            )
            
            replicates = st.number_input(
                "Number of biological replicates",
                min_value=3,
                max_value=10,
                value=3
            )
        
        st.markdown("### Experimental Protocol")
        
        if phenotypes and conditions:
            strains_list = [s.strip() for s in strains.split("\n") if s.strip()] if strains else ["Wild-type", "Mutant strains"]
            
            st.markdown(f"""
            **Objective:** Comprehensively characterize the {" and ".join(phenotypes)} phenotypes
            
            **Strains:**
            {", ".join(strains_list[:5])}{" and others" if len(strains_list) > 5 else ""}
            
            **Conditions:**
            {", ".join(conditions)}
            
            **Measurements:**
            {", ".join(phenotypes)}
            
            **Replicates:** {replicates} biological replicates
            
            **Analysis approaches:**
            - Quantitative comparison across strains and conditions
            - Correlation analysis between different phenotypes
            - Microscopy and image analysis for structural characterization
            - Statistical modeling of phenotype relationships
            
            **Expected outcome:**
            Detailed understanding of phenotypic trade-offs and condition-dependent phenotype expression.
            """)
            
    else:  # Metabolic analysis
        # Metabolic analysis form
        st.write("**Metabolic Analysis Design**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            analysis_type = st.selectbox(
                "Analysis method",
                options=["Metabolomics", "Flux Balance Analysis", "13C metabolic flux analysis", 
                        "Respirometry", "ATP measurements", "Enzyme activity assays"],
                index=0
            )
            
            conditions = st.multiselect(
                "Metabolic states to compare",
                options=["Planktonic growth", "Biofilm growth", "High motility", 
                        "Starvation response", "Antibiotic stress", "Oxygen limitation"],
                default=["Planktonic growth", "Biofilm growth"]
            )
        
        with col2:
            key_pathways = st.multiselect(
                "Key pathways of interest",
                options=["Central carbon metabolism", "Energy production", "EPS biosynthesis", 
                        "Amino acid metabolism", "Nucleotide metabolism", "Fatty acid metabolism"],
                default=["Central carbon metabolism", "Energy production"]
            )
            
            replicates = st.number_input(
                "Number of biological replicates",
                min_value=3,
                max_value=10,
                value=3
            )
        
        st.markdown("### Experimental Protocol")
        
        if conditions and key_pathways:
            st.markdown(f"""
            **Objective:** Characterize metabolic changes associated with different bacterial phenotypes
            
            **Method:** {analysis_type}
            
            **Conditions to compare:**
            {", ".join(conditions)}
            
            **Metabolic focus:**
            {", ".join(key_pathways)}
            
            **Replicates:** {replicates} biological replicates
            
            **Analysis approaches:**
            - Differential abundance analysis of metabolites
            - Pathway enrichment analysis
            - Integration with transcriptomic data
            - Metabolic model constraints based on measurements
            
            **Expected outcome:**
            Identification of metabolic shifts and resource allocation changes during phenotype transitions.
            """)
    
    # Summary and download section
    if st.button("Generate Experimental Protocol Document"):
        st.success("Experimental protocol generated!")
        
        st.markdown("""
        ### Summary of Experimental Approach
        
        This experimental design aims to test hypotheses derived from computational simulation of bacterial
        phenotype regulation. The proposed methods will characterize the genetic and metabolic basis
        of the biofilm-motility transition, identify key regulatory components, and validate computational
        predictions.
        
        The results will provide:
        1. Validation of key regulatory genes identified in simulations
        2. Characterization of metabolic trade-offs between phenotypes
        3. Understanding of adaptation mechanisms in varying environments
        4. Insights into potential targets for controlling bacterial behavior
        """)
        
        # Placeholder for protocol download
        st.download_button(
            label="Download Full Protocol",
            data="# Experimental Protocol\n\nThis document contains the detailed experimental protocol generated from the Multi-Omics Integration Platform.",
            file_name="experimental_protocol.md",
            mime="text/markdown"
        )
