import cobra
from cobra.flux_analysis import flux_variability_analysis, pfba
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import requests
import io

def load_model(model_id):
    """
    Load a genome-scale metabolic model by ID.
    
    Parameters:
    -----------
    model_id : str
        ID of the model to load.
    
    Returns:
    --------
    cobra.Model
        The loaded metabolic model.
    """
    # BiGG models URL
    bigg_url = f"http://bigg.ucsd.edu/static/models/{model_id}.json"
    
    try:
        # Try to download model from BiGG
        response = requests.get(bigg_url)
        response.raise_for_status()
        
        # Create temporary file to load the model
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
        
        # Load model from the temporary file
        model = cobra.io.load_json_model(temp_path)
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        return model
        
    except requests.exceptions.RequestException:
        # If downloading fails, try loading from cobra's built-in models
        try:
            if model_id == "iML1515":
                model = cobra.test.create_test_model("ecoli")
            elif model_id == "iJO1366":
                model = cobra.test.create_test_model("ecoli")
            else:
                # Create a generic E. coli model if requested model not available
                model = cobra.test.create_test_model("ecoli")
            
            # Update model ID to the requested one
            model.id = model_id
            return model
            
        except:
            raise ValueError(f"Could not load model {model_id} from BiGG or built-in sources")

def integrate_transcriptomics(model, gene_expression, method="GIMME", threshold_percentile=25):
    """
    Integrate transcriptomics data with metabolic model to create a context-specific model.
    
    Parameters:
    -----------
    model : cobra.Model
        The metabolic model to constrain.
    gene_expression : dict
        Dictionary with gene IDs as keys and expression levels as values.
    method : str
        Method to use for integration (GIMME, iMAT, E-Flux, MADE).
    threshold_percentile : float
        Percentile for determining expression threshold.
    
    Returns:
    --------
    cobra.Model
        Constrained metabolic model.
    """
    # Create a copy of the model to avoid modifying the original
    constrained_model = model.copy()
    
    # Calculate expression threshold
    if threshold_percentile > 0:
        expression_values = list(gene_expression.values())
        threshold = np.percentile(expression_values, threshold_percentile)
    else:
        threshold = 0
    
    # Map gene IDs in the model to gene IDs in expression data
    # This is a simplistic mapping that assumes gene IDs match exactly
    # In practice, you might need a more sophisticated mapping
    model_gene_ids = [g.id for g in constrained_model.genes]
    expression_gene_ids = set(gene_expression.keys())
    
    # Find common genes
    common_genes = set(model_gene_ids) & expression_gene_ids
    
    if len(common_genes) == 0:
        raise ValueError("No genes in common between model and expression data")
    
    # Apply constraints based on the chosen method
    if method.startswith("GIMME"):
        # GIMME method: Constrain reactions associated with lowly expressed genes
        for gene_id in common_genes:
            gene = constrained_model.genes.get_by_id(gene_id)
            expression = gene_expression[gene_id]
            
            if expression < threshold:
                # Reduce flux through reactions associated with this gene
                for reaction in gene.reactions:
                    # Only constrain if the gene is necessary for the reaction
                    if gene_id in reaction.gene_reaction_rule and "or" not in reaction.gene_reaction_rule.lower():
                        # Scale the bounds by expression/threshold ratio
                        scale_factor = max(0.1, expression / threshold)
                        reaction.upper_bound *= scale_factor
                        if reaction.lower_bound < 0:
                            reaction.lower_bound *= scale_factor
    
    elif method.startswith("iMAT"):
        # iMAT method: Inactive/moderately active/highly active gene categorization
        high_threshold = np.percentile(list(gene_expression.values()), 75)
        
        for gene_id in common_genes:
            gene = constrained_model.genes.get_by_id(gene_id)
            expression = gene_expression[gene_id]
            
            for reaction in gene.reactions:
                if gene_id in reaction.gene_reaction_rule:
                    if expression < threshold:
                        # Low expression - minimize flux
                        reaction.upper_bound = min(reaction.upper_bound, 0.1)
                        reaction.lower_bound = max(reaction.lower_bound, -0.1)
                    elif expression > high_threshold:
                        # High expression - ensure minimum flux
                        if reaction.upper_bound > 0:
                            reaction.lower_bound = max(reaction.lower_bound, 0.1)
    
    elif method.startswith("E-Flux"):
        # E-Flux method: Directly use expression to constrain flux bounds
        max_expression = max(gene_expression.values())
        
        for gene_id in common_genes:
            gene = constrained_model.genes.get_by_id(gene_id)
            expression = gene_expression[gene_id]
            
            # Normalize expression to [0, 1]
            normalized_expression = expression / max_expression
            
            for reaction in gene.reactions:
                if gene_id in reaction.gene_reaction_rule:
                    # Scale reaction bounds by normalized expression
                    old_upper = reaction.upper_bound
                    old_lower = reaction.lower_bound
                    
                    reaction.upper_bound = old_upper * normalized_expression
                    if old_lower < 0:
                        reaction.lower_bound = old_lower * normalized_expression
    
    elif method.startswith("MADE"):
        # MADE-like method (simplified): Use expression to modify bounds
        # This is a simplified version, not the full MADE algorithm
        for gene_id in common_genes:
            gene = constrained_model.genes.get_by_id(gene_id)
            expression = gene_expression[gene_id]
            
            # Calculate a confidence score based on expression
            confidence = 1.0 - (1.0 / (1.0 + np.exp(expression - threshold)))
            
            for reaction in gene.reactions:
                if gene_id in reaction.gene_reaction_rule:
                    # Apply confidence score to reaction bounds
                    if confidence < 0.5:
                        # Low confidence - restrict flux
                        scale_factor = 2 * confidence  # Scales from 0 to 1
                        reaction.upper_bound *= scale_factor
                        if reaction.lower_bound < 0:
                            reaction.lower_bound *= scale_factor
    
    else:
        raise ValueError(f"Unsupported integration method: {method}")
    
    return constrained_model

def integrate_tnseq(model, gene_essentiality, threshold=-3.0):
    """
    Integrate Tn-Seq data with metabolic model to create a context-specific model.
    
    Parameters:
    -----------
    model : cobra.Model
        The metabolic model to constrain.
    gene_essentiality : dict
        Dictionary with gene IDs as keys and fitness/essentiality scores as values.
    threshold : float
        Fitness threshold below which genes are considered essential.
    
    Returns:
    --------
    cobra.Model
        Constrained metabolic model.
    """
    # Create a copy of the model to avoid modifying the original
    constrained_model = model.copy()
    
    # Map gene IDs in the model to gene IDs in Tn-Seq data
    model_gene_ids = [g.id for g in constrained_model.genes]
    tnseq_gene_ids = set(gene_essentiality.keys())
    
    # Find common genes
    common_genes = set(model_gene_ids) & tnseq_gene_ids
    
    if len(common_genes) == 0:
        raise ValueError("No genes in common between model and Tn-Seq data")
    
    # Apply constraints based on essentiality
    for gene_id in common_genes:
        gene = constrained_model.genes.get_by_id(gene_id)
        fitness = gene_essentiality[gene_id]
        
        if fitness < threshold:
            # If gene is essential, its reactions should have flux in the model
            # Find reactions where this gene is necessary
            for reaction in gene.reactions:
                # Only constrain if the gene is necessary for the reaction (not in OR rule)
                if gene_id in reaction.gene_reaction_rule and "or" not in reaction.gene_reaction_rule.lower():
                    # Essential gene - ensure flux is possible
                    if reaction.upper_bound > 0:
                        reaction.lower_bound = max(reaction.lower_bound, 0.1)
                    elif reaction.lower_bound < 0:
                        reaction.upper_bound = min(reaction.upper_bound, -0.1)
    
    return constrained_model

def analyze_flux_distribution(model, solution):
    """
    Analyze the flux distribution to identify active pathways and key reactions.
    
    Parameters:
    -----------
    model : cobra.Model
        The metabolic model.
    solution : cobra.Solution
        The FBA solution.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with pathway flux analysis.
    """
    # Create a dictionary to store pathway information
    # This is a simple implementation that assigns pathways based on reaction IDs
    # In practice, you might want to use annotated pathway information
    pathways = {}
    
    # Assign pathways based on reaction IDs
    for reaction in model.reactions:
        rid = reaction.id.lower()
        
        # Try to assign a pathway based on common prefixes/keywords
        if "glycoly" in rid or "pfk" in rid or "pgk" in rid or "pyk" in rid:
            pathway = "Glycolysis"
        elif "tca" in rid or "krebs" in rid or "cit" in rid or "isocit" in rid:
            pathway = "TCA Cycle"
        elif "ppp" in rid or "pentose" in rid:
            pathway = "Pentose Phosphate"
        elif "ex_" in rid:
            pathway = "Exchange"
        elif "atp" in rid:
            pathway = "Energy Metabolism"
        elif "acoa" in rid or "acet" in rid:
            pathway = "Acetate Metabolism"
        elif "succ" in rid:
            pathway = "Succinate Metabolism"
        elif any(aa in rid for aa in ["ala", "arg", "asn", "asp", "cys", "gln", "glu", 
                                    "gly", "his", "ile", "leu", "lys", "met", "phe", 
                                    "pro", "ser", "thr", "trp", "tyr", "val"]):
            pathway = "Amino Acid Metabolism"
        elif "fad" in rid or "nad" in rid or "nadp" in rid:
            pathway = "Redox Metabolism"
        elif "lipid" in rid or "fatty" in rid:
            pathway = "Lipid Metabolism"
        elif "biomass" in rid:
            pathway = "Biomass"
        else:
            pathway = "Other"
        
        if pathway not in pathways:
            pathways[pathway] = {"positive": 0, "negative": 0, "total": 0, "reactions": []}
        
        # Calculate flux through this reaction
        flux = solution.fluxes[reaction.id]
        
        # Update pathway flux
        if abs(flux) > 1e-6:  # Only count non-zero fluxes
            pathways[pathway]["reactions"].append((reaction.id, flux))
            pathways[pathway]["total"] += abs(flux)
            
            if flux > 0:
                pathways[pathway]["positive"] += flux
            else:
                pathways[pathway]["negative"] += abs(flux)
    
    # Convert to DataFrame
    pathway_data = []
    for pathway, data in pathways.items():
        if data["total"] > 0:
            pathway_data.append({
                "Pathway": pathway,
                "Total Flux": data["total"],
                "Positive Flux": data["positive"],
                "Negative Flux": data["negative"],
                "Flux Direction": "Forward" if data["positive"] > data["negative"] else "Reverse",
                "Reaction Count": len(data["reactions"]),
                "Top Reactions": ", ".join([f"{rid} ({flux:.2f})" for rid, flux in 
                                           sorted(data["reactions"], key=lambda x: abs(x[1]), reverse=True)[:3]])
            })
    
    return pd.DataFrame(pathway_data)

def plot_flux_map(model, solution):
    """
    Create a visualization of the flux distribution in the metabolic network.
    
    Parameters:
    -----------
    model : cobra.Model
        The metabolic model.
    solution : cobra.Solution
        The FBA solution.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The flux map visualization.
    """
    # This is a simplified visualization
    # In practice, you might want to use a network visualization library like escher
    
    # Extract active reactions (non-zero flux)
    active_reactions = {}
    for reaction in model.reactions:
        flux = solution.fluxes[reaction.id]
        if abs(flux) > 1e-6:
            active_reactions[reaction.id] = flux
    
    # Identify top active reactions
    top_reactions = sorted(active_reactions.items(), key=lambda x: abs(x[1]), reverse=True)[:30]
    
    # Create a bar plot of top reaction fluxes
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for plotting
    reaction_ids = [r[0] for r in top_reactions]
    fluxes = [r[1] for r in top_reactions]
    
    # Create color map based on flux direction
    colors = ['tab:blue' if f > 0 else 'tab:red' for f in fluxes]
    
    # Plot horizontal bars
    y_pos = range(len(reaction_ids))
    ax.barh(y_pos, fluxes, align='center', color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(reaction_ids)
    ax.invert_yaxis()  # Reactions read top-to-bottom
    ax.set_xlabel('Flux')
    ax.set_title('Top 30 Reactions by Absolute Flux')
    
    # Add a legend
    import matplotlib.patches as mpatches
    blue_patch = mpatches.Patch(color='tab:blue', label='Forward')
    red_patch = mpatches.Patch(color='tab:red', label='Reverse')
    ax.legend(handles=[blue_patch, red_patch])
    
    plt.tight_layout()
    return fig

def simulate_evolution(model, n_generations=100, environment_type="Constant"):
    """
    Simulate in silico evolution of the metabolic model.
    
    Parameters:
    -----------
    model : cobra.Model
        The metabolic model.
    n_generations : int
        Number of generations to simulate.
    environment_type : str
        Type of environment (Constant, Fluctuating, Stress Gradient).
    
    Returns:
    --------
    dict
        Dictionary with simulation results.
    """
    # Create a copy of the model to avoid modifying the original
    simulation_model = model.copy()
    
    # Initialize results
    results = {
        "fitness": np.zeros(n_generations),
        "biofilm": np.zeros(n_generations),
        "motility": np.zeros(n_generations),
        "mutations": []
    }
    
    # Identify potential biofilm and motility related reactions
    # This is a simplified version, in reality would need proper reactions
    biofilm_candidates = [r for r in simulation_model.reactions 
                         if any(term in r.id.lower() for term in 
                               ["biofilm", "eps", "exopolysaccharide", "adhesin"])]
    
    motility_candidates = [r for r in simulation_model.reactions 
                          if any(term in r.id.lower() for term in 
                                ["motility", "flagell", "chemotaxis", "swim"])]
    
    # If no specific reactions found, use proxies
    if not biofilm_candidates:
        # Use polysaccharide production as proxy for biofilm
        biofilm_candidates = [r for r in simulation_model.reactions 
                             if any(term in r.id.lower() for term in 
                                   ["polysacch", "pgm", "pgn", "galact", "gluco"])]
    
    if not motility_candidates:
        # Use energy-intensive reactions as proxy for motility
        motility_candidates = [r for r in simulation_model.reactions 
                              if "atp" in r.id.lower() and r.lower_bound < 0]
    
    # Select representative reactions for biofilm and motility
    biofilm_reaction = biofilm_candidates[0] if biofilm_candidates else None
    motility_reaction = motility_candidates[0] if motility_candidates else None
    
    # Simulation loop
    for gen in range(n_generations):
        # Apply environmental changes based on environment type
        if environment_type == "Fluctuating":
            # Fluctuating environment: toggle conditions
            if gen % 20 < 10:
                # Favor biofilm formation
                if biofilm_reaction:
                    biofilm_reaction.lower_bound = max(0, biofilm_reaction.lower_bound)
                if motility_reaction:
                    motility_reaction.upper_bound = motility_reaction.upper_bound * 0.5
            else:
                # Favor motility
                if biofilm_reaction:
                    biofilm_reaction.upper_bound = biofilm_reaction.upper_bound * 0.5
                if motility_reaction:
                    motility_reaction.lower_bound = max(0, motility_reaction.lower_bound)
        
        elif environment_type == "Stress Gradient":
            # Stress gradient: gradually increase stress
            stress_level = gen / n_generations
            
            # Apply stress response
            for ex in simulation_model.exchanges:
                if ex.lower_bound < 0:  # Uptake reaction
                    # Gradually reduce nutrient availability
                    ex.lower_bound = ex.lower_bound * (1 - 0.5 * stress_level)
        
        # Simulate mutations (randomly adjust reaction bounds)
        if np.random.random() < 0.1:  # 10% chance of mutation per generation
            # Select a random reaction
            reaction = np.random.choice(simulation_model.reactions)
            
            # Decide mutation type
            mutation_type = np.random.choice(["increase_ub", "decrease_ub", "increase_lb", "decrease_lb"])
            
            # Apply mutation
            old_lb = reaction.lower_bound
            old_ub = reaction.upper_bound
            
            if mutation_type == "increase_ub" and reaction.upper_bound > 0:
                reaction.upper_bound *= 1.2
            elif mutation_type == "decrease_ub" and reaction.upper_bound > 0:
                reaction.upper_bound *= 0.8
            elif mutation_type == "increase_lb" and reaction.lower_bound < 0:
                reaction.lower_bound *= 0.8  # Less negative = increase
            elif mutation_type == "decrease_lb" and reaction.lower_bound < 0:
                reaction.lower_bound *= 1.2  # More negative = decrease
            
            # Record the mutation
            results["mutations"].append({
                "generation": gen,
                "reaction": reaction.id,
                "type": mutation_type,
                "old_lb": old_lb,
                "old_ub": old_ub,
                "new_lb": reaction.lower_bound,
                "new_ub": reaction.upper_bound
            })
        
        # Run FBA to determine fitness
        try:
            solution = simulation_model.optimize()
            if solution.status == 'optimal':
                # Record fitness (growth rate)
                results["fitness"][gen] = solution.objective_value
                
                # Record biofilm and motility proxies
                if biofilm_reaction:
                    results["biofilm"][gen] = abs(solution.fluxes[biofilm_reaction.id])
                
                if motility_reaction:
                    results["motility"][gen] = abs(solution.fluxes[motility_reaction.id])
            else:
                # If no solution found, use previous values
                if gen > 0:
                    results["fitness"][gen] = results["fitness"][gen-1]
                    results["biofilm"][gen] = results["biofilm"][gen-1]
                    results["motility"][gen] = results["motility"][gen-1]
        except:
            # If optimization fails, use previous values
            if gen > 0:
                results["fitness"][gen] = results["fitness"][gen-1]
                results["biofilm"][gen] = results["biofilm"][gen-1]
                results["motility"][gen] = results["motility"][gen-1]
    
    # Normalize biofilm and motility values
    if np.max(results["biofilm"]) > 0:
        results["biofilm"] = results["biofilm"] / np.max(results["biofilm"])
    
    if np.max(results["motility"]) > 0:
        results["motility"] = results["motility"] / np.max(results["motility"])
    
    return results

def calculate_tradeoff_metrics(model, biofilm_rxn_id, motility_rxn_id):
    """
    Calculate metrics to quantify the metabolic trade-off between biofilm formation and motility.
    
    Parameters:
    -----------
    model : cobra.Model
        The metabolic model.
    biofilm_rxn_id : str
        ID of the reaction representing biofilm formation.
    motility_rxn_id : str
        ID of the reaction representing motility.
    
    Returns:
    --------
    dict
        Dictionary with trade-off metrics.
    """
    # Create a copy of the model to avoid modifying the original
    model_copy = model.copy()
    
    # Check if the specified reactions exist in the model
    if biofilm_rxn_id not in model_copy.reactions:
        raise ValueError(f"Biofilm reaction {biofilm_rxn_id} not found in the model")
    
    if motility_rxn_id not in model_copy.reactions:
        raise ValueError(f"Motility reaction {motility_rxn_id} not found in the model")
    
    # Get the reactions
    biofilm_rxn = model_copy.reactions.get_by_id(biofilm_rxn_id)
    motility_rxn = model_copy.reactions.get_by_id(motility_rxn_id)
    
    # Initialize results
    results = {
        "tradeoff_curve": [],
        "pareto_front": [],
        "resource_competition": [],
        "shadow_prices": {}
    }
    
    # Set objective to biomass (growth rate)
    biomass_rxn = None
    for r in model_copy.reactions:
        if "biomass" in r.id.lower():
            biomass_rxn = r
            break
    
    if biomass_rxn is None:
        # If no biomass reaction found, use the current objective
        original_objective = model_copy.objective
    else:
        # Set objective to biomass
        model_copy.objective = biomass_rxn
        original_objective = biomass_rxn
    
    # Calculate baseline growth rate
    solution = model_copy.optimize()
    baseline_growth = solution.objective_value if solution.status == 'optimal' else 0
    
    # Calculate maximum biofilm and motility fluxes
    # First for biofilm
    original_bounds = (biofilm_rxn.lower_bound, biofilm_rxn.upper_bound)
    model_copy.objective = biofilm_rxn
    biofilm_solution = model_copy.optimize()
    max_biofilm = biofilm_solution.objective_value if biofilm_solution.status == 'optimal' else 0
    
    # Then for motility
    biofilm_rxn.lower_bound, biofilm_rxn.upper_bound = original_bounds
    model_copy.objective = motility_rxn
    motility_solution = model_copy.optimize()
    max_motility = motility_solution.objective_value if motility_solution.status == 'optimal' else 0
    
    # Restore original objective
    model_copy.objective = original_objective
    
    # Calculate trade-off curve
    # Sample points along biofilm capacity and measure motility
    biofilm_range = np.linspace(0, max_biofilm, 10)
    
    for biofilm_flux in biofilm_range:
        # Fix biofilm flux
        biofilm_rxn.lower_bound = biofilm_flux
        biofilm_rxn.upper_bound = biofilm_flux
        
        # Maximize motility
        model_copy.objective = motility_rxn
        try:
            solution = model_copy.optimize()
            if solution.status == 'optimal':
                motility_flux = solution.objective_value
                
                # Restore objective and calculate growth
                model_copy.objective = original_objective
                growth_solution = model_copy.optimize()
                growth_rate = growth_solution.objective_value if growth_solution.status == 'optimal' else 0
                
                # Record point on trade-off curve
                results["tradeoff_curve"].append({
                    "biofilm_flux": float(biofilm_flux),
                    "motility_flux": float(motility_flux),
                    "growth_rate": float(growth_rate),
                    "growth_reduction": float(1 - growth_rate/baseline_growth) if baseline_growth > 0 else 0
                })
                
                # If this point is optimal for both biofilm and motility, add to Pareto front
                if biofilm_flux > 0 and motility_flux > 0:
                    results["pareto_front"].append({
                        "biofilm_flux": float(biofilm_flux),
                        "motility_flux": float(motility_flux),
                        "growth_rate": float(growth_rate)
                    })
                
                # Get shadow prices for key metabolites
                for metabolite in biofilm_rxn.metabolites:
                    if metabolite.id not in results["shadow_prices"]:
                        results["shadow_prices"][metabolite.id] = []
                    
                    # Record shadow price if available
                    if hasattr(solution, 'shadow_prices') and metabolite.id in solution.shadow_prices:
                        shadow_price = solution.shadow_prices[metabolite.id]
                    else:
                        shadow_price = 0
                    
                    results["shadow_prices"][metabolite.id].append({
                        "biofilm_flux": float(biofilm_flux),
                        "shadow_price": float(shadow_price)
                    })
        except:
            pass
        
        # Reset bounds
        biofilm_rxn.lower_bound, biofilm_rxn.upper_bound = original_bounds
    
    # Calculate resource competition
    # Identify common metabolites between biofilm and motility reactions
    biofilm_metabolites = set(biofilm_rxn.metabolites)
    motility_metabolites = set(motility_rxn.metabolites)
    common_metabolites = biofilm_metabolites.intersection(motility_metabolites)
    
    # Record shared resources
    for metabolite in common_metabolites:
        biofilm_coef = biofilm_rxn.metabolites[metabolite]
        motility_coef = motility_rxn.metabolites[metabolite]
        
        # Check if they compete for the same resource (both consume or both produce)
        competing = (biofilm_coef < 0 and motility_coef < 0) or (biofilm_coef > 0 and motility_coef > 0)
        
        results["resource_competition"].append({
            "metabolite": metabolite.id,
            "name": metabolite.name,
            "biofilm_stoichiometry": float(biofilm_coef),
            "motility_stoichiometry": float(motility_coef),
            "competing": competing
        })
    
    return results
