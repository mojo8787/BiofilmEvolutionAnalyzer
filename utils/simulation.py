import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

class BacterialEvolutionSimulator:
    """
    Simulator for bacterial evolution under various environmental conditions.
    """
    
    def __init__(self, n_genes=1000, n_samples=100, n_generations=300):
        """
        Initialize the simulator.
        
        Parameters:
        -----------
        n_genes : int
            Number of genes in the simulation.
        n_samples : int
            Number of bacterial samples to simulate.
        n_generations : int
            Number of generations to simulate.
        """
        self.n_genes = n_genes
        self.n_samples = n_samples
        self.n_generations = n_generations
        self.random_state = np.random.RandomState(42)
        
        # Initialize gene properties
        self.gene_names = [f"gene_{i}" for i in range(n_genes)]
        
        # Mark some genes as regulatory
        self.regulatory_genes = self.random_state.choice(
            self.gene_names, 
            size=int(n_genes * 0.05),  # 5% of genes are regulatory
            replace=False
        )
        
        # Mark some genes as biofilm-related
        self.biofilm_genes = self.random_state.choice(
            self.gene_names, 
            size=int(n_genes * 0.1),  # 10% of genes are biofilm-related
            replace=False
        )
        
        # Mark some genes as motility-related
        self.motility_genes = self.random_state.choice(
            [g for g in self.gene_names if g not in self.biofilm_genes], 
            size=int(n_genes * 0.1),  # 10% of genes are motility-related
            replace=False
        )
        
        # Create a gene-gene regulatory network
        self.regulatory_network = self._create_regulatory_network()
        
        # Initialize population
        self.population = None
        self.phenotypes = None
        self.evolution_history = None
    
    def _create_regulatory_network(self):
        """
        Create a simple gene regulatory network.
        
        Returns:
        --------
        dict
            Dictionary representing the regulatory network.
        """
        network = {}
        
        # Each regulatory gene regulates a set of target genes
        for reg_gene in self.regulatory_genes:
            # Number of targets for this regulator
            n_targets = self.random_state.randint(5, 50)
            
            # Select target genes
            targets = self.random_state.choice(
                self.gene_names, 
                size=n_targets,
                replace=False
            )
            
            # Determine if activator or repressor for each target
            regulation_type = {}
            for target in targets:
                # Positive weight = activation, negative = repression
                weight = self.random_state.uniform(-1, 1)
                regulation_type[target] = weight
            
            network[reg_gene] = regulation_type
        
        return network
    
    def initialize_population(self, phenotype_bias=None):
        """
        Initialize a population of bacterial samples.
        
        Parameters:
        -----------
        phenotype_bias : str, optional
            Bias the initial population towards "biofilm" or "motile" phenotype.
        """
        # Initialize gene expression matrix
        self.population = np.zeros((self.n_samples, self.n_genes))
        
        # Generate random initial gene expression
        for i in range(self.n_samples):
            if phenotype_bias == "biofilm":
                # Bias towards biofilm phenotype
                for j, gene in enumerate(self.gene_names):
                    if gene in self.biofilm_genes:
                        # Biofilm genes have higher expression
                        self.population[i, j] = self.random_state.normal(0.7, 0.2)
                    elif gene in self.motility_genes:
                        # Motility genes have lower expression
                        self.population[i, j] = self.random_state.normal(0.3, 0.2)
                    else:
                        # Other genes have random expression
                        self.population[i, j] = self.random_state.normal(0.5, 0.2)
            
            elif phenotype_bias == "motile":
                # Bias towards motile phenotype
                for j, gene in enumerate(self.gene_names):
                    if gene in self.motility_genes:
                        # Motility genes have higher expression
                        self.population[i, j] = self.random_state.normal(0.7, 0.2)
                    elif gene in self.biofilm_genes:
                        # Biofilm genes have lower expression
                        self.population[i, j] = self.random_state.normal(0.3, 0.2)
                    else:
                        # Other genes have random expression
                        self.population[i, j] = self.random_state.normal(0.5, 0.2)
            
            else:
                # No bias, random initialization
                self.population[i, j] = self.random_state.normal(0.5, 0.2)
        
        # Clip values to valid range [0, 1]
        self.population = np.clip(self.population, 0, 1)
        
        # Calculate initial phenotypes
        self._calculate_phenotypes()
    
    def _calculate_phenotypes(self):
        """
        Calculate phenotypes based on gene expression.
        """
        # Initialize phenotype matrix (samples x phenotypes)
        self.phenotypes = np.zeros((self.n_samples, 3))  # [biofilm, motility, fitness]
        
        # Get indices of biofilm and motility genes
        biofilm_indices = [self.gene_names.index(g) for g in self.biofilm_genes]
        motility_indices = [self.gene_names.index(g) for g in self.motility_genes]
        
        for i in range(self.n_samples):
            # Calculate biofilm formation capacity
            biofilm_expression = np.mean(self.population[i, biofilm_indices])
            self.phenotypes[i, 0] = biofilm_expression
            
            # Calculate motility
            motility_expression = np.mean(self.population[i, motility_indices])
            self.phenotypes[i, 1] = motility_expression
            
            # Calculate fitness (with trade-off between biofilm and motility)
            # Fitness depends on environment, will be calculated during evolution
            self.phenotypes[i, 2] = 0
    
    def _apply_regulatory_effects(self):
        """
        Apply regulatory effects to gene expression.
        """
        # Create a copy of the current population
        new_population = self.population.copy()
        
        # Apply regulatory effects
        for i in range(self.n_samples):
            for reg_gene, targets in self.regulatory_network.items():
                reg_index = self.gene_names.index(reg_gene)
                reg_expression = self.population[i, reg_index]
                
                # Apply effect to each target gene
                for target, weight in targets.items():
                    target_index = self.gene_names.index(target)
                    
                    # Calculate regulatory effect
                    effect = weight * reg_expression
                    
                    # Apply effect
                    new_population[i, target_index] += effect
            
            # Ensure values stay in valid range
            new_population[i] = np.clip(new_population[i], 0, 1)
        
        self.population = new_population
    
    def _calculate_fitness(self, environment):
        """
        Calculate fitness based on environment and phenotypes.
        
        Parameters:
        -----------
        environment : str
            The environment type: 'biofilm-favoring', 'motility-favoring', or 'fluctuating'.
        """
        for i in range(self.n_samples):
            biofilm = self.phenotypes[i, 0]
            motility = self.phenotypes[i, 1]
            
            # Base fitness
            base_fitness = 0.5
            
            # Environment-specific fitness contribution
            if environment == 'biofilm-favoring':
                env_fitness = 0.5 * biofilm - 0.1 * motility
            elif environment == 'motility-favoring':
                env_fitness = 0.5 * motility - 0.1 * biofilm
            elif environment == 'fluctuating':
                # In fluctuating environment, need both traits but with trade-off
                env_fitness = 0.3 * biofilm + 0.3 * motility - 0.4 * biofilm * motility
            elif environment == 'antibiotic-stress':
                # In antibiotic stress, biofilm provides protection
                env_fitness = 0.6 * biofilm - 0.2 * motility
            else:
                # Neutral environment
                env_fitness = 0.2 * biofilm + 0.2 * motility
            
            # Calculate total fitness with metabolic cost for expressing both phenotypes
            metabolic_cost = 0.2 * (biofilm + motility)
            self.phenotypes[i, 2] = base_fitness + env_fitness - metabolic_cost
        
        # Ensure fitness is non-negative
        self.phenotypes[:, 2] = np.maximum(0, self.phenotypes[:, 2])
    
    def _selection(self):
        """
        Perform selection based on fitness.
        
        Returns:
        --------
        numpy.ndarray
            Indices of selected individuals.
        """
        # Get fitness values
        fitness = self.phenotypes[:, 2]
        
        # Calculate selection probabilities
        selection_prob = fitness / np.sum(fitness)
        
        # Select individuals (with replacement)
        selected_indices = self.random_state.choice(
            self.n_samples,
            size=self.n_samples,
            p=selection_prob,
            replace=True
        )
        
        return selected_indices
    
    def _mutation(self, mutation_rate=0.01):
        """
        Apply random mutations to the population.
        
        Parameters:
        -----------
        mutation_rate : float
            Probability of a gene mutating.
        """
        # For each individual and gene, apply mutation with probability mutation_rate
        mutation_mask = self.random_state.random((self.n_samples, self.n_genes)) < mutation_rate
        
        # Calculate mutation effects (random changes in gene expression)
        mutation_effect = self.random_state.normal(0, 0.2, size=(self.n_samples, self.n_genes))
        
        # Apply mutations
        self.population[mutation_mask] += mutation_effect[mutation_mask]
        
        # Ensure values stay in valid range
        self.population = np.clip(self.population, 0, 1)
    
    def evolve(self, environment_sequence, mutation_rate=0.01):
        """
        Evolve the population over multiple generations.
        
        Parameters:
        -----------
        environment_sequence : list
            Sequence of environments for each generation.
        mutation_rate : float
            Probability of a gene mutating.
            
        Returns:
        --------
        pandas.DataFrame
            Evolution history.
        """
        # Initialize history to track evolution
        history = {
            'generation': [],
            'environment': [],
            'mean_biofilm': [],
            'mean_motility': [],
            'mean_fitness': [],
            'best_fitness': [],
            'biofilm_motility_correlation': []
        }
        
        # Make sure environment sequence matches number of generations
        if len(environment_sequence) < self.n_generations:
            # Repeat the sequence if needed
            environment_sequence = (environment_sequence * 
                                   (self.n_generations // len(environment_sequence) + 1))[:self.n_generations]
        
        # Store gene expression snapshots at specific generations
        gene_expression_snapshots = {}
        snapshot_generations = [0, self.n_generations // 4, self.n_generations // 2, 
                               3 * self.n_generations // 4, self.n_generations - 1]
        
        # Evolution loop
        for gen in range(self.n_generations):
            # Get current environment
            current_env = environment_sequence[gen]
            
            # Apply regulatory effects
            self._apply_regulatory_effects()
            
            # Calculate phenotypes
            self._calculate_phenotypes()
            
            # Calculate fitness based on environment
            self._calculate_fitness(current_env)
            
            # Record history
            history['generation'].append(gen)
            history['environment'].append(current_env)
            history['mean_biofilm'].append(np.mean(self.phenotypes[:, 0]))
            history['mean_motility'].append(np.mean(self.phenotypes[:, 1]))
            history['mean_fitness'].append(np.mean(self.phenotypes[:, 2]))
            history['best_fitness'].append(np.max(self.phenotypes[:, 2]))
            
            # Calculate correlation between biofilm and motility
            corr = np.corrcoef(self.phenotypes[:, 0], self.phenotypes[:, 1])[0, 1]
            history['biofilm_motility_correlation'].append(corr)
            
            # Store gene expression snapshot if this is a designated generation
            if gen in snapshot_generations:
                gene_expression_snapshots[gen] = self.population.copy()
            
            # Selection
            selected = self._selection()
            
            # Create new population from selected individuals
            self.population = self.population[selected]
            
            # Mutation
            self._mutation(mutation_rate)
        
        # Save the evolution history
        self.evolution_history = pd.DataFrame(history)
        
        # Add gene expression snapshots to the results
        self.gene_expression_snapshots = gene_expression_snapshots
        
        return self.evolution_history
    
    def analyze_evolution(self):
        """
        Analyze the evolution results.
        
        Returns:
        --------
        dict
            Dictionary with analysis results.
        """
        if self.evolution_history is None:
            raise ValueError("No evolution has been run yet. Call evolve() first.")
        
        results = {}
        
        # Calculate overall adaptation
        initial_fitness = self.evolution_history['mean_fitness'].iloc[0]
        final_fitness = self.evolution_history['mean_fitness'].iloc[-1]
        
        results['fitness_improvement'] = final_fitness - initial_fitness
        results['relative_improvement'] = final_fitness / initial_fitness if initial_fitness > 0 else float('inf')
        
        # Calculate biofilm-motility trade-off
        biofilm_motility_correlation = self.evolution_history['biofilm_motility_correlation'].mean()
        results['biofilm_motility_correlation'] = biofilm_motility_correlation
        
        # Analyze adaptation to different environments
        env_types = self.evolution_history['environment'].unique()
        env_adaptation = {}
        
        for env in env_types:
            env_data = self.evolution_history[self.evolution_history['environment'] == env]
            
            # Skip if not enough data
            if len(env_data) < 2:
                continue
                
            # Calculate adaptation rate in this environment
            initial_fitness = env_data['mean_fitness'].iloc[0]
            final_fitness = env_data['mean_fitness'].iloc[-1]
            
            env_adaptation[env] = {
                'initial_fitness': float(initial_fitness),
                'final_fitness': float(final_fitness),
                'improvement': float(final_fitness - initial_fitness),
                'phenotype_shift': {
                    'biofilm': float(env_data['mean_biofilm'].iloc[-1] - env_data['mean_biofilm'].iloc[0]),
                    'motility': float(env_data['mean_motility'].iloc[-1] - env_data['mean_motility'].iloc[0])
                }
            }
        
        results['environment_adaptation'] = env_adaptation
        
        # Identify key genes that changed during evolution
        if hasattr(self, 'gene_expression_snapshots'):
            # Compare initial and final gene expression
            initial_expr = self.gene_expression_snapshots[0].mean(axis=0)
            final_expr = self.gene_expression_snapshots[self.n_generations - 1].mean(axis=0)
            
            # Calculate change in expression
            expression_change = final_expr - initial_expr
            
            # Identify top changed genes
            top_increased = []
            top_decreased = []
            
            # Sort genes by absolute change
            sorted_indices = np.argsort(np.abs(expression_change))[::-1]
            
            for idx in sorted_indices[:20]:
                gene_name = self.gene_names[idx]
                change = expression_change[idx]
                
                gene_info = {
                    'gene': gene_name,
                    'initial_expression': float(initial_expr[idx]),
                    'final_expression': float(final_expr[idx]),
                    'change': float(change),
                    'is_regulatory': gene_name in self.regulatory_genes,
                    'is_biofilm': gene_name in self.biofilm_genes,
                    'is_motility': gene_name in self.motility_genes
                }
                
                if change > 0:
                    top_increased.append(gene_info)
                else:
                    top_decreased.append(gene_info)
            
            results['top_increased_genes'] = top_increased[:10]
            results['top_decreased_genes'] = top_decreased[:10]
        
        return results
    
    def visualize_evolution(self, save_path=None):
        """
        Visualize the evolution results.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the visualization.
        
        Returns:
        --------
        matplotlib.figure.Figure
            The visualization figure.
        """
        if self.evolution_history is None:
            raise ValueError("No evolution has been run yet. Call evolve() first.")
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot fitness over time
        ax = axes[0]
        ax.plot(self.evolution_history['generation'], self.evolution_history['mean_fitness'], 
                label='Mean Fitness', linewidth=2)
        ax.plot(self.evolution_history['generation'], self.evolution_history['best_fitness'], 
                label='Best Fitness', linewidth=2, linestyle='--')
        
        # Highlight different environments
        env_changes = []
        current_env = self.evolution_history['environment'].iloc[0]
        
        for i, env in enumerate(self.evolution_history['environment']):
            if env != current_env:
                env_changes.append(i)
                current_env = env
        
        for change in env_changes:
            ax.axvline(x=self.evolution_history['generation'].iloc[change], 
                      color='gray', linestyle=':', alpha=0.7)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('Fitness Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot biofilm and motility over time
        ax = axes[1]
        ax.plot(self.evolution_history['generation'], self.evolution_history['mean_biofilm'], 
                label='Biofilm Formation', linewidth=2, color='blue')
        ax.plot(self.evolution_history['generation'], self.evolution_history['mean_motility'], 
                label='Motility', linewidth=2, color='red')
        
        # Highlight different environments
        for change in env_changes:
            ax.axvline(x=self.evolution_history['generation'].iloc[change], 
                      color='gray', linestyle=':', alpha=0.7)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Phenotype Level')
        ax.set_title('Phenotype Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot biofilm vs motility (trade-off)
        ax = axes[2]
        scatter = ax.scatter(
            self.evolution_history['mean_biofilm'], 
            self.evolution_history['mean_motility'],
            c=self.evolution_history['generation'], 
            cmap='viridis', 
            s=50, 
            alpha=0.7
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Generation')
        
        # Connect points in sequence
        ax.plot(
            self.evolution_history['mean_biofilm'], 
            self.evolution_history['mean_motility'],
            'k-', alpha=0.3
        )
        
        # Add start and end points
        ax.plot(
            self.evolution_history['mean_biofilm'].iloc[0],
            self.evolution_history['mean_motility'].iloc[0],
            'ko', markersize=10, label='Start'
        )
        ax.plot(
            self.evolution_history['mean_biofilm'].iloc[-1],
            self.evolution_history['mean_motility'].iloc[-1],
            'r*', markersize=15, label='End'
        )
        
        ax.set_xlabel('Biofilm Formation')
        ax.set_ylabel('Motility')
        ax.set_title('Biofilm-Motility Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

class BiofilmMotilePredictionModel:
    """
    Machine learning model to predict bacterial phenotypes from gene expression.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the model.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('random_forest' or 'linear').
        """
        self.model_type = model_type
        
        if model_type == 'random_forest':
            self.biofilm_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.motility_model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'linear':
            self.biofilm_model = LinearRegression()
            self.motility_model = LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.feature_importance = None
    
    def train(self, gene_expression, phenotypes):
        """
        Train the prediction model.
        
        Parameters:
        -----------
        gene_expression : numpy.ndarray
            Gene expression data (samples x genes).
        phenotypes : numpy.ndarray
            Phenotype data (samples x phenotypes) where phenotypes are [biofilm, motility, fitness].
        
        Returns:
        --------
        self
            Trained model.
        """
        # Extract biofilm and motility phenotypes
        biofilm = phenotypes[:, 0]
        motility = phenotypes[:, 1]
        
        # Train models
        self.biofilm_model.fit(gene_expression, biofilm)
        self.motility_model.fit(gene_expression, motility)
        
        # Extract feature importance if available
        if hasattr(self.biofilm_model, 'feature_importances_'):
            self.feature_importance = {
                'biofilm': self.biofilm_model.feature_importances_,
                'motility': self.motility_model.feature_importances_
            }
        
        return self
    
    def predict(self, gene_expression):
        """
        Predict phenotypes from gene expression.
        
        Parameters:
        -----------
        gene_expression : numpy.ndarray
            Gene expression data (samples x genes).
        
        Returns:
        --------
        numpy.ndarray
            Predicted phenotypes (samples x 2) where columns are [biofilm, motility].
        """
        # Predict biofilm and motility
        biofilm_pred = self.biofilm_model.predict(gene_expression)
        motility_pred = self.motility_model.predict(gene_expression)
        
        # Combine predictions
        predictions = np.column_stack((biofilm_pred, motility_pred))
        
        return predictions
    
    def get_important_genes(self, gene_names, top_n=20):
        """
        Get the most important genes for predicting phenotypes.
        
        Parameters:
        -----------
        gene_names : list
            List of gene names.
        top_n : int
            Number of top genes to return.
        
        Returns:
        --------
        dict
            Dictionary with important genes for each phenotype.
        """
        if self.feature_importance is None:
            return None
        
        important_genes = {}
        
        for phenotype, importances in self.feature_importance.items():
            # Get indices of top genes
            top_indices = np.argsort(importances)[::-1][:top_n]
            
            # Get gene names and importances
            top_genes = [
                {
                    'gene': gene_names[i],
                    'importance': float(importances[i])
                }
                for i in top_indices
            ]
            
            important_genes[phenotype] = top_genes
        
        return important_genes
    
    def plot_important_genes(self, gene_names, top_n=20):
        """
        Plot the most important genes for predicting phenotypes.
        
        Parameters:
        -----------
        gene_names : list
            List of gene names.
        top_n : int
            Number of top genes to plot.
        
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure.
        """
        if self.feature_importance is None:
            return None
        
        # Create figure with two subplots (one for each phenotype)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        for i, (phenotype, importances) in enumerate(self.feature_importance.items()):
            # Get indices of top genes
            top_indices = np.argsort(importances)[::-1][:top_n]
            
            # Get gene names and importances
            top_gene_names = [gene_names[j] for j in top_indices]
            top_importances = [importances[j] for j in top_indices]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(top_gene_names))
            axes[i].barh(y_pos, top_importances, align='center')
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(top_gene_names)
            axes[i].invert_yaxis()  # Labels read top-to-bottom
            axes[i].set_xlabel('Feature Importance')
            axes[i].set_title(f'Top Genes for {phenotype.capitalize()} Prediction')
        
        plt.tight_layout()
        return fig
