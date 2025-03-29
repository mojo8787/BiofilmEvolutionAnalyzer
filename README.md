# Bacterial Multi-Omics Integration Platform

A comprehensive computational platform for integrating multi-omics data to predict bacterial phenotypes and identify regulatory mechanisms governing the biofilm-motility transition.

## Author
Developed by **Almotasem Bellah Younis, PhD**

LinkedIn | Twitter | ResearchGate  
Email: motasem.youniss@gmail.com  
Phone: +427 325 80600  
Location: Veletržní 817/9, 603 00 Brno-střed, Czech Republic

![Bacterial Phenotype Space](https://raw.githubusercontent.com/placeholderusername/bacterial-omics-platform/main/static/phenotype_space.png)

## Overview

This platform integrates transcriptomics, Tn-Seq, and phenotypic data to provide insights into the molecular mechanisms that determine bacterial phenotypes, with a particular focus on the trade-off between biofilm formation and motility. The application leverages machine learning techniques and metabolic modeling to predict phenotypes and identify key regulatory elements.

## Features

- **Multi-Omics Data Integration**: Import and process transcriptomics, Tn-Seq, and phenotype data
- **Interactive Data Exploration**: Visualize and analyze relationships between datasets
- **Machine Learning Models**: Train and evaluate models to predict phenotypes from omics data
- **Neural Network Support**: Implement deep learning models for complex phenotype prediction
- **Regulatory Mechanism Discovery**: Identify key regulators using feature importance analysis
- **Metabolic Modeling**: Explore resource allocation trade-offs using genome-scale metabolic models
- **In Silico Evolution**: Simulate bacterial adaptation under different environmental conditions
- **Automated Experiment Design**: Generate optimal experimental designs based on data gaps and research objectives

## Application Structure

The application is built using Streamlit and organized into the following components:

```
├── app.py                  # Main application entry point
├── pages/                  # Individual application pages
│   ├── 1_Data_Import.py    # Data import functionality
│   ├── 2_Data_Exploration.py # Data visualization and exploration
│   ├── 3_Machine_Learning.py # Phenotype prediction models
│   ├── 4_Metabolic_Modeling.py # Metabolic analysis
│   ├── 5_Simulation.py     # In silico evolution simulations
│   └── 6_Experiment_Design.py # Automated experiment design
├── utils/                  # Utility functions
│   ├── data_processing.py  # Data preprocessing functions
│   ├── ml_models.py        # Machine learning model definitions
│   ├── metabolic_models.py # Metabolic modeling functions
│   ├── simulation.py       # Simulation related functionality
│   ├── experiment_design.py # Experiment design algorithms
│   └── visualization.py    # Plotting and visualization tools
└── .streamlit/config.toml  # Streamlit configuration
```

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python packages:
  - streamlit
  - pandas
  - numpy
  - matplotlib
  - plotly
  - scikit-learn
  - tensorflow
  - networkx
  - seaborn
  - cobra (for metabolic modeling)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bacterial-omics-platform.git
cd bacterial-omics-platform
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage Instructions

### 1. Data Import

Start by importing your omics datasets:
- Transcriptomics data: Gene expression measurements across different conditions
- Tn-Seq data: Gene fitness/essentiality scores
- Phenotype data: Measurements of biofilm formation, motility, etc.

The platform accepts CSV files with specific formats:
- Transcriptomics: Rows represent genes, columns represent samples
- Tn-Seq: Rows represent genes, columns represent conditions
- Phenotypes: Rows represent samples, columns represent phenotype measurements

### 2. Data Exploration

After importing data, explore relationships between datasets:
- View data summaries and distributions
- Generate correlation matrices and heatmaps
- Create PCA plots to identify patterns

### 3. Machine Learning

Build predictive models to identify key regulatory genes:
- Select target phenotype (biofilm formation, motility, etc.)
- Choose feature selection method (variance, correlation)
- Train models (Random Forest, Gradient Boosting, etc.)
- Evaluate performance and analyze feature importance

### 4. Metabolic Modeling

Integrate omics data with genome-scale metabolic models:
- Load standard metabolic models or custom SBML models
- Apply constraints based on transcriptomics and Tn-Seq data
- Analyze flux distributions and resource allocation
- Explore trade-offs between biofilm and motility-related pathways

### 5. Simulation

Simulate bacterial evolution under different conditions:
- Set up simulations with varying environments
- Track phenotypic changes across generations
- Identify adaptive mutations and compensatory mechanisms

### 6. Experiment Design

Generate optimized experimental designs to efficiently explore the phenotype space:
- Create factorial and response surface designs for systematic exploration
- Analyze knowledge gaps in existing experimental data
- Generate optimal experiment sequences using Bayesian optimization
- Produce experimental protocol templates for laboratory implementation

## Workflow Examples

### Example 1: Identify Key Regulators of Biofilm Formation

1. Import transcriptomics and phenotype data
2. In Data Exploration, identify samples with extreme biofilm phenotypes
3. Using Machine Learning, train a classifier to predict biofilm formation
4. Analyze feature importance to identify regulatory genes
5. Validate findings using Metabolic Modeling

### Example 2: Study Resource Allocation Trade-offs

1. Import all omics datasets
2. Create context-specific metabolic models using transcriptomics data
3. Compare flux distributions between biofilm-forming and motile strains
4. Identify key metabolic pathways involved in the phenotypic switch
5. Simulate evolutionary trajectories under selective pressure

### Example 3: Design Experiments to Test Regulatory Hypotheses

1. Import existing experimental data and train initial models
2. Use Experiment Design to identify knowledge gaps in the current dataset
3. Generate an optimal sequence of experiments using Bayesian optimization
4. Create a factorial design to test interactions between key regulators
5. Export experimental protocols for laboratory implementation

## Future Enhancements

- Enhanced SHAP analysis for deeper model interpretability
- Integration with BioCyc and KEGG for pathway enrichment analysis
- Support for additional omics data types (proteomics, metabolomics)
- Collaborative features for sharing models and results
- GPU acceleration for large-scale simulations

## Scientific Background

The platform is designed to explore the molecular basis of bacterial phenotype transitions, particularly focusing on the inverse relationship between biofilm formation and motility. This phenotypic switch is central to bacterial adaptation, virulence, and response to environmental stresses.

Key biological concepts addressed:
- C-di-GMP signaling in bacterial lifestyle decisions
- Metabolic resource allocation theory
- Evolutionary trade-offs in bacterial adaptation
- Regulatory network architecture in phenotype determination

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This platform builds on the work of numerous researchers in the fields of systems biology, microbiology, and machine learning.
- Special thanks to the developers of COBRA, scikit-learn, and other open-source tools that make this research possible.