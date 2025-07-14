# Setup Guide for Bacterial Multi-Omics Integration Platform

This guide provides instructions on how to set up and run the Bacterial Multi-Omics Integration Platform.

## Environment Setup

### Required Python Packages

The platform requires the following Python packages:

```
streamlit>=1.18.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.5.0
plotly>=5.8.0
scikit-learn>=1.0.0
seaborn>=0.12.0
networkx>=2.8.0
cobra>=0.25.0  # For metabolic modeling
```

### Optional Packages

For enhanced functionality, you can also install:

```
shap>=0.41.0   # For model interpretability (SHAP analysis)
```

## Installation Using Replit

1. Clone this Repl or fork it to your account.
2. Replit will automatically install the necessary dependencies.
3. Run the application by clicking the Run button.

## Installation on Local Machine

If you want to run the application locally:

1. Ensure you have Python 3.8+ installed.
2. Install the required packages using pip:
   ```bash
   pip install streamlit pandas numpy matplotlib plotly scikit-learn seaborn networkx
   ```
3. For metabolic modeling functionality, install COBRApy:
   ```bash
   pip install cobra
   ```
4. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bacterial-omics-platform.git
   cd bacterial-omics-platform
   ```
5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Verifying the Installation

After starting the application, you should see:

1. The main dashboard with a 3D visualization of example data
2. A sidebar with links to the five main modules
3. The ability to navigate between different sections

## Troubleshooting Common Issues

### SHAP Package Issues

If you encounter issues with the SHAP package, the application will still work, but with limited model interpretability features. The platform is designed to gracefully handle the absence of SHAP.

### COBRApy Package Issues

If the COBRApy package is not available, the metabolic modeling features will be limited. The platform will display informative messages about the missing functionality.

### Data Import Problems

If you encounter issues importing your data:
- Ensure your CSV files follow the expected format
- Check that column and row headers are correctly formatted
- Verify that there are no missing values in critical fields

## Data Format Requirements

For proper functioning, your input files should follow these formats:

### Transcriptomics Data
- CSV file with genes as rows
- First column should contain gene identifiers
- Other columns represent samples/conditions
- Cell values should be normalized expression values

### Tn-Seq Data
- CSV file with genes as rows
- First column should contain gene identifiers
- Other columns represent conditions
- Cell values should be fitness/essentiality scores

### Phenotype Data
- CSV file with samples as rows
- First column should contain sample identifiers
- Other columns represent phenotype measurements
- Cell values should be quantitative measurements

## Setting Up Streamlit Configuration

For optimal performance, ensure your `.streamlit/config.toml` file contains:

```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```

This configuration ensures the application is accessible and performs properly.

## Running in a Virtual Environment (Recommended)

If you are using a Homebrew-managed Python 3.13 on macOS (or any system where *pip* is "externally-managed" per [PEP 668](https://peps.python.org/pep-0668/)), you **must** create a virtual environment before installing the requirements.

```bash
# 1. From the project root
python3 -m venv .venv           # create venv in place

# 2. Activate it (macOS/Linux-bash/zsh)
source .venv/bin/activate

# PowerShell (Windows)
# .venv\Scripts\Activate.ps1

# 3. Install Python packages
pip install -r requirements.txt       # installs everything listed

#   └─ Note: TensorFlow wheels are not yet available for Python 3.13.
#            If you need the ML pages that rely on TensorFlow, use
#            Python 3.11 or 3.12, or comment-out the TF import lines.

# 4. Launch Streamlit from inside the venv
streamlit run app.py
```

### Fixing missing-module errors at runtime

If, after launching Streamlit, you see errors such as:

```
ModuleNotFoundError: No module named 'plotly'
```

it means that the dependency wasn’t installed inside your current environment.  Make sure you’re *inside* the venv (prompt should be prefixed by `(.venv)`), then install the module manually:

```bash
pip install plotly scikit-learn matplotlib
```

### Apple-silicon notes (M-series Macs)
TensorFlow official wheels are still experimental for arm64/py>3.11.  Until an official wheel is released, the **Machine Learning** page that calls TensorFlow will raise `ModuleNotFoundError`.  Two work-arounds:

1. Create your virtual environment with Python 3.11 and run `pip install tensorflow-macos`.
2. Disable the TensorFlow-specific features by commenting out lines that import or call `tensorflow` in `utils/ml_models.py` and the relevant pages.

---
These steps match precisely what we executed during the live debug session, so you can reproduce a working setup from scratch.