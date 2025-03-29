# Contributing to the Bacterial Multi-Omics Integration Platform

We welcome contributions to the Bacterial Multi-Omics Integration Platform! This document provides guidelines for contributing to the project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## How to Contribute

### Reporting Bugs

If you find a bug in the platform, please submit a detailed bug report including:

1. A clear description of the bug
2. Steps to reproduce the issue
3. Expected behavior vs. actual behavior
4. Screenshots if applicable
5. System information (browser, OS, etc.)

### Suggesting Enhancements

We welcome suggestions for enhancing the platform. When proposing new features:

1. Clearly describe the enhancement and its use case
2. Explain how it would benefit the platform's users
3. If possible, outline a potential implementation approach

### Code Contributions

If you'd like to contribute code to the project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Write or update tests as needed
5. Update documentation to reflect your changes
6. Commit your changes (`git commit -m 'Add some amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Pull Request Process

1. Ensure your code follows the project's coding style
2. Update documentation as necessary
3. Include tests that validate your changes
4. The PR should work on the main development branch
5. Detail the changes your PR introduces

## Development Setup

To set up the development environment:

1. Clone the repository
2. Install the required dependencies as described in SETUP_GUIDE.md
3. Start the development server with hot reloading: `streamlit run app.py`

## Project Structure

Understanding the project structure will help you contribute effectively:

```
├── app.py                  # Main application entry point
├── pages/                  # Individual application pages
│   ├── 1_Data_Import.py    # Data import functionality
│   ├── 2_Data_Exploration.py # Data visualization and exploration
│   ├── 3_Machine_Learning.py # Phenotype prediction models
│   ├── 4_Metabolic_Modeling.py # Metabolic analysis
│   └── 5_Simulation.py     # In silico evolution simulations
├── utils/                  # Utility functions
│   ├── data_processing.py  # Data preprocessing functions
│   ├── ml_models.py        # Machine learning model definitions
│   ├── metabolic_models.py # Metabolic modeling functions
│   ├── simulation.py       # Simulation related functionality
│   └── visualization.py    # Plotting and visualization tools
└── .streamlit/config.toml  # Streamlit configuration
```

## Coding Standards

### Python Style Guide

- Follow PEP 8 guidelines for Python code
- Use docstrings to document functions and classes
- Keep functions focused on a single responsibility
- Use type hints where appropriate

### Streamlit Best Practices

- Make the interface intuitive and user-friendly
- Provide clear instructions for users
- Include appropriate error handling and user feedback
- Ensure the application is responsive
- Optimize computation for larger datasets

## Testing

For new features or bug fixes, please include appropriate tests:

- Unit tests for utility functions
- Integration tests for page functionality
- End-to-end tests for critical workflows

## Documentation

Good documentation is crucial for the project:

- Update docstrings for any new or modified functions
- Keep the README.md updated with new features
- Update user guides as needed
- Add comments for complex algorithms or non-obvious code

## Feature Priorities

Current feature priorities include:

1. Enhanced visualization options for omics data
2. Integration with additional data types
3. More advanced statistical analysis tools
4. Improved export functionality
5. Enhanced model interpretability

## Getting Help

If you need help with contributing:

- Check existing documentation
- Review related issues and pull requests
- Reach out to the maintainers

Thank you for contributing to the Bacterial Multi-Omics Integration Platform!