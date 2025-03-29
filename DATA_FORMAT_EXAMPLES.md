# Example Data Formats

This guide provides sample formats for the data files expected by the Bacterial Multi-Omics Integration Platform. Understanding these formats will help you prepare your own data for analysis.

## Transcriptomics Data

Transcriptomics data should be in CSV format with genes as rows and samples as columns. The first column contains gene identifiers.

### Example:

```csv
gene_id,sample1,sample2,sample3,sample4
gene001,10.5,12.3,9.8,11.2
gene002,5.6,6.2,5.9,6.0
gene003,8.9,3.2,7.4,2.1
gene004,12.3,11.8,12.0,11.5
gene005,7.5,8.1,7.2,8.5
```

### Key Format Requirements:
- First column: Gene identifiers
- Other columns: Sample names
- Values: Normalized expression values (e.g., log2-transformed counts)
- No missing values in the data matrix

## Tn-Seq Data

Tn-Seq data should be in CSV format with genes as rows and experimental conditions as columns. The first column contains gene identifiers.

### Example:

```csv
gene_id,condition1,condition2,condition3
gene001,-0.5,0.8,-1.2
gene002,-4.3,-3.9,-4.5
gene003,0.1,0.3,0.2
gene004,-2.5,-2.3,-2.8
gene005,1.2,1.5,1.1
```

### Key Format Requirements:
- First column: Gene identifiers
- Other columns: Experimental conditions
- Values: Fitness scores or log2 fold-change values
  - Negative values typically indicate gene essentiality
  - Values close to zero typically indicate neutral fitness
  - Positive values may indicate beneficial mutations
- Missing values should be handled before import or marked as NaN

## Phenotype Data

Phenotype data should be in CSV format with samples as rows and different phenotype measurements as columns. The first column contains sample identifiers.

### Example:

```csv
sample_id,biofilm_formation,motility,antibiotic_resistance
sample1,0.82,0.25,0.65
sample2,0.35,0.78,0.42
sample3,0.91,0.15,0.88
sample4,0.28,0.92,0.31
```

### Key Format Requirements:
- First column: Sample identifiers (must match those in transcriptomics data)
- Other columns: Phenotype measurements
- Values: Normalized measurements (ideally between 0 and 1)
- Each phenotype should be in a separate column
- Missing values should be handled before import or marked as NaN

## Important Considerations

### Sample ID Consistency

Sample IDs must be consistent across datasets to allow proper integration. For example, the sample names in the transcriptomics data should match the sample IDs in the phenotype data.

### Data Preprocessing

For best results, your data should be:
- Normalized appropriately for its data type
- Filtered to remove low-quality measurements
- Free of batch effects or technical artifacts

### File Size Limitations

While the platform can handle reasonably large datasets, performance may degrade with extremely large files:
- Transcriptomics: Up to ~20,000 genes x 100 samples
- Tn-Seq: Up to ~5,000 genes x 20 conditions
- Phenotype: Up to ~500 samples x 20 phenotypes

## Example Data Loading

The platform provides example data that you can use to explore functionality before uploading your own data. To access this:

1. Navigate to the "Data Import" page
2. Look for the "Load Example Data" button
3. Select the type of example data you want to load
4. Proceed with the analysis using this example data

## Data Privacy

All data processing occurs locally in your browser and server. No data is sent to external services.