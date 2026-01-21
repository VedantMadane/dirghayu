# Pipelines

Bioinformatics workflows for WGS processing and analysis.

## Structure

```
pipelines/
├── variant_calling/    # FASTQ → VCF pipeline
├── annotation/         # Variant annotation workflows
├── protein_analysis/   # Structure prediction & stability
├── qc/                # Quality control checks
└── ml_inference/      # ML model inference pipelines
```

## Workflows

### 1. Variant Calling
- Input: FASTQ files
- Output: Annotated VCF + QC reports
- Tools: BWA-MEM2, DeepVariant, VEP

### 2. Protein Analysis
- Input: VCF with coding variants
- Output: Protein stability predictions
- Tools: AlphaFold2, FoldX, AlphaMissense

### 3. Risk Scoring
- Input: Annotated variants
- Output: PRS and disease risk scores
- Tools: PRSice-2, custom ML models
