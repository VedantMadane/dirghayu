# Dirghayu Quick Start Guide

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create sample data
python scripts/download_data.py
```

## Run Demo (Local)

```bash
# Run complete pipeline demo
python demo.py data/sample.vcf
```

**Demo includes:**
- ✅ VCF parsing
- ✅ Variant annotation (Ensembl + gnomAD APIs)
- ✅ ML model training (nutrient deficiency predictor)
- ✅ Personalized health report generation

**Expected output:**
- Parsed variants table
- Annotated variants with gene symbols
- Nutrient deficiency risk scores (0-1)
- Personalized recommendations

## Run API Server

```bash
# Start FastAPI server
python src/api/server.py
```

**Access:**
- API Docs (Swagger): http://localhost:8000/docs
- OpenAPI Spec: http://localhost:8000/openapi.json
- ReDoc: http://localhost:8000/redoc

**Example API calls:**

```bash
# Health check
curl http://localhost:8000/

# Annotate single variant (MTHFR C677T)
curl -X POST http://localhost:8000/api/v1/annotate/variant \
  -H "Content-Type: application/json" \
  -d '{"chrom":"1","pos":11856378,"ref":"C","alt":"T"}'

# Upload VCF for nutrient predictions
curl -X POST http://localhost:8000/api/v1/predict/nutrients \
  -F "vcf_file=@data/sample.vcf"

# Comprehensive analysis
curl -X POST "http://localhost:8000/api/v1/analyze/comprehensive?patient_id=SAMPLE001" \
  -F "vcf_file=@data/sample.vcf"
```

## Project Structure

```
dirghayu/
├── src/
│   ├── data/           # VCF parsing, annotation
│   ├── models/         # ML models (nutrient predictor, etc.)
│   └── api/            # FastAPI server
├── scripts/            # Utility scripts
├── data/               # Downloaded datasets
├── models/             # Trained model checkpoints
├── demo.py             # End-to-end demo
└── requirements.txt    # Python dependencies
```

## Use Your Own Data

```bash
# Run with your VCF file
python demo.py /path/to/your/genome.vcf

# Or via API
curl -X POST http://localhost:8000/api/v1/analyze/comprehensive \
  -F "vcf_file=@/path/to/your/genome.vcf" \
  -F "patient_id=YOUR_ID"
```

## Cloud Deployment (GCP with T4 GPU)

**Note: Requires GCP account and cloud configuration**

```python
# src/cloud/deploy_gcp.py (NOT RUN LOCALLY)
from google.cloud import compute_v1
from google.cloud import storage

# This code won't run locally - requires GCP authentication
# Included for reference when deploying to cloud

def deploy_to_gcp():
    """Deploy to GCP with T4 GPU"""
    
    # Instance configuration
    instance_config = {
        "name": "dirghayu-ml-server",
        "machine_type": "n1-standard-4",
        "zone": "asia-southeast1-c",  # Your preferred zone
        "gpus": [{
            "type": "nvidia-tesla-t4",
            "count": 1
        }],
        "disk": {
            "size_gb": 100,
            "image": "projects/ml-images/global/images/c0-deeplearning-common-gpu-v20230925-debian-11-py310"
        }
    }
    
    # Deploy model to Cloud Storage
    storage_client = storage.Client()
    bucket = storage_client.bucket("dirghayu-models")
    
    # Upload trained models
    blob = bucket.blob("nutrient_predictor.pth")
    blob.upload_from_filename("models/nutrient_predictor.pth")
    
    print("✓ Deployed to GCP")
    print(f"  Instance: {instance_config['name']}")
    print(f"  Zone: {instance_config['zone']}")
    print(f"  GPU: NVIDIA Tesla T4")

# DON'T RUN THIS WITHOUT GCP CREDENTIALS
# deploy_to_gcp()
```

## Data Sources

### Public Datasets

1. **GenomeIndia** (10k genomes)
   - URL: https://clingen.igib.res.in/genomeIndia/
   - **Requires registration** - not auto-downloaded

2. **1000 Genomes** (South Asian populations)
   - Included: GIH, ITU, STU, BEB, PJL
   - Auto-downloaded as proxy

3. **gnomAD** (population frequencies)
   - API: https://gnomad.broadinstitute.org/api
   - Used for variant annotation

4. **AlphaMissense** (pathogenicity predictions)
   - 900MB download
   - Uncomment in `scripts/download_data.py` to download

### For Production

```bash
# Download full datasets (large files)
python scripts/download_data.py --full
```

## Next Steps

1. **Get Real Genomic Data:**
   - 23andMe, AncestryDNA export
   - Whole Genome Sequencing (WGS) from providers
   - Targeted gene panels

2. **Train on Clinical Data:**
   - Collect nutrient deficiency labels
   - Hospital/clinic partnerships
   - Update models with `models/nutrient_predictor.py`

3. **Add More Models:**
   - Disease risk (CVD, T2D, cancer)
   - Longevity prediction
   - Drug metabolism (pharmacogenomics)

4. **Deploy to Production:**
   - Containerize with Docker
   - Deploy to GCP/AWS
   - Set up authentication
   - HIPAA compliance

## Troubleshooting

**Issue: API calls failing (Ensembl/gnomAD)**
- Solution: Rate limiting active - reduce concurrent requests
- Alternative: Download full databases locally

**Issue: Model not found**
- Solution: Run `python demo.py` first to train model
- Or: Provide pre-trained model checkpoint

**Issue: cyvcf2 not installing**
- Solution: Fallback VCF parser included
- Works without cyvcf2 (slower but functional)

**Issue: Out of memory**
- Solution: Process VCF in batches
- Use Parquet files for large datasets

## Support

Questions? Issues?
- GitHub: [Issues page]
- Docs: `docs/` folder
- Examples: `demo.py` for working code
