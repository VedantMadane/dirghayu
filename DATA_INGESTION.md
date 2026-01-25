# Data Ingestion & Training on Large Scale Genome Repositories

Dirghayu is designed to scale from single-sample analysis to population-level training on terabytes of genomic data (e.g., GenomeIndia, 1000 Genomes, UK Biobank).

To train the AI models (`LifespanNet-India`, `DiseaseNet-Multi`) on 100GB+ datasets, we cannot load raw VCF files into RAM. Instead, we use a **Streaming + Columnar** approach.

## 🚀 Strategy: VCF → Parquet → PyTorch Stream

1.  **Ingest**: Convert raw VCFs (row-based, slow text parsing) into **Parquet** files (columnar, compressed, fast binary reads).
2.  **Stream**: Use a custom PyTorch `IterableDataset` to stream batches of data from disk during training.
3.  **Train**: Update models incrementally without memory limits.

---

## 🛠 Step 1: Convert VCF Repos to Parquet

Use the provided conversion script (to be created) to process your 100GB+ VCF repository.

```bash
# Example: Convert a directory of VCFs to partitioned Parquet dataset
python scripts/vcf_to_parquet.py \
    --input_dir /path/to/genome_repo/vcfs/ \
    --output_dir /path/to/processed_data/ \
    --threads 16
```

**Why Parquet?**
-   **Size Reduction**: 100GB VCF -> ~20-30GB Parquet (Snappy compression).
-   **Speed**: Reading a batch of genotypes is 100x faster than parsing VCF text.
-   **Queryable**: You can use SQL (via DuckDB) to inspect the data.

---

## 🔗 Step 2: Connect to Data Source

### Option A: Local / High-Performance NAS
Just point the training script to your processed directory.
```bash
python scripts/train_models.py --data_dir /mnt/genomics_data/processed/
```

### Option B: Cloud Buckets (AWS S3 / GCS)
If your repo is on the cloud, mount it using `s3fs` or `gcsfuse` so it appears as a local filesystem to PyTorch.

**AWS S3 Example:**
```bash
# Mount bucket
mkdir -p /mnt/s3_data
s3fs my-genomics-bucket /mnt/s3_data

# Train
python scripts/train_models.py --data_dir /mnt/s3_data/parquet/
```

---

## 🧬 Step 3: Training with the `GenomicBigDataset`

The `GenomicBigDataset` class (in `src/data/dataset.py`) handles the complexity:
1.  It finds all `.parquet` files in your data directory.
2.  It uses `pyarrow` to read chunks of data efficiently.
3.  It handles "shuffling" via an in-memory buffer to ensure statistical randomness.

```python
# Code snippet (how it works internally)
dataset = GenomicBigDataset(
    data_dir="/path/to/data",
    features=["rs123", "rs456", ...], # List of variants to use as features
    target_col="lifespan"
)
dataloader = DataLoader(dataset, batch_size=1024)
```

## 📝 Requirements for Repository Data

Your repository data should eventually be structured as a table (DataFrame) with:
-   **Genotype Columns**: e.g., `rs1801133` (values: 0, 1, 2)
-   **Phenotype Columns**: e.g., `age`, `has_t2d`, `bmi`

*Note: The `vcf_to_parquet.py` script helps flatten VCFs into this format, merging with a clinical metadata CSV if provided.*
