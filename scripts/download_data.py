#!/usr/bin/env python3
"""
Download public genomics datasets for Dirghayu

Data sources:
1. GenomeIndia (10k Indian genomes) - https://clingen.igib.res.in/genomeIndia/
2. gnomAD (population frequencies)
3. AlphaMissense (pathogenicity predictions)
4. 1000 Genomes Project
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import gzip
import shutil

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def download_file(url: str, dest: Path, desc: str = "Downloading"):
    """Download file with progress bar"""
    if dest.exists():
        print(f"[OK] {dest.name} already exists, skipping")
        return

    print(f"Downloading {desc}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with (
        open(dest, "wb") as f,
        tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    print(f"[OK] Downloaded {dest.name}")


def download_genome_india():
    """
    GenomeIndia Project: 10,000 Indian genomes
    https://clingen.igib.res.in/genomeIndia/

    Note: This downloads summary statistics and variant frequencies.
    Full VCF access requires registration.
    """
    print("\n=== GenomeIndia Data ===")
    genome_india_dir = DATA_DIR / "genome_india"
    genome_india_dir.mkdir(exist_ok=True)

    # GenomeIndia variant frequency database (public subset)
    # TODO: Update with actual public data URLs when available
    print("[!] GenomeIndia full data requires registration at:")
    print("  https://clingen.igib.res.in/genomeIndia/")
    print("  Download VCF files manually and place in:", genome_india_dir)

    # For now, we'll use 1000 Genomes Indian samples as proxy
    print("\n[*] Downloading 1000 Genomes Indian samples as proxy...")


def download_gnomad():
    """
    gnomAD: Population allele frequencies
    https://gnomad.broadinstitute.org/
    """
    print("\n=== gnomAD Data ===")
    gnomad_dir = DATA_DIR / "gnomad"
    gnomad_dir.mkdir(exist_ok=True)

    # Download small example VCF for testing
    # Full gnomAD is ~1TB, use API or BigQuery for production
    test_vcf_url = "https://gnomad-public-us-east-1.s3.amazonaws.com/release/4.0/vcf/genomes/gnomad.genomes.v4.0.sites.chr22.vcf.bgz"

    dest = gnomad_dir / "gnomad_chr22_example.vcf.bgz"

    print("[*] Downloading gnomAD chr22 example (for testing)...")
    print("[!] Full gnomAD is 1TB+. For production, use:")
    print("  - gnomAD API: https://gnomad.broadinstitute.org/api")
    print("  - BigQuery: bigquery-public-data.gnomad_r4_0.*")

    # Uncomment to actually download (600MB)
    # download_file(test_vcf_url, dest, "gnomAD chr22")


def download_alphamissense():
    """
    AlphaMissense: AI-predicted pathogenicity for all possible missense variants
    https://github.com/google-deepmind/alphamissense
    """
    print("\n=== AlphaMissense Data ===")
    alphamissense_dir = DATA_DIR / "alphamissense"
    alphamissense_dir.mkdir(exist_ok=True)

    # AlphaMissense predictions (all possible missense variants)
    url = "https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz"
    dest = alphamissense_dir / "AlphaMissense_hg38.tsv.gz"

    print("[*] Downloading AlphaMissense predictions...")
    print("[!] This is 900MB compressed, 5GB uncompressed")

    # Uncomment to download
    # download_file(url, dest, "AlphaMissense predictions")


def download_1000genomes_sample():
    """
    Download small 1000 Genomes sample for testing
    Focus on Indian populations: GIH, ITU, STU, BEB, PJL
    """
    print("\n=== 1000 Genomes Project (Indian subset) ===")
    kg_dir = DATA_DIR / "1000genomes"
    kg_dir.mkdir(exist_ok=True)

    # Sample metadata
    metadata_url = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/1000genomes.sequence.index"

    print("[*] Downloading 1000 Genomes metadata...")
    print("\nIndian populations:")
    print("  - GIH: Gujarati Indian from Houston, Texas")
    print("  - ITU: Indian Telugu from the UK")
    print("  - STU: Sri Lankan Tamil from the UK")
    print("  - BEB: Bengali from Bangladesh")
    print("  - PJL: Punjabi from Lahore, Pakistan")

    # For actual VCF data, use:
    print("\n[!] For full VCF files:")
    print(
        "  ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/"
    )


def create_sample_vcf():
    """
    Create a minimal example VCF for testing pipeline
    """
    print("\n=== Creating Sample VCF ===")

    sample_vcf = DATA_DIR / "sample.vcf"

    vcf_content = """##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	SAMPLE1
1	69511	rs75062661	A	G	100	PASS	AF=0.0002	GT	0/1
1	865628	rs1278270	G	A	100	PASS	AF=0.32	GT	1/1
19	44908684	rs429358	C	T	100	PASS	AF=0.15	GT	0/1
1	11856378	rs1801133	C	T	100	PASS	AF=0.30	GT	1/1
"""

    with open(sample_vcf, "w") as f:
        f.write(vcf_content)

    print(f"[OK] Created sample VCF at: {sample_vcf}")
    print("  Contains variants:")
    print("    - rs429358 (APOE e4 - Alzheimer's risk)")
    print("    - rs1801133 (MTHFR C677T - Folate metabolism)")


def main():
    print("=" * 60)
    print("Dirghayu Data Download Script")
    print("=" * 60)

    # Create sample VCF for testing
    create_sample_vcf()

    # Show info for larger downloads
    download_genome_india()
    download_1000genomes_sample()
    download_gnomad()
    download_alphamissense()

    print("\n" + "=" * 60)
    print("[OK] Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Register at GenomeIndia to access full dataset")
    print("2. Uncomment download functions for large files when ready")
    print("3. Run: python scripts/parse_vcf.py data/sample.vcf")


if __name__ == "__main__":
    main()
