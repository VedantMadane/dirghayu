#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Dirghayu Demo (no ML dependencies required)

Demonstrates:
1. VCF parsing
2. Basic variant analysis
"""

import sys
import io
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import parse_vcf_file


def run_quick_demo(vcf_path: Path):
    """Run quick demo without ML dependencies"""
    
    print("=" * 80)
    print("DIRGHAYU: India-First Longevity Genomics Platform - Quick Demo")
    print("=" * 80)
    
    # Step 1: Parse VCF
    print("\n[1/2] Parsing VCF file...")
    print(f"  Input: {vcf_path}")
    
    variants_df = parse_vcf_file(vcf_path)
    print(f"  [OK] Found {len(variants_df)} variants")
    
    if len(variants_df) == 0:
        print("  [!] No variants found!")
        return
    
    print("\n  All variants:")
    print(variants_df[['chrom', 'pos', 'rsid', 'ref', 'alt', 'genotype']].to_string())
    
    # Step 2: Analyze key variants
    print("\n[2/2] Analyzing clinical significance...")
    
    # Look for key variants
    key_variants = {
        'rs1801133': {
            'gene': 'MTHFR',
            'name': 'C677T',
            'impact': 'Folate metabolism - Higher homocysteine levels',
            'recommendation': 'Consider methylfolate supplementation (800 mcg/day)'
        },
        'rs429358': {
            'gene': 'APOE',
            'name': 'ε4 allele',
            'impact': 'Increased Alzheimer\'s disease risk (3-4x)',
            'recommendation': 'Focus on cardiovascular health, Mediterranean diet'
        },
        'rs1801131': {
            'gene': 'MTHFR',
            'name': 'A1298C',
            'impact': 'Folate metabolism - Combined with C677T increases risk',
            'recommendation': 'Monitor homocysteine levels, B-vitamin supplementation'
        },
        'rs1333049': {
            'gene': 'CDKN2B-AS1',
            'name': '9p21.3 locus',
            'impact': 'Coronary artery disease risk marker',
            'recommendation': 'Regular cardiovascular screening, healthy lifestyle'
        },
        'rs713598': {
            'gene': 'TAS2R38',
            'name': 'PTC taster',
            'impact': 'Bitter taste perception - affects vegetable preferences',
            'recommendation': 'May affect dietary choices; ensure varied nutrition'
        }
    }
    
    print("\n" + "=" * 80)
    print("🧬 GENETIC INSIGHTS")
    print("=" * 80)
    
    found_any = False
    for _, var in variants_df.iterrows():
        rsid = var['rsid']
        if rsid in key_variants:
            found_any = True
            info = key_variants[rsid]
            
            print(f"\n[VARIANT DETECTED]: {rsid}")
            print(f"  Gene:        {info['gene']}")
            print(f"  Name:        {info['name']}")
            print(f"  Genotype:    {var['genotype']}")
            print(f"  Position:    chr{var['chrom']}:{var['pos']}")
            print(f"  Change:      {var['ref']} -> {var['alt']}")
            print(f"\n  Impact:      {info['impact']}")
            print(f"  Action:      {info['recommendation']}")
    
    if not found_any:
        print("\n  No high-impact variants detected in this sample")
    
    print("\n" + "=" * 80)
    print("[OK] Demo complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Install ML dependencies: uv pip install --system -r requirements.txt")
    print("  2. Run full demo: python demo.py data/clinvar_sample.vcf")
    print("  3. Upload your real VCF from a sequencing provider")
    print("  4. Register at GenomeIndia for Indian population-specific data")
    print("=" * 80)


if __name__ == "__main__":
    # Check if VCF file provided
    if len(sys.argv) > 1:
        vcf_path = Path(sys.argv[1])
    else:
        # Use clinical sample VCF
        vcf_path = Path("data/clinvar_sample.vcf")
    
    if not vcf_path.exists():
        print(f"Error: VCF file not found: {vcf_path}")
        print("\nUsage:")
        print("  python quick_demo.py <path_to_vcf_file>")
        print("\nOr create sample data first:")
        print("  python scripts/download_real_vcf.py")
        sys.exit(1)
    
    # Run demo
    run_quick_demo(vcf_path)
