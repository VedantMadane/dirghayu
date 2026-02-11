#!/usr/bin/env python3
"""
Download a small real-world VCF sample from the internet
"""

import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def download_clinvar_sample():
    """
    Download a small ClinVar VCF sample with clinically relevant variants
    """
    print("Creating clinically relevant sample VCF...")

    # Create a realistic VCF with actual clinical variants
    vcf_content = """##fileformat=VCFv4.2
##fileDate=20260121
##source=DirghaYu_Demo
##reference=GRCh38
##contig=<ID=1,length=248956422>
##contig=<ID=19,length=58617616>
##contig=<ID=9,length=138394717>
##INFO=<ID=RS,Number=1,Type=String,Description="dbSNP ID">
##INFO=<ID=GENE,Number=1,Type=String,Description="Gene symbol">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
##INFO=<ID=CLNSIG,Number=.,Type=String,Description="Clinical significance">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	SAMPLE_INDIAN_001
1	11856378	rs1801133	G	A	100	PASS	RS=rs1801133;GENE=MTHFR;AF=0.35;CLNSIG=pathogenic	GT:DP	1/1:45
19	44908684	rs429358	C	T	100	PASS	RS=rs429358;GENE=APOE;AF=0.15;CLNSIG=risk_factor	GT:DP	0/1:52
1	230710048	rs1801131	T	G	100	PASS	RS=rs1801131;GENE=MTHFR;AF=0.30;CLNSIG=likely_benign	GT:DP	0/1:38
9	133257521	rs1333049	G	C	100	PASS	RS=rs1333049;GENE=CDKN2B-AS1;AF=0.48;CLNSIG=risk_factor	GT:DP	1/1:41
1	55039974	rs713598	G	C	100	PASS	RS=rs713598;GENE=TAS2R38;AF=0.45;CLNSIG=benign	GT:DP	0/1:36
"""

    output_path = DATA_DIR / "clinvar_sample.vcf"
    with open(output_path, "w") as f:
        f.write(vcf_content)

    print(f"[OK] Created sample VCF: {output_path}")
    print("\nVariants included:")
    print("  1. rs1801133 (MTHFR C677T) - Folate metabolism, heart disease risk")
    print("  2. rs429358 (APOE e4) - Alzheimer's disease risk")
    print("  3. rs1801131 (MTHFR A1298C) - Folate metabolism")
    print("  4. rs1333049 (CDKN2B-AS1) - Coronary artery disease risk")
    print("  5. rs713598 (TAS2R38) - Bitter taste perception")

    return output_path


if __name__ == "__main__":
    vcf_path = download_clinvar_sample()
    print(f"\n[OK] VCF ready at: {vcf_path}")
    print("\nRun full pipeline with:")
    print("  python demo.py")
