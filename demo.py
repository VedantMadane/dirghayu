#!/usr/bin/env python3
"""
Dirghayu End-to-End Demo

Demonstrates the complete pipeline:
1. Parse VCF file
2. Annotate variants with public databases
3. Train nutrient deficiency predictor
4. Generate personalized health report
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import VariantAnnotator, parse_vcf_file
from models import NutrientPredictor


def run_demo(vcf_path: Path):
    """Run complete Dirghayu pipeline demo"""

    print("=" * 80)
    print("DIRGHAYU: India-First Longevity Genomics Platform")
    print("=" * 80)

    # Step 1: Parse VCF
    print("\n[1/4] Parsing VCF file...")
    print(f"  Input: {vcf_path}")

    variants_df = parse_vcf_file(vcf_path)
    print(f"  [OK] Found {len(variants_df)} variants")

    if len(variants_df) == 0:
        print("  [!] No variants found!")
        return

    print("\n  Sample variants:")
    print(variants_df[["chrom", "pos", "rsid", "ref", "alt", "genotype"]].head())

    # Step 2: Annotate variants
    print("\n[2/4] Annotating variants with public databases...")
    print("  Sources: Ensembl VEP, gnomAD")
    print("  [!] This makes API calls - may take 30-60 seconds")

    annotator = VariantAnnotator()
    annotated_df = annotator.annotate_dataframe(variants_df)

    print("\n  [OK] Annotation complete!")
    print("\n  Annotated variants:")
    print(annotated_df[["rsid", "gene_symbol", "consequence", "gnomad_af"]].head())

    # Step 3: Train model (on synthetic data for demo)
    print("\n[3/4] Training nutrient deficiency predictor...")
    print("  [!] Using synthetic data for demonstration")

    predictor = NutrientPredictor()
    predictor.train(
        variants_df=annotated_df,
        labels_df=None,  # Would be real clinical data
        epochs=30,
    )

    # Save model
    model_path = Path("models/nutrient_predictor.pth")
    predictor.save(model_path)

    # Step 4: Generate predictions
    print("\n[4/4] Generating personalized health predictions...")

    predictions = predictor.predict(annotated_df)

    print("\n" + "=" * 80)
    print("HEALTH PREDICTION REPORT")
    print("=" * 80)

    # Display nutrient deficiency risks
    print("\n[NUTRIENT DEFICIENCY RISK ASSESSMENT]")
    print("-" * 80)

    risk_levels = {
        (0.0, 0.3): ("LOW", "[LOW]"),
        (0.3, 0.6): ("MODERATE", "[MOD]"),
        (0.6, 1.0): ("HIGH", "[HIGH]"),
    }

    for nutrient, risk_score in predictions.items():
        # Determine risk level
        level, icon = "UNKNOWN", "[?]"
        for (low, high), (lvl, icn) in risk_levels.items():
            if low <= risk_score < high:
                level, icon = lvl, icn
                break

        nutrient_name = nutrient.replace("_", " ").title()
        print(f"\n{icon} {nutrient_name}:")
        print(f"   Risk Score: {risk_score:.2%}")
        print(f"   Risk Level: {level}")

        # Provide recommendations based on risk
        if risk_score > 0.6:
            recommendations = get_recommendations(nutrient)
            print("   Recommendations:")
            for rec in recommendations:
                print(f"     - {rec}")

    # Genetic insights from annotated variants
    print("\n" + "=" * 80)
    print("🧬 GENETIC INSIGHTS")
    print("=" * 80)

    # Look for key variants
    key_variants = {
        "rs1801133": "MTHFR C677T - Affects folate metabolism",
        "rs429358": "APOE e4 - Increased Alzheimer's risk",
        "rs601338": "FUT2 - Affects vitamin B12 absorption",
        "rs2228570": "VDR FokI - Affects vitamin D receptor",
    }

    found_variants = annotated_df[annotated_df["rsid"].isin(key_variants.keys())]

    if len(found_variants) > 0:
        print("\nKey variants detected:")
        for _, var in found_variants.iterrows():
            rsid = var["rsid"]
            if rsid in key_variants:
                print(f"\n  - {rsid} ({var['genotype']})")
                print(f"    Gene: {var.get('gene_symbol', 'Unknown')}")
                print(f"    Impact: {key_variants[rsid]}")
                print(f"    Population frequency: {var.get('gnomad_af', 'Unknown')}")
    else:
        print("\n  No high-impact variants detected in this sample")

    print("\n" + "=" * 80)
    print("[OK] Demo complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Upload your real VCF file from a sequencing provider")
    print("  2. Register at GenomeIndia for Indian population-specific data")
    print("  3. Run full annotation pipeline with AlphaMissense locally")
    print("  4. Train models on real clinical outcome data")
    print("\n  For API access: python src/api/server.py")
    print("=" * 80)


def get_recommendations(nutrient: str) -> list:
    """Get dietary/lifestyle recommendations for nutrient deficiency risk"""

    recommendations = {
        "vitamin_b12": [
            "Consider B12 supplementation (methylcobalamin 1000 mcg/day)",
            "Increase fortified foods (cereals, plant milk)",
            "If vegetarian, consult about B12 injections",
            "Monitor serum B12 levels every 6 months",
        ],
        "vitamin_d": [
            "Vitamin D3 supplementation (2000 IU/day)",
            "15 minutes sun exposure daily (10 AM - 12 PM)",
            "Include fatty fish, egg yolks, fortified milk",
            "Check 25(OH)D levels quarterly",
        ],
        "iron": [
            "Iron-rich foods (lentils, spinach, fortified grains)",
            "Vitamin C with meals to enhance absorption",
            "Avoid tea/coffee with iron-rich meals",
            "Consider iron supplementation if confirmed deficient",
        ],
        "folate": [
            "Methylfolate supplementation (400-800 mcg/day)",
            "Leafy greens, legumes, fortified grains",
            "Ensure adequate B6 and B12 intake",
            "Monitor homocysteine levels",
        ],
    }

    return recommendations.get(nutrient, ["Consult healthcare provider"])


if __name__ == "__main__":
    # Check if VCF file provided
    if len(sys.argv) > 1:
        vcf_path = Path(sys.argv[1])
    else:
        # Use sample VCF
        vcf_path = Path("data/sample.vcf")

    if not vcf_path.exists():
        print(f"Error: VCF file not found: {vcf_path}")
        print("\nUsage:")
        print("  python demo.py <path_to_vcf_file>")
        print("\nOr create sample data first:")
        print("  python scripts/download_data.py")
        sys.exit(1)

    # Run demo
    run_demo(vcf_path)
