"""
Pharmacogenomics Module

Predicts drug metabolism and response based on genetic variants.
Critical for personalized medicine, especially for Indian population.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import pandas as pd


class MetabolizerStatus(Enum):
    """Drug metabolizer phenotypes"""
    POOR = "poor"  # Very slow metabolism
    INTERMEDIATE = "intermediate"  # Slow metabolism
    NORMAL = "normal"  # Normal metabolism
    RAPID = "rapid"  # Fast metabolism
    ULTRA_RAPID = "ultra_rapid"  # Very fast metabolism


@dataclass
class DrugRecommendation:
    """Personalized drug recommendation"""
    drug_name: str
    metabolizer_status: MetabolizerStatus
    dose_adjustment: str
    alternative_drugs: List[str]
    warnings: List[str]
    evidence_level: str  # "high", "moderate", "low"
    clinical_note: str


class PharmacogenomicsAnalyzer:
    """
    Analyzes pharmacogenomic variants to predict drug response
    
    Focus on drugs commonly prescribed in India:
    - Clopidogrel (anti-platelet, after heart stent)
    - Warfarin (blood thinner)
    - Statins (cholesterol)
    - Metformin (diabetes)
    - Codeine (pain)
    """
    
    # CYP2C19 star alleles (Clopidogrel metabolism)
    # CRITICAL for India: 30% of Indians are poor metabolizers
    CYP2C19_ALLELES = {
        "reference": {"rsids": [], "activity": "normal"},
        "*2": {"rsids": ["rs4244285"], "activity": "none"},  # Most common loss-of-function
        "*3": {"rsids": ["rs4986893"], "activity": "none"},
        "*17": {"rsids": ["rs12248560"], "activity": "increased"},
    }
    
    # CYP2C9 + VKORC1 (Warfarin dosing)
    WARFARIN_GENES = {
        "CYP2C9": {
            "*2": {"rs1799853": "T"},  # Reduced activity
            "*3": {"rs1057910": "C"},  # Reduced activity
        },
        "VKORC1": {
            "rs9923231": {"T": "sensitive", "C": "normal"}
        }
    }
    
    # SLCO1B1 (Statin side effects)
    STATIN_VARIANTS = {
        "rs4149056": {
            "T/T": "normal_risk",
            "C/T": "increased_risk",
            "C/C": "high_risk"  # 17x higher myopathy risk
        }
    }
    
    # CYP2D6 (Codeine, tramadol, many antidepressants)
    CYP2D6_VARIANTS = {
        # Complex gene with copy number variations
        "*4": {"rs3892097": "none"},  # Most common null allele
        "*10": {"rs1065852": "decreased"},  # Common in Asians
        "*41": {"rs28371725": "decreased"}
    }
    
    # SLC22A1 (Metformin response)
    METFORMIN_VARIANTS = {
        "rs622342": {
            "A/A": "normal_response",
            "A/C": "reduced_response",
            "C/C": "reduced_response"
        }
    }
    
    def __init__(self):
        self.recommendations = []
    
    def analyze_clopidogrel(self, variants_df: pd.DataFrame) -> DrugRecommendation:
        """
        Analyze CYP2C19 for clopidogrel (Plavix) response
        
        CRITICAL IN INDIA:
        - 30% of Indians are CYP2C19 poor metabolizers
        - Clopidogrel is inactive prodrug, needs CYP2C19 to activate
        - Poor metabolizers have 3x higher risk of stent thrombosis
        """
        
        # Check for loss-of-function alleles
        has_star2 = self._check_variant(variants_df, "rs4244285", "A")
        has_star3 = self._check_variant(variants_df, "rs4986893", "A")
        has_star17 = self._check_variant(variants_df, "rs12248560", "T")
        
        # Determine metabolizer status
        lof_count = sum([has_star2, has_star3])
        
        if lof_count >= 2:
            status = MetabolizerStatus.POOR
            dose_adj = "AVOID clopidogrel"
            alternatives = ["Prasugrel", "Ticagrelor"]
            warnings = [
                "⚠ CRITICAL: Poor metabolizer",
                "Clopidogrel unlikely to be effective",
                "3x higher risk of cardiovascular events",
                "Switch to alternative antiplatelet agent"
            ]
            
        elif lof_count == 1:
            status = MetabolizerStatus.INTERMEDIATE
            dose_adj = "Consider higher dose (150mg vs 75mg) OR switch to alternative"
            alternatives = ["Prasugrel", "Ticagrelor"]
            warnings = [
                "Intermediate metabolizer",
                "Reduced clopidogrel effectiveness",
                "Consider alternative or higher dose"
            ]
            
        elif has_star17:
            status = MetabolizerStatus.RAPID
            dose_adj = "Standard dose (75mg)"
            alternatives = []
            warnings = [
                "Rapid metabolizer",
                "Standard clopidogrel dosing appropriate",
                "May have increased bleeding risk"
            ]
            
        else:
            status = MetabolizerStatus.NORMAL
            dose_adj = "Standard dose (75mg)"
            alternatives = []
            warnings = []
        
        return DrugRecommendation(
            drug_name="Clopidogrel (Plavix)",
            metabolizer_status=status,
            dose_adjustment=dose_adj,
            alternative_drugs=alternatives,
            warnings=warnings,
            evidence_level="high",
            clinical_note=(
                "CYP2C19 testing is FDA-recommended before clopidogrel use. "
                "Particularly important in Indian population where 30% are poor metabolizers."
            )
        )
    
    def analyze_warfarin(self, variants_df: pd.DataFrame) -> DrugRecommendation:
        """
        Analyze CYP2C9 and VKORC1 for warfarin dosing
        
        Warfarin has narrow therapeutic window
        Genetic variants explain 30-50% of dose variability
        """
        
        # CYP2C9 status
        has_star2 = self._check_variant(variants_df, "rs1799853", "T")
        has_star3 = self._check_variant(variants_df, "rs1057910", "C")
        
        # VKORC1 sensitivity
        vkorc1_genotype = self._get_genotype(variants_df, "rs9923231")
        
        # Calculate dose adjustment
        if has_star2 or has_star3:
            cyp2c9_factor = 0.7 if (has_star2 or has_star3) else 1.0
            cyp2c9_factor = 0.5 if (has_star2 and has_star3) else cyp2c9_factor
        else:
            cyp2c9_factor = 1.0
        
        if vkorc1_genotype == "T/T":
            vkorc1_factor = 0.6  # Sensitive, need lower dose
        elif vkorc1_genotype in ["C/T", "T/C"]:
            vkorc1_factor = 0.8
        else:
            vkorc1_factor = 1.0
        
        combined_factor = cyp2c9_factor * vkorc1_factor
        standard_dose = 5.0  # mg/day
        recommended_dose = standard_dose * combined_factor
        
        if combined_factor < 0.6:
            warnings = [
                "⚠ Sensitive to warfarin",
                f"Start with {recommended_dose:.1f}mg/day (vs standard 5mg)",
                "Increased bleeding risk with standard dosing",
                "Monitor INR closely"
            ]
        else:
            warnings = []
        
        return DrugRecommendation(
            drug_name="Warfarin",
            metabolizer_status=MetabolizerStatus.NORMAL,  # Not applicable
            dose_adjustment=f"Start with {recommended_dose:.1f}mg/day",
            alternative_drugs=["Apixaban", "Rivaroxaban (don't require monitoring)"],
            warnings=warnings,
            evidence_level="high",
            clinical_note=(
                f"Genetic-guided dosing. Standard dose: 5mg. "
                f"Recommended: {recommended_dose:.1f}mg based on CYP2C9/VKORC1."
            )
        )
    
    def analyze_statins(self, variants_df: pd.DataFrame) -> DrugRecommendation:
        """
        Analyze SLCO1B1 for statin-induced myopathy risk
        
        Statins are very commonly prescribed in India for cholesterol
        """
        
        genotype = self._get_genotype(variants_df, "rs4149056")
        
        if genotype == "C/C":
            risk = "high"
            warnings = [
                "⚠ HIGH RISK of statin-induced myopathy",
                "17x higher risk with simvastatin 80mg",
                "Avoid high-dose simvastatin",
                "Consider alternative statin or lower dose"
            ]
            alternatives = [
                "Rosuvastatin (lower myopathy risk)",
                "Pravastatin (not affected by SLCO1B1)",
                "Atorvastatin at lower doses"
            ]
            dose_adj = "Avoid simvastatin >40mg. Use alternative statin."
            
        elif genotype in ["C/T", "T/C"]:
            risk = "moderate"
            warnings = [
                "Moderate risk of statin-induced myopathy",
                "Avoid high-dose simvastatin (80mg)",
                "Monitor for muscle pain"
            ]
            alternatives = ["Rosuvastatin", "Pravastatin"]
            dose_adj = "Use simvastatin ≤40mg OR switch to alternative"
            
        else:  # T/T
            risk = "low"
            warnings = []
            alternatives = []
            dose_adj = "Standard dosing appropriate"
        
        return DrugRecommendation(
            drug_name="Statins (especially Simvastatin)",
            metabolizer_status=MetabolizerStatus.NORMAL,
            dose_adjustment=dose_adj,
            alternative_drugs=alternatives,
            warnings=warnings,
            evidence_level="high",
            clinical_note=(
                f"SLCO1B1 *5 (rs4149056) genotype: {genotype}. "
                f"Myopathy risk: {risk}. FDA label includes this information."
            )
        )
    
    def analyze_metformin(self, variants_df: pd.DataFrame) -> DrugRecommendation:
        """
        Analyze SLC22A1 for metformin response
        
        Metformin is first-line for Type 2 diabetes (very common in India)
        """
        
        genotype = self._get_genotype(variants_df, "rs622342")
        
        if genotype in ["C/C", "A/C", "C/A"]:
            warnings = [
                "Reduced metformin response",
                "May need higher doses",
                "Alternative medications may be more effective"
            ]
            alternatives = [
                "DPP-4 inhibitors",
                "SGLT2 inhibitors",
                "Sulfonylureas (check for other genetic factors)"
            ]
            dose_adj = "May need higher metformin doses OR consider alternatives"
            
        else:  # A/A
            warnings = []
            alternatives = []
            dose_adj = "Standard metformin dosing"
        
        return DrugRecommendation(
            drug_name="Metformin",
            metabolizer_status=MetabolizerStatus.NORMAL,
            dose_adjustment=dose_adj,
            alternative_drugs=alternatives,
            warnings=warnings,
            evidence_level="moderate",
            clinical_note=(
                f"SLC22A1 genotype: {genotype}. "
                "Metformin response is also influenced by lifestyle factors."
            )
        )
    
    def analyze_codeine(self, variants_df: pd.DataFrame) -> DrugRecommendation:
        """
        Analyze CYP2D6 for codeine metabolism
        
        Codeine is prodrug, converted to morphine by CYP2D6
        """
        
        # Simplified analysis (CYP2D6 is complex with CNVs)
        has_star4 = self._check_variant(variants_df, "rs3892097", "A")
        has_star10 = self._check_variant(variants_df, "rs1065852", "T")
        
        if has_star4:
            status = MetabolizerStatus.POOR
            warnings = [
                "⚠ Poor CYP2D6 metabolizer",
                "Codeine will NOT be effective for pain relief",
                "Codeine not converted to active morphine"
            ]
            alternatives = ["Morphine", "Oxycodone", "Hydromorphone", "Non-opioid analgesics"]
            dose_adj = "AVOID codeine - will not work"
            
        elif has_star10:
            status = MetabolizerStatus.INTERMEDIATE
            warnings = ["Reduced codeine effectiveness"]
            alternatives = ["Alternative opioid or higher dose"]
            dose_adj = "May need higher doses or alternative"
            
        else:
            status = MetabolizerStatus.NORMAL
            warnings = []
            alternatives = []
            dose_adj = "Standard codeine dosing"
        
        return DrugRecommendation(
            drug_name="Codeine",
            metabolizer_status=status,
            dose_adjustment=dose_adj,
            alternative_drugs=alternatives,
            warnings=warnings,
            evidence_level="high",
            clinical_note=(
                "CYP2D6 also affects many antidepressants (SSRIs, TCAs) "
                "and other opioids (tramadol, oxycodone)."
            )
        )
    
    def comprehensive_analysis(self, variants_df: pd.DataFrame) -> Dict[str, DrugRecommendation]:
        """
        Run all pharmacogenomic analyses
        
        Returns dictionary of drug recommendations
        """
        
        return {
            "clopidogrel": self.analyze_clopidogrel(variants_df),
            "warfarin": self.analyze_warfarin(variants_df),
            "statins": self.analyze_statins(variants_df),
            "metformin": self.analyze_metformin(variants_df),
            "codeine": self.analyze_codeine(variants_df)
        }
    
    def _check_variant(self, df: pd.DataFrame, rsid: str, alt_allele: str) -> bool:
        """Check if variant is present"""
        if rsid not in df['rsid'].values:
            return False
        
        row = df[df['rsid'] == rsid].iloc[0]
        genotype = row['genotype']
        
        # Check if alt allele is present
        return alt_allele in genotype and genotype != "0/0"
    
    def _get_genotype(self, df: pd.DataFrame, rsid: str) -> str:
        """Get genotype for variant"""
        if rsid not in df['rsid'].values:
            return "unknown"
        
        row = df[df['rsid'] == rsid].iloc[0]
        
        # Convert 0/0, 0/1, 1/1 to actual alleles
        ref = row['ref']
        alt = row['alt']
        genotype = row['genotype']
        
        if genotype == "0/0":
            return f"{ref}/{ref}"
        elif genotype in ["0/1", "1/0"]:
            return f"{ref}/{alt}"
        elif genotype == "1/1":
            return f"{alt}/{alt}"
        else:
            return "unknown"


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../..")
    from src.data import parse_vcf_file
    
    # Parse VCF
    vcf_path = "../../data/sample.vcf"
    variants_df = parse_vcf_file(vcf_path)
    
    # Run pharmacogenomics analysis
    pgx = PharmacogenomicsAnalyzer()
    results = pgx.comprehensive_analysis(variants_df)
    
    print("=" * 80)
    print("PHARMACOGENOMICS REPORT")
    print("=" * 80)
    
    for drug, recommendation in results.items():
        print(f"\n### {recommendation.drug_name}")
        print(f"Metabolizer Status: {recommendation.metabolizer_status.value}")
        print(f"Dose Adjustment: {recommendation.dose_adjustment}")
        
        if recommendation.warnings:
            print("\nWarnings:")
            for warning in recommendation.warnings:
                print(f"  {warning}")
        
        if recommendation.alternative_drugs:
            print(f"\nAlternatives: {', '.join(recommendation.alternative_drugs)}")
        
        print(f"\nClinical Note: {recommendation.clinical_note}")
        print(f"Evidence Level: {recommendation.evidence_level}")
        print("-" * 80)
