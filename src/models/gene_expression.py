"""
Gene Expression & Backtracking Model

Maps lifestyle/environmental interventions to gene expression changes.
Used for "Explainability & Backtracking" features.
"""

from typing import Dict, List, TypedDict


class PrecautionImpact(TypedDict):
    precaution: str
    mechanism: str
    target_genes: List[str]
    expression_effect: str  # "Upregulated" or "Downregulated"
    clinical_benefit: str


class BacktrackingEngine:
    def __init__(self):
        # Knowledge Base: Precaution -> Gene Expression
        self.knowledge_base = {
            "cvd": [
                {
                    "precaution": "Mediterranean Diet (Olive Oil)",
                    "mechanism": "Polyphenols reduce oxidative stress",
                    "target_genes": ["PON1", "LDLR"],
                    "expression_effect": "Upregulated",
                    "clinical_benefit": "Improved lipid clearance",
                },
                {
                    "precaution": "Aerobic Exercise",
                    "mechanism": "Shear stress on endothelium",
                    "target_genes": ["eNOS", "VEGF"],
                    "expression_effect": "Upregulated",
                    "clinical_benefit": "Better vasodilation and blood pressure control",
                },
            ],
            "t2d": [
                {
                    "precaution": "Increase Soluble Fiber",
                    "mechanism": "Short-chain fatty acid production",
                    "target_genes": ["GLP1", "PYY"],
                    "expression_effect": "Upregulated",
                    "clinical_benefit": "Enhanced insulin secretion",
                },
                {
                    "precaution": "Intermittent Fasting",
                    "mechanism": "AMPK activation pathway",
                    "target_genes": ["SIRT1", "PPARG"],
                    "expression_effect": "Modulated",
                    "clinical_benefit": "Improved insulin sensitivity",
                },
            ],
            "cancer": [
                {
                    "precaution": "Curcumin (Turmeric) Intake",
                    "mechanism": "Anti-inflammatory signaling inhibition",
                    "target_genes": ["NF-kB", "COX-2", "TNF-alpha"],
                    "expression_effect": "Downregulated",
                    "clinical_benefit": "Reduced chronic inflammation and tumor promotion",
                },
                {
                    "precaution": "Cruciferous Vegetables (Broccoli)",
                    "mechanism": "Sulforaphane pathway",
                    "target_genes": ["Nrf2", "GSTP1"],
                    "expression_effect": "Upregulated",
                    "clinical_benefit": "Enhanced detoxification of carcinogens",
                },
            ],
            "longevity": [
                {
                    "precaution": "Caloric Restriction",
                    "mechanism": "mTOR inhibition",
                    "target_genes": ["mTOR", "IGF-1"],
                    "expression_effect": "Downregulated",
                    "clinical_benefit": "Extended healthspan and cellular repair",
                }
            ],
        }

    def backtrack_risk(self, disease_type: str) -> List[PrecautionImpact]:
        """
        Given a disease risk, return actionable precautions and their
        genetic mechanisms (Backtracking).
        """
        return self.knowledge_base.get(disease_type, [])

    def simulate_gene_response(self, genes: List[str], intervention: str) -> Dict[str, float]:
        """
        Simulate quantitative gene expression change for an intervention.
        (Mock logic for visualization)
        """
        changes = {}
        for gene in genes:
            # Random but consistent change based on hash
            seed = hash(intervention + gene) % 200
            change = (seed - 100) / 50.0  # -2.0 to +2.0 fold change
            changes[gene] = change
        return changes
