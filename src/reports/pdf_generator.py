"""
Clinical Report Generator

Generates a professional PDF report of genomic findings.
Uses FPDF for layout and includes charts/images.
"""

import os
import tempfile
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
from fpdf import FPDF


class ClinicalReport(FPDF):
    def header(self):
        # Logo
        # self.image('logo.png', 10, 8, 33)
        self.set_font("Arial", "B", 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, "Dirghayu Clinical Genomics Report", 0, 0, "C")
        # Line break
        self.ln(20)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font("Arial", "I", 8)
        # Page number
        self.cell(0, 10, "Page " + str(self.page_no()) + "/{nb}", 0, 0, "C")


class ReportGenerator:
    def __init__(self, patient_info: Dict[str, str]):
        self.pdf = ClinicalReport()
        self.pdf.alias_nb_pages()
        self.patient_info = patient_info

    def generate(
        self,
        lifespan_data: Dict,
        disease_risks: Dict,
        top_variants: List[Dict],
        pharmacogenomics: List[Dict],
        output_path: str = "report.pdf",
    ):
        self.pdf.add_page()

        # 1. Patient Summary
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 10, "Patient Information", 0, 1)
        self.pdf.set_font("Arial", "", 10)

        for k, v in self.patient_info.items():
            self.pdf.cell(50, 8, f"{k}: {v}", 0, 1)

        self.pdf.ln(5)
        self.pdf.cell(0, 8, f"Report Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
        self.pdf.ln(10)

        # 2. Executive Summary (Longevity)
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 10, "Executive Summary: Longevity & Aging", 0, 1)
        self.pdf.set_font("Arial", "", 10)

        bio_age = lifespan_data.get("biological_age", "N/A")
        pred_life = lifespan_data.get("predicted_lifespan", "N/A")

        self.pdf.multi_cell(
            0,
            6,
            f"Based on the genetic analysis, the patient's estimated Biological Age is {bio_age:.1f} years. "
            f"The projected lifespan, assuming current lifestyle factors, is approximately {pred_life:.1f} years. "
            "This is influenced by key variants in longevity-associated genes (e.g., FOXO3A).",
        )
        self.pdf.ln(10)

        # 3. Disease Risk Profile
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 10, "Disease Risk Profile", 0, 1)
        self.pdf.set_font("Arial", "", 10)

        # Create a simple bar chart image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig, ax = plt.subplots(figsize=(6, 3))
            diseases = list(disease_risks.keys())
            scores = list(disease_risks.values())
            colors = ["red" if s > 0.7 else "orange" if s > 0.4 else "green" for s in scores]

            ax.barh(diseases, scores, color=colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Risk Score (0-1)")
            plt.tight_layout()
            plt.savefig(tmp.name)
            plt.close()

            self.pdf.image(tmp.name, x=10, w=170)
            os.unlink(tmp.name)

        self.pdf.ln(80)  # Move past image

        # 4. Pharmacogenomics (GNN Insights)
        self.pdf.add_page()
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 10, "Pharmacogenomic Insights (Drug Response)", 0, 1)
        self.pdf.set_font("Arial", "", 10)

        self.pdf.multi_cell(
            0,
            6,
            "The following drug-gene interactions were analyzed using our Graph Neural Network model. "
            "These predictions indicate likely efficacy and toxicity risks.",
        )
        self.pdf.ln(5)

        # Table Header
        self.pdf.set_font("Arial", "B", 10)
        self.pdf.cell(40, 8, "Drug", 1)
        self.pdf.cell(40, 8, "Gene", 1)
        self.pdf.cell(30, 8, "Efficacy", 1)
        self.pdf.cell(30, 8, "Toxicity Risk", 1)
        self.pdf.cell(50, 8, "Recommendation", 1)
        self.pdf.ln()

        self.pdf.set_font("Arial", "", 9)
        for pgx in pharmacogenomics:
            drug = pgx.get("drug", "N/A")
            gene = pgx.get("gene", "N/A")
            eff = pgx.get("efficacy", 0.0)
            tox = pgx.get("toxicity", 0.0)
            rec = pgx.get("recommendation", "Standard Dose")

            self.pdf.cell(40, 8, drug, 1)
            self.pdf.cell(40, 8, gene, 1)
            self.pdf.cell(30, 8, f"{eff * 100:.0f}%", 1)
            self.pdf.cell(30, 8, f"{tox * 100:.0f}%", 1)
            self.pdf.cell(50, 8, rec[:25], 1)  # Truncate if long
            self.pdf.ln()

        self.pdf.ln(10)

        # 5. Key Variants
        self.pdf.set_font("Arial", "B", 12)
        self.pdf.cell(0, 10, "Significant Genetic Variants Detected", 0, 1)
        self.pdf.set_font("Arial", "", 10)

        for v in top_variants:
            rsid = v.get("rsid", "N/A")
            gene = v.get("gene", "N/A")
            impact = v.get("impact", "Unknown")

            self.pdf.set_font("Arial", "B", 10)
            self.pdf.cell(0, 6, f"{rsid} ({gene})", 0, 1)
            self.pdf.set_font("Arial", "", 10)
            self.pdf.multi_cell(0, 6, f"Impact: {impact}")
            self.pdf.ln(2)

        # Output
        self.pdf.output(output_path)
        return output_path
