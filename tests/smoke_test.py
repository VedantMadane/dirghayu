
import pytest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reports.pdf_generator import ReportGenerator
from models.drug_response_gnn import DrugGeneGNN, predict_drug_response
from models.lifespan_net import LifespanNetIndia
import torch

def test_report_generation():
    """Smoke test for PDF generation"""
    patient_info = {"Name": "Test Patient", "Age": "30"}
    generator = ReportGenerator(patient_info)

    # Mock data
    lifespan = {"biological_age": 35.0, "predicted_lifespan": 80.0}
    disease_risks = {"CVD": 0.2, "T2D": 0.5}
    top_variants = [{"rsid": "rs123", "gene": "TEST", "impact": "Low"}]
    pgx = [{"drug": "Aspirin", "gene": "GENE1", "efficacy": 0.8, "toxicity": 0.1}]

    # Output to temp file
    out_path = "test_report.pdf"
    try:
        generator.generate(lifespan, disease_risks, top_variants, pgx, out_path)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)

def test_gnn_model():
    """Smoke test for DrugGeneGNN"""
    model = DrugGeneGNN(num_genes=10, num_drugs=10)
    g_idx = torch.tensor([0, 1])
    d_idx = torch.tensor([0, 1])

    out = model(g_idx, d_idx)
    assert "efficacy" in out
    assert "toxicity_risk" in out
    assert out["efficacy"].shape == (2, 1)

def test_predict_wrapper():
    """Test the wrapper function"""
    # Should handle unknown drugs gracefully
    res = predict_drug_response("UnknownDrug", "UnknownGene")
    assert "efficacy" in res

    # Should work for known drugs (mocked)
    res = predict_drug_response("Clopidogrel", "CYP2C19")
    assert 0 <= res["efficacy"] <= 1

def test_lifespan_model_dims():
    """Ensure model accepts 100 clinical features"""
    model = LifespanNetIndia(genomic_dim=50, clinical_dim=100, lifestyle_dim=10)
    g = torch.randn(1, 50)
    c = torch.randn(1, 100)
    l = torch.randn(1, 10)

    out = model(g, c, l)
    assert "predicted_lifespan" in out
