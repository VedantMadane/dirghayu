"""
Dirghayu API Server

FastAPI server that auto-generates OpenAPI 3.0 specification.
Provides endpoints for genomic analysis and health predictions.
"""

import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import VariantAnnotator, parse_vcf_file
from models import NutrientPredictor


# Pydantic models for request/response
class VariantInput(BaseModel):
    """Single variant for annotation"""

    chrom: str = Field(..., example="1", description="Chromosome")
    pos: int = Field(..., example=11856378, description="Position")
    ref: str = Field(..., example="C", description="Reference allele")
    alt: str = Field(..., example="T", description="Alternate allele")


class VariantAnnotationResponse(BaseModel):
    """Annotated variant response"""

    variant_id: str
    chrom: str
    pos: int
    ref: str
    alt: str
    gene_symbol: Optional[str] = None
    consequence: Optional[str] = None
    protein_change: Optional[str] = None
    gnomad_af: Optional[float] = None
    gnomad_af_south_asian: Optional[float] = None


class NutrientPredictionResponse(BaseModel):
    """Nutrient deficiency predictions"""

    vitamin_b12_risk: float = Field(..., ge=0, le=1, description="Risk score 0-1")
    vitamin_d_risk: float = Field(..., ge=0, le=1)
    iron_risk: float = Field(..., ge=0, le=1)
    folate_risk: float = Field(..., ge=0, le=1)
    recommendations: Dict[str, List[str]]


class HealthReportResponse(BaseModel):
    """Comprehensive health report"""

    patient_id: str
    total_variants: int
    annotated_variants: int
    nutrient_predictions: NutrientPredictionResponse
    key_variants: List[Dict]
    risk_summary: Dict[str, str]


# Initialize FastAPI app
app = FastAPI(
    title="Dirghayu API",
    description="""
    **Dirghayu: India-First Longevity-Focused Whole Genome Mapping Platform**
    
    This API provides genomic analysis and health prediction services:
    
    * **Variant Annotation**: Enrich variants with population frequencies, pathogenicity scores
    * **Health Predictions**: ML-based predictions for nutrient deficiencies, disease risks
    * **Personalized Reports**: Comprehensive health reports based on genomics
    
    Built with:
    - FastAPI (auto-generates OpenAPI 3.0 spec)
    - PyTorch for ML models
    - Ensembl VEP & gnomAD for annotation
    """,
    version="0.1.0",
    contact={
        "name": "Dirghayu Team",
        "url": "https://github.com/yourusername/dirghayu",
    },
    license_info={
        "name": "MIT",
    },
)

# Global instances
annotator = VariantAnnotator()
nutrient_predictor = None  # Lazy load


def get_nutrient_predictor():
    """Lazy load nutrient predictor"""
    global nutrient_predictor

    if nutrient_predictor is None:
        model_path = Path("models/nutrient_predictor.pth")

        if model_path.exists():
            nutrient_predictor = NutrientPredictor(model_path)
        else:
            # Train on synthetic data if no model exists
            nutrient_predictor = NutrientPredictor()
            print("⚠ No trained model found, using untrained model")

    return nutrient_predictor


# API Endpoints


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Dirghayu Genomics API",
        "status": "healthy",
        "version": "0.1.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


@app.post("/api/v1/annotate/variant", response_model=VariantAnnotationResponse)
async def annotate_variant(variant: VariantInput):
    """
    Annotate a single genetic variant

    Enriches with:
    - Gene symbol and consequence
    - Population frequencies (gnomAD)
    - Protein-level changes

    **Example:**
    ```json
    {
        "chrom": "1",
        "pos": 11856378,
        "ref": "C",
        "alt": "T"
    }
    ```
    """
    try:
        annotation = annotator.annotate_variant(
            chrom=variant.chrom, pos=variant.pos, ref=variant.ref, alt=variant.alt
        )

        return VariantAnnotationResponse(
            variant_id=annotation.variant_id,
            chrom=annotation.chrom,
            pos=annotation.pos,
            ref=annotation.ref,
            alt=annotation.alt,
            gene_symbol=annotation.gene_symbol,
            consequence=annotation.consequence,
            protein_change=annotation.protein_change,
            gnomad_af=annotation.gnomad_af,
            gnomad_af_south_asian=annotation.gnomad_af_south_asian,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predict/nutrients")
async def predict_nutrients(vcf_file: UploadFile = File(...)):
    """
    Predict nutrient deficiency risks from VCF file

    Upload a VCF file and receive predictions for:
    - Vitamin B12 deficiency risk
    - Vitamin D deficiency risk
    - Iron deficiency risk
    - Folate deficiency risk

    Returns risk scores (0-1) and personalized recommendations.
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".vcf") as tmp:
            content = await vcf_file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # Parse VCF
        variants_df = parse_vcf_file(tmp_path)

        # Annotate
        annotated_df = annotator.annotate_dataframe(variants_df)

        # Predict
        predictor = get_nutrient_predictor()
        predictions = predictor.predict(annotated_df)

        # Generate recommendations
        recommendations = {}
        for nutrient, risk in predictions.items():
            if risk > 0.6:
                recommendations[nutrient] = get_recommendations(nutrient)
            else:
                recommendations[nutrient] = ["Maintain current diet and lifestyle"]

        # Clean up temp file
        tmp_path.unlink()

        return NutrientPredictionResponse(
            vitamin_b12_risk=predictions.get("vitamin_b12", 0.0),
            vitamin_d_risk=predictions.get("vitamin_d", 0.0),
            iron_risk=predictions.get("iron", 0.0),
            folate_risk=predictions.get("folate", 0.0),
            recommendations=recommendations,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analyze/comprehensive", response_model=HealthReportResponse)
async def comprehensive_analysis(vcf_file: UploadFile = File(...), patient_id: str = "unknown"):
    """
    Comprehensive genomic analysis

    Upload VCF and receive:
    - Full variant annotation
    - Nutrient deficiency predictions
    - Key variant identification
    - Risk summary

    This is the main endpoint for complete health reports.
    """
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".vcf") as tmp:
            content = await vcf_file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # Parse VCF
        variants_df = parse_vcf_file(tmp_path)
        total_variants = len(variants_df)

        # Annotate
        annotated_df = annotator.annotate_dataframe(variants_df)
        annotated_count = len(annotated_df)

        # Nutrient predictions
        predictor = get_nutrient_predictor()
        nutrient_risks = predictor.predict(annotated_df)

        recommendations = {}
        for nutrient, risk in nutrient_risks.items():
            if risk > 0.6:
                recommendations[nutrient] = get_recommendations(nutrient)
            else:
                recommendations[nutrient] = ["Maintain current diet"]

        nutrient_response = NutrientPredictionResponse(
            vitamin_b12_risk=nutrient_risks.get("vitamin_b12", 0.0),
            vitamin_d_risk=nutrient_risks.get("vitamin_d", 0.0),
            iron_risk=nutrient_risks.get("iron", 0.0),
            folate_risk=nutrient_risks.get("folate", 0.0),
            recommendations=recommendations,
        )

        # Identify key variants
        key_variant_rsids = {
            "rs1801133": "MTHFR C677T - Folate metabolism",
            "rs429358": "APOE ε4 - Alzheimer's risk",
            "rs601338": "FUT2 - B12 absorption",
            "rs2228570": "VDR FokI - Vitamin D",
        }

        key_variants = []
        for _, var in annotated_df.iterrows():
            if var.get("rsid") in key_variant_rsids:
                key_variants.append(
                    {
                        "rsid": var["rsid"],
                        "gene": var.get("gene_symbol", "Unknown"),
                        "genotype": var["genotype"],
                        "description": key_variant_rsids[var["rsid"]],
                    }
                )

        # Risk summary
        risk_summary = {}
        for nutrient, risk in nutrient_risks.items():
            if risk > 0.6:
                risk_summary[nutrient] = "HIGH"
            elif risk > 0.3:
                risk_summary[nutrient] = "MODERATE"
            else:
                risk_summary[nutrient] = "LOW"

        # Clean up
        tmp_path.unlink()

        return HealthReportResponse(
            patient_id=patient_id,
            total_variants=total_variants,
            annotated_variants=annotated_count,
            nutrient_predictions=nutrient_response,
            key_variants=key_variants,
            risk_summary=risk_summary,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_recommendations(nutrient: str) -> List[str]:
    """Get recommendations for nutrient"""
    recs = {
        "vitamin_b12": [
            "Consider B12 supplementation (1000 mcg/day)",
            "Increase fortified foods",
            "Monitor serum B12 every 6 months",
        ],
        "vitamin_d": [
            "Vitamin D3 supplementation (2000 IU/day)",
            "15 min sun exposure daily",
            "Check 25(OH)D levels quarterly",
        ],
        "iron": [
            "Iron-rich foods (lentils, spinach)",
            "Vitamin C with meals",
            "Avoid tea/coffee with iron-rich meals",
        ],
        "folate": [
            "Methylfolate supplementation (400 mcg/day)",
            "Leafy greens, legumes",
            "Monitor homocysteine levels",
        ],
    }
    return recs.get(nutrient, ["Consult healthcare provider"])


# Run server
if __name__ == "__main__":
    import uvicorn

    print("=" * 80)
    print("Starting Dirghayu API Server")
    print("=" * 80)
    print("\nEndpoints:")
    print("  📊 API Docs (Swagger):  http://localhost:8000/docs")
    print("  📄 OpenAPI Spec:        http://localhost:8000/openapi.json")
    print("  📖 ReDoc:               http://localhost:8000/redoc")
    print("\nExample requests:")
    print("  curl http://localhost:8000/")
    print("  curl -X POST http://localhost:8000/api/v1/annotate/variant \\")
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"chrom":"1","pos":11856378,"ref":"C","alt":"T"}\'')
    print("=" * 80)

    uvicorn.run(app, host="0.0.0.0", port=8000)
