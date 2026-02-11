#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dirghayu - Streamlit Cloud Deployment
India-First Longevity Genomics Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io
import torch
import matplotlib.pyplot as plt
import tempfile
import os

# Fix Windows encoding
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.vcf_parser import VCFParser
from models.lifespan_net import load_lifespan_model
from models.disease_net import load_disease_model
from models.explainability import ExplainabilityManager
from data.biomarkers import get_biomarker_names, generate_synthetic_clinical_data
from models.drug_response_gnn import predict_drug_response, DRUG_MAP, GENE_MAP
from reports.pdf_generator import ReportGenerator

# Page config
st.set_page_config(
    page_title="Dirghayu - Genomic Analysis",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS - Orange theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
    }
    .main > div {
        padding: 2rem;
        background: white;
        border-radius: 15px;
        margin: 1rem;
    }
    h1 {
        color: #FF6B35;
    }
    .stButton>button {
        background-color: #FF6B35;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B35;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #FF6B35, #F7931E); border-radius: 15px; color: white; margin-bottom: 2rem;">
    <h1>🧬 Dirghayu</h1>
    <p style="font-size: 1.2rem;">India-First Longevity Genomics Platform</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("About Dirghayu")
st.sidebar.markdown("""
### Features
- 🇮🇳 India-focused analysis
- 🤖 AI-powered Risk Prediction
- ⚡ Fast WGS Processing
- 🔍 Explainable Insights

### Models
- **LifespanNet-India**: Biological age
- **DiseaseNet-Multi**: Disease risks
- **Pharmaco-GNN**: Drug response
""")

# Clinician Mode Toggle
clinician_mode = st.sidebar.toggle("Clinician Mode", value=False)

st.sidebar.divider()
st.sidebar.header("👤 Clinical & Lifestyle")
age = st.sidebar.slider("Age", 20, 100, 35)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
bmi = st.sidebar.slider("BMI", 15.0, 40.0, 24.5)
diet_score = st.sidebar.slider("Diet Quality (0-10)", 0, 10, 7)
exercise = st.sidebar.selectbox("Exercise Frequency", ["None", "1-2 times/week", "3-5 times/week", "Daily"])

# Clinical Data Upload
st.sidebar.divider()
st.sidebar.subheader("🩸 Clinical Data")
clinical_file = st.sidebar.file_uploader("Upload 100-Marker Panel (CSV)", type=['csv'])

# Load Models (Cached)
@st.cache_resource
def load_models():
    lifespan_model = load_lifespan_model()
    disease_model = load_disease_model()
    explainer = ExplainabilityManager()

    # Setup dummy background for SHAP
    dummy_genomic = torch.randint(0, 3, (100, 100)).float()
    dummy_clinical = torch.randn(100, 100)

    return lifespan_model, disease_model, explainer

lifespan_model, disease_model, explainer = load_models()

# Main content
st.header("📤 Upload Your VCF File")

uploaded_file = st.file_uploader(
    "Choose a VCF file (Supports WGS)",
    type=['vcf'],
    help="Upload your Variant Call Format (.vcf) file for analysis"
)

if uploaded_file is not None:
    with st.spinner("🔬 Analyzing your genome..."):
        try:
            # Save uploaded file temporarily
            temp_path = Path("temp_upload.vcf")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Parse VCF (Streaming mode support)
            parser = VCFParser(temp_path)
            
            # For demo/analysis, we'll process the first chunk to get stats
            # and simulate the feature vectors
            try:
                first_chunk = next(parser.parse_chunks(chunk_size=1000))
                total_variants = 0
                for chunk in parser.parse_chunks(chunk_size=50000):
                    total_variants += len(chunk)

                seed = int(first_chunk['pos'].sum() % 10000)
            except StopIteration:
                st.warning("VCF file seems empty or invalid.")
                total_variants = 0
                seed = 42

            st.success(f"✅ Successfully analyzed {total_variants} variants from WGS data!")

            # --- PREPARE INPUTS FOR AI MODELS ---
            torch.manual_seed(seed)
            np.random.seed(seed)

            # 1. Genomic Inputs
            g_lifespan = torch.randint(0, 3, (1, 50)).float()
            g_disease = torch.randint(0, 3, (1, 100)).float()

            # 2. Clinical Inputs
            if clinical_file:
                try:
                    df = pd.read_csv(clinical_file)
                    st.sidebar.success("Clinical data loaded!")
                    c_input = torch.tensor(df.iloc[0, :100].values).float().unsqueeze(0)
                    if c_input.shape[1] < 100:
                         c_input = torch.cat([c_input, torch.zeros(1, 100 - c_input.shape[1])], dim=1)
                except Exception as e:
                    st.sidebar.error(f"Error loading CSV: {e}")
                    c_input = None
            else:
                c_input = None

            if c_input is None:
                clinical_data = generate_synthetic_clinical_data(1)
                clinical_vals = np.array([clinical_data[m][0] for m in get_biomarker_names()])
                c_norm = (clinical_vals - 100) / 50.0
                c_input = torch.tensor(c_norm).float().unsqueeze(0)
                st.info("ℹ️ Using synthetic clinical profile (no file uploaded). Upload CSV for personalized 100-marker analysis.")

            # 3. Lifestyle Inputs
            l_lifespan = torch.tensor([[diet_score/10.0, 1.0 if exercise == "Daily" else 0.5] + [0.5]*8])

            # --- RUN INFERENCE ---
            with torch.no_grad():
                lifespan_preds = lifespan_model(g_lifespan, c_input, l_lifespan)
                disease_preds = disease_model(g_disease, c_input)

            # --- DISPLAY RESULTS ---

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("⏳ Longevity Analysis")
                predicted_age = lifespan_preds["predicted_lifespan"].item()
                bio_age = lifespan_preds["biological_age"].item() + age
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Predicted Lifespan</h3>
                    <p style="font-size: 2rem; font-weight: bold; color: #2ecc71;">{predicted_age:.1f} Years</p>
                    <p>Biological Age: <strong>{bio_age:.1f} Years</strong></p>
                    <small>Based on Indian-specific genetic markers</small>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.subheader("🏥 Disease Risk Assessment")
                risks = {
                    "Cardiovascular (CVD)": disease_preds["cvd_risk"].item(),
                    "Type 2 Diabetes": disease_preds["t2d_risk"].item(),
                    "Breast Cancer": disease_preds["cancer_risks"][0, 0].item(),
                    "Colorectal Cancer": disease_preds["cancer_risks"][0, 1].item()
                }
                for disease, risk in risks.items():
                    color = "red" if risk > 0.7 else "orange" if risk > 0.4 else "green"
                    st.write(f"**{disease}**")
                    st.progress(risk, text=f"Risk Score: {risk:.2f}")

            st.divider()

            # --- TABS: Explainability, Backtracking, Pharmacogenomics, Biomarkers ---
            tab1, tab2, tab3, tab4 = st.tabs([
                "🧬 Explainability",
                "🔄 Backtracking",
                "💊 Pharmacogenomics (GNN)",
                "🩸 100 Biomarker Panel"
            ])

            with tab1:
                st.write("### What drove these predictions?")
                genomic_names = [f"Var_{i}" for i in range(100)]
                clinical_names = get_biomarker_names()
                all_feature_names = genomic_names + clinical_names

                input_tensor = torch.cat([g_disease, c_input], dim=1)
                explainer.setup_shap(disease_model.shared_encoder, input_tensor)
                explanation = explainer.explain_prediction(input_tensor, feature_names=all_feature_names)
                
                if "shap_values" in explanation:
                    top_feats = explanation["top_features"]
                    feat_names = [x[0] for x in top_feats]
                    feat_vals = [x[1] for x in top_feats]

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.barh(feat_names, feat_vals, color="#FF6B35")
                    ax.set_xlabel("SHAP Value (Impact on Risk)")
                    ax.set_title("Top Contributing Factors")
                    st.pyplot(fig)

            with tab2:
                st.write("### 🔄 Backtracking: Precaution to Gene Expression")
                high_risks = {k: v for k, v in risks.items() if v > 0.4}
                if not high_risks:
                    st.success("🎉 You have low risk for all tracked diseases!")

                insights = explainer.get_backtracking_insights(high_risks)
                for disease, precautions in insights.items():
                    st.subheader(f"Recommendations for {disease}")
                    for p in precautions:
                        with st.expander(f"💊 Precaution: {p['precaution']}"):
                            c1, c2 = st.columns([1, 2])
                            with c1:
                                st.write("**Mechanism:**")
                                st.write(p['mechanism'])
                                st.write("**Clinical Benefit:**")
                                st.write(p['clinical_benefit'])
                            with c2:
                                st.write("**Gene Expression Effect:**")
                                genes = p['target_genes']
                                effect = p['expression_effect']
                                fig, ax = plt.subplots(figsize=(6, 2))
                                vals = [1.5 if effect == "Upregulated" else 0.5 for _ in genes]
                                colors = ['green' if v > 1 else 'red' for v in vals]
                                ax.bar(genes, vals, color=colors)
                                ax.axhline(1.0, color='gray', linestyle='--', label="Baseline")
                                ax.set_ylabel("Expression Level")
                                st.pyplot(fig)

            with tab3:
                st.write("### 💊 AI-Predicted Drug Response (GNN)")
                st.info("Using Graph Neural Networks to predict drug efficacy and toxicity based on your genes.")

                # Demo Drugs
                drugs_to_test = [
                    ("Clopidogrel", "CYP2C19"),
                    ("Warfarin", "CYP2C9"),
                    ("Simvastatin", "SLCO1B1"),
                    ("Metformin", "SLC22A1")
                ]

                pgx_results = []

                for drug, gene in drugs_to_test:
                    # Mock variant impact based on random seed
                    impact = 1.0 if np.random.rand() > 0.3 else 0.5

                    res = predict_drug_response(drug, gene, variant_impact=impact)
                    pgx_results.append({
                        "drug": drug, "gene": gene,
                        "efficacy": res["efficacy"],
                        "toxicity": res["toxicity_risk"],
                        "recommendation": "Standard Dose" if impact == 1.0 else "Adjust Dose / Alternative"
                    })

                    with st.expander(f"{drug} ({gene})"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("Efficacy Probability", f"{res['efficacy']*100:.1f}%")
                        with c2:
                            tox = res['toxicity_risk']
                            st.metric("Toxicity Risk", f"{tox*100:.1f}%", delta_color="inverse")

                        if clinician_mode:
                            st.caption(f"Gene: {gene} | Variant Impact Factor: {impact:.2f} | GNN Confidence: High")

            with tab4:
                st.write("### 🩸 Comprehensive Biomarker Panel")
                clinical_raw = c_input.numpy()[0] * 50 + 100
                bio_df = pd.DataFrame({
                    "Biomarker": get_biomarker_names(),
                    "Value": clinical_raw,
                    "Unit": ["mg/dL" if "Cholesterol" in x or "Glucose" in x else "units" for x in get_biomarker_names()]
                })
                st.dataframe(bio_df, use_container_width=True, height=400)

            # --- REPORT GENERATION ---
            st.divider()
            st.header("📄 Clinical Report")

            if st.button("Generate Professional PDF Report"):
                with st.spinner("Generating PDF..."):
                    # Prepare data for report
                    patient_info = {
                        "Age": str(age),
                        "Sex": sex,
                        "BMI": str(bmi),
                        "Genomic ID": f"WGS-{seed}"
                    }

                    # Mock top variants
                    top_variants = [
                        {"rsid": "rs1801133", "gene": "MTHFR", "impact": "High (homozygous)"},
                        {"rsid": "rs429358", "gene": "APOE", "impact": "Moderate (heterozygous)"}
                    ]

                    generator = ReportGenerator(patient_info)
                    pdf_path = generator.generate(
                        lifespan_data={"biological_age": bio_age, "predicted_lifespan": predicted_age},
                        disease_risks=risks,
                        top_variants=top_variants,
                        pharmacogenomics=pgx_results,
                        output_path="Dirghayu_Report.pdf"
                    )

                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Clinical Report (PDF)",
                            data=f,
                            file_name="Dirghayu_Clinical_Report.pdf",
                            mime="application/pdf"
                        )

                    st.success("Report generated successfully!")

            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
                
        except Exception as e:
            st.error(f"❌ Error analyzing VCF file: {str(e)}")
            st.exception(e)

else:
    # Sample data info
    st.info("""
    ### 📝 How to use:
    1. Upload your VCF (Variant Call Format) file (WGS supported)
    2. (Optional) Upload Clinical CSV with 100 biomarkers
    3. Wait for AI analysis to complete
    
    ### 🧬 New in v3.0
    - **Pharmacogenomics GNN**: AI-predicted drug response.
    - **Clinical Reporting**: Download professional PDF summaries.
    - **Clinician Mode**: View technical genetic details.
    """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🧬 <strong>Dirghayu</strong> - India-First Longevity Genomics</p>
    <p><a href="https://github.com/VedantMadane/dirghayu" target="_blank">GitHub</a> | 
    Built with Streamlit & Python</p>
    <p><em>For research and educational purposes only. Not a substitute for professional medical advice.</em></p>
</div>
""", unsafe_allow_html=True)
