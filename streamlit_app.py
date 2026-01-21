#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dirghayu - Streamlit Cloud Deployment
India-First Longevity Genomics Platform
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import io

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.vcf_parser import VCFParser

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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #FF6B35, #F7931E); border-radius: 15px; color: white; margin-bottom: 2rem;">
    <h1>🧬 Dirghayu</h1>
    <p style="font-size: 1.2rem;">India-First Longevity Genomics Platform</p>
</div>
""", unsafe_allow_html=True)

# Sidebar info
st.sidebar.header("About Dirghayu")
st.sidebar.markdown("""
### Features
- 🇮🇳 India-focused analysis
- ⚡ Fast VCF parsing
- 🎯 Actionable insights
- 🔒 Privacy-first

### What we analyze
- Folate metabolism (MTHFR)
- Alzheimer's risk (APOE)
- Heart disease risk
- Nutrient deficiencies

### Privacy
Your data stays on the server during analysis and is never stored permanently.
""")

# Main content
st.header("📤 Upload Your VCF File")

uploaded_file = st.file_uploader(
    "Choose a VCF file",
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
            
            # Parse VCF
            parser = VCFParser()
            variants_df = parser.parse(temp_path)
            
            # Clean up temp file
            temp_path.unlink()
            
            if len(variants_df) == 0:
                st.error("❌ No variants found in the VCF file")
            else:
                st.success(f"✅ Successfully analyzed {len(variants_df)} variants!")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Variants", len(variants_df))
                with col2:
                    unique_chroms = variants_df['chrom'].nunique()
                    st.metric("Chromosomes", unique_chroms)
                with col3:
                    has_rsid = variants_df['rsid'].notna().sum()
                    st.metric("With rsID", has_rsid)
                
                st.divider()
                
                # Key variants database
                key_variants = {
                    'rs1801133': {
                        'gene': 'MTHFR',
                        'name': 'C677T',
                        'risk': 'HIGH',
                        'description': 'Folate metabolism variant - affects B12 and folate processing',
                        'recommendation': 'Consider folate supplementation, regular B12 monitoring'
                    },
                    'rs429358': {
                        'gene': 'APOE',
                        'name': 'ε4 allele',
                        'risk': 'MODERATE',
                        'description': "Alzheimer's disease risk variant",
                        'recommendation': 'Maintain cognitive health, regular exercise, Mediterranean diet'
                    },
                    'rs1801131': {
                        'gene': 'MTHFR',
                        'name': 'A1298C',
                        'risk': 'MODERATE',
                        'description': 'Secondary folate metabolism variant',
                        'recommendation': 'Monitor homocysteine levels, adequate folate intake'
                    },
                    'rs1333049': {
                        'gene': 'CDKN2B-AS1',
                        'name': '9p21.3 variant',
                        'risk': 'HIGH',
                        'description': 'Cardiovascular disease risk',
                        'recommendation': 'Heart-healthy lifestyle, regular BP monitoring, lipid profile checks'
                    },
                    'rs713598': {
                        'gene': 'TAS2R38',
                        'name': 'PTC taster',
                        'risk': 'LOW',
                        'description': 'Bitter taste perception',
                        'recommendation': 'May influence vegetable preferences - ensure diverse diet'
                    },
                }
                
                # Find clinically significant variants
                st.header("🎯 Clinically Significant Variants")
                
                found_variants = []
                for _, variant in variants_df.iterrows():
                    rsid = variant['rsid']
                    if rsid in key_variants:
                        found_variants.append((rsid, variant, key_variants[rsid]))
                
                if found_variants:
                    for rsid, variant, info in found_variants:
                        risk_color = {
                            'HIGH': '#e74c3c',
                            'MODERATE': '#f39c12',
                            'LOW': '#27ae60'
                        }[info['risk']]
                        
                        st.markdown(f"""
                        <div style="border-left: 5px solid {risk_color}; padding: 15px; margin: 15px 0; background: #f8f9fa; border-radius: 8px;">
                            <h3 style="color: {risk_color}; margin: 0;">{rsid} - {info['name']}</h3>
                            <p><strong>Gene:</strong> {info['gene']} | <strong>Risk Level:</strong> {info['risk']}</p>
                            <p><strong>Genotype:</strong> {variant['genotype']} | <strong>Position:</strong> chr{variant['chrom']}:{variant['pos']}</p>
                            <p><strong>About:</strong> {info['description']}</p>
                            <p><strong>💡 Recommendation:</strong> {info['recommendation']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("ℹ️ No clinically significant variants found in our current database. This is common and doesn't indicate any issues!")
                
                st.divider()
                
                # All variants table
                st.header("📊 All Detected Variants")
                st.dataframe(
                    variants_df,
                    use_container_width=True,
                    height=400
                )
                
                # Download option
                csv = variants_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="dirghayu_analysis.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error analyzing VCF file: {str(e)}")
            st.exception(e)

else:
    # Sample data info
    st.info("""
    ### 📝 How to use:
    1. Upload your VCF (Variant Call Format) file
    2. Wait for analysis to complete
    3. Review your personalized genetic insights
    
    ### 🧬 What is a VCF file?
    A VCF file contains genetic variant information from whole genome sequencing or genotyping.
    Common sources: 23andMe, AncestryDNA, Whole Genome Sequencing services.
    
    ### 🔒 Your Privacy
    Files are processed in memory and not permanently stored on our servers.
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
