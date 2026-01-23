#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dirghayu - HuggingFace Spaces Deployment
Standalone version with built-in VCF parser
"""

import gradio as gr
import pandas as pd
from pathlib import Path


def parse_vcf_file(vcf_path):
    """Parse VCF file and return DataFrame"""
    variants = []
    
    try:
        with open(vcf_path, 'r') as f:
            for line in f:
                # Skip header lines
                if line.startswith('#'):
                    continue
                    
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                
                # Extract basic info
                chrom = parts[0].replace('chr', '')
                pos = parts[1]
                rsid = parts[2] if parts[2] != '.' else None
                ref = parts[3]
                alt = parts[4]
                
                # Extract genotype if available
                genotype = '0/1'  # default
                if len(parts) > 9:
                    gt_field = parts[9].split(':')[0]
                    genotype = gt_field
                
                variants.append({
                    'chrom': chrom,
                    'pos': int(pos),
                    'rsid': rsid,
                    'ref': ref,
                    'alt': alt,
                    'genotype': genotype
                })
        
        return pd.DataFrame(variants)
    
    except Exception as e:
        print(f"Error parsing VCF: {e}")
        return pd.DataFrame()


def analyze_vcf(vcf_file):
    """Analyze VCF and return HTML report"""
    if vcf_file is None:
        return "<h3>❌ No file uploaded</h3>"
    
    try:
        # Parse VCF
        vcf_path = Path(vcf_file.name)
        variants_df = parse_vcf_file(vcf_path)
        
        if len(variants_df) == 0:
            return "<h3>❌ No variants found in VCF file</h3><p>Please ensure your VCF file is properly formatted.</p>"
        
        # Key variants database
        key_variants = {
            'rs1801133': {
                'gene': 'MTHFR',
                'name': 'C677T',
                'risk': 'HIGH',
                'emoji': '🧬',
                'description': 'Folate metabolism - Higher homocysteine levels',
                'recommendation': 'Consider methylfolate supplementation (800 mcg/day)'
            },
            'rs429358': {
                'gene': 'APOE',
                'name': 'ε4 allele',
                'risk': 'MODERATE',
                'emoji': '🧠',
                'description': "Increased Alzheimer's disease risk (3-4x)",
                'recommendation': 'Focus on cardiovascular health, Mediterranean diet'
            },
            'rs1801131': {
                'gene': 'MTHFR',
                'name': 'A1298C',
                'risk': 'MODERATE',
                'emoji': '🧬',
                'description': 'Folate metabolism - Combined with C677T increases risk',
                'recommendation': 'Monitor homocysteine levels, B-vitamin supplementation'
            },
            'rs1333049': {
                'gene': 'CDKN2B-AS1',
                'name': '9p21.3 locus',
                'risk': 'HIGH',
                'emoji': '❤️',
                'description': 'Coronary artery disease risk marker',
                'recommendation': 'Regular cardiovascular screening, healthy lifestyle'
            },
            'rs713598': {
                'gene': 'TAS2R38',
                'name': 'PTC taster',
                'risk': 'LOW',
                'emoji': '👅',
                'description': 'Bitter taste perception - affects vegetable preferences',
                'recommendation': 'Ensure varied vegetable intake'
            },
        }
        
        # Generate report
        html = f"""
        <div style="font-family: 'Segoe UI', Tahoma, sans-serif; max-width: 900px; margin: 0 auto;">
            <div style="background: linear-gradient(135deg, #FF6B35, #F7931E); padding: 30px; border-radius: 15px; color: white; text-align: center; margin-bottom: 20px;">
                <h1 style="margin: 0; font-size: 2.5em;">🧬 Dirghayu Analysis</h1>
                <p style="font-size: 1.2em; margin: 10px 0 0 0;">India-First Longevity Genomics</p>
            </div>
            
            <div style="background: white; padding: 25px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #ddd; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <h2 style="color: #FF6B35; margin-top: 0;">📊 Analysis Summary</h2>
                <p style="font-size: 1.1em;"><strong>{len(variants_df)}</strong> variants analyzed from your VCF file</p>
        """
        
        # Find key variants
        found_variants = []
        for _, var in variants_df.iterrows():
            rsid = var['rsid']
            if rsid in key_variants:
                found_variants.append((rsid, var, key_variants[rsid]))
        
        if found_variants:
            html += f"<p style='font-size: 1.1em;'><strong>{len(found_variants)}</strong> clinically significant variants found</p>"
            html += "</div>"
            
            html += "<h2 style='color: #FF6B35;'>🎯 Clinically Significant Variants</h2>"
            
            for rsid, var, info in found_variants:
                color = {'HIGH': '#e74c3c', 'MODERATE': '#f39c12', 'LOW': '#27ae60'}[info['risk']]
                
                html += f"""
                <div style="background: white; border-left: 5px solid {color}; padding: 25px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h3 style="color: {color}; margin: 0 0 10px 0; font-size: 1.5em;">
                        {info['emoji']} {rsid} - {info['name']}
                    </h3>
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0;">
                        <p style="margin: 5px 0;"><strong>Gene:</strong> {info['gene']}</p>
                        <p style="margin: 5px 0;"><strong>Genotype:</strong> <code style="background: #e9ecef; padding: 2px 6px; border-radius: 3px;">{var['genotype']}</code></p>
                        <p style="margin: 5px 0;"><strong>Position:</strong> chr{var['chrom']}:{var['pos']}</p>
                        <p style="margin: 5px 0;"><strong>Risk Level:</strong> <span style="color: {color}; font-weight: bold;">{info['risk']}</span></p>
                    </div>
                    <p style="margin: 15px 0;"><strong>Impact:</strong> {info['description']}</p>
                    <div style="background: #e8f5e9; padding: 15px; border-left: 3px solid #4caf50; border-radius: 5px; margin-top: 15px;">
                        <p style="margin: 0;"><strong>💡 Recommendation:</strong> {info['recommendation']}</p>
                    </div>
                </div>
                """
        else:
            html += "<p style='color: #666; font-size: 1.1em;'>No clinically significant variants found in our current database.</p>"
            html += "</div>"
            html += """
            <div style="background: #e3f2fd; padding: 20px; border-radius: 8px; border-left: 4px solid #2196f3; margin: 20px 0;">
                <p style="margin: 0;"><strong>ℹ️ Note:</strong> This is common and doesn't indicate any issues. Our database focuses on high-impact variants relevant to Indian population health.</p>
            </div>
            """
        
        # Disclaimer
        html += """
        <div style="background: #fff3cd; padding: 20px; border-radius: 8px; border-left: 4px solid #ffc107; margin: 30px 0;">
            <h3 style="margin: 0 0 10px 0; color: #856404;">⚠️ Important Disclaimer</h3>
            <p style="margin: 5px 0; color: #856404;"><strong>This report is for research and educational purposes only.</strong></p>
            <ul style="color: #856404; margin: 10px 0;">
                <li>NOT for clinical diagnosis or treatment decisions</li>
                <li>Consult a healthcare provider before acting on genetic results</li>
                <li>Genetic risk ≠ disease certainty</li>
                <li>Lifestyle and environment are critical factors</li>
            </ul>
        </div>
        </div>
        """
        
        return html
        
    except Exception as e:
        return f"""
        <div style="background: #f8d7da; padding: 20px; border-radius: 8px; border-left: 4px solid #dc3545;">
            <h3 style="color: #721c24; margin: 0 0 10px 0;">❌ Error Processing VCF File</h3>
            <p style="color: #721c24; margin: 0;"><strong>Error:</strong> {str(e)}</p>
            <p style="color: #721c24; margin: 10px 0 0 0;">Please ensure your file is a valid VCF format.</p>
        </div>
        """


# Create Gradio interface
with gr.Blocks(
    title="Dirghayu - India-First Genomic Analysis",
    theme=gr.themes.Soft(primary_hue="orange")
) as app:
    
    gr.HTML("""
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #FF6B35, #F7931E); border-radius: 15px; color: white; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 3em;">🧬 Dirghayu</h1>
            <p style="font-size: 1.3em; margin: 10px 0 0 0;">India-First Longevity Genomics Platform</p>
            <p style="font-size: 1em; margin: 5px 0 0 0; opacity: 0.9;">Upload your VCF file for personalized genetic insights</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            vcf_input = gr.File(
                label="📤 Upload VCF File",
                file_types=[".vcf"],
                type="filepath"
            )
        with gr.Column(scale=1):
            analyze_btn = gr.Button(
                "🔍 Analyze Genome",
                variant="primary",
                size="lg"
            )
    
    output = gr.HTML(label="Analysis Results")
    
    analyze_btn.click(fn=analyze_vcf, inputs=vcf_input, outputs=output)
    
    gr.Markdown("""
    ### 🌟 About Dirghayu
    
    - 🇮🇳 **India-focused** genomic analysis with population-specific insights
    - ⚡ **Fast VCF parsing** - results in seconds
    - 🎯 **Actionable health insights** based on latest research
    - 🔒 **Privacy-first** - your data is processed in memory and never stored
    
    ### 🧬 What We Analyze
    
    - **Folate metabolism** (MTHFR variants) - critical for Indian populations
    - **Alzheimer's risk** (APOE genotypes)
    - **Cardiovascular disease** risk markers
    - **Nutrient metabolism** and deficiencies
    - **Taste perception** and dietary preferences
    
    ### 📖 How to Use
    
    1. Upload your VCF file (from 23andMe, AncestryDNA, or whole genome sequencing)
    2. Click "Analyze Genome"
    3. Review your personalized genetic insights
    4. Consult with a healthcare provider for clinical decisions
    
    ---
    
    **Version:** 0.1.0 | **Source Code:** [GitHub](https://github.com/VedantMadane/dirghayu)
    """)

if __name__ == "__main__":
    app.launch()
