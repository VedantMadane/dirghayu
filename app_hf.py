#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dirghayu - HuggingFace Spaces Deployment
Simplified Gradio app for genomic analysis
"""

import gradio as gr
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from data import parse_vcf_file
except ImportError:
    print("Installing dependencies...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    from data import parse_vcf_file


def analyze_vcf(vcf_file):
    """Analyze VCF and return HTML report"""
    if vcf_file is None:
        return "<h3>❌ No file uploaded</h3>"
    
    try:
        # Parse VCF
        vcf_path = Path(vcf_file.name)
        variants_df = parse_vcf_file(vcf_path)
        
        if len(variants_df) == 0:
            return "<h3>❌ No variants found</h3>"
        
        # Key variants database
        key_variants = {
            'rs1801133': {'gene': 'MTHFR', 'name': 'C677T', 'risk': 'HIGH', 'emoji': '🧬'},
            'rs429358': {'gene': 'APOE', 'name': 'ε4', 'risk': 'MODERATE', 'emoji': '🧠'},
            'rs1801131': {'gene': 'MTHFR', 'name': 'A1298C', 'risk': 'MODERATE', 'emoji': '🧬'},
            'rs1333049': {'gene': 'CDKN2B-AS1', 'name': '9p21.3', 'risk': 'HIGH', 'emoji': '❤️'},
            'rs713598': {'gene': 'TAS2R38', 'name': 'PTC', 'risk': 'LOW', 'emoji': '👅'},
        }
        
        # Generate report
        html = f"""
        <div style="font-family: 'Segoe UI', sans-serif;">
            <div style="background: linear-gradient(135deg, #FF6B35, #F7931E); padding: 30px; border-radius: 15px; color: white; text-align: center; margin-bottom: 20px;">
                <h1 style="margin: 0;">🧬 Dirghayu Analysis</h1>
                <p>India-First Longevity Genomics</p>
            </div>
            
            <div style="background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #ddd;">
                <h2>📊 Summary</h2>
                <p><strong>{len(variants_df)}</strong> variants analyzed</p>
            </div>
        """
        
        # Find key variants
        found = False
        for _, var in variants_df.iterrows():
            rsid = var['rsid']
            if rsid in key_variants:
                found = True
                info = key_variants[rsid]
                color = {'HIGH': '#e74c3c', 'MODERATE': '#f39c12', 'LOW': '#27ae60'}[info['risk']]
                
                html += f"""
                <div style="background: white; border-left: 5px solid {color}; padding: 20px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h3 style="color: {color}; margin: 0;">
                        {info['emoji']} {rsid} - {info['name']}
                    </h3>
                    <p><strong>Gene:</strong> {info['gene']} | <strong>Genotype:</strong> {var['genotype']} | <strong>Risk:</strong> {info['risk']}</p>
                    <p><strong>Position:</strong> chr{var['chrom']}:{var['pos']}</p>
                </div>
                """
        
        if not found:
            html += "<p>No clinically significant variants found in database.</p>"
        
        html += "</div>"
        return html
        
    except Exception as e:
        return f"<h3>❌ Error: {str(e)}</h3>"


# Create Gradio interface
with gr.Blocks(title="Dirghayu - Genomic Analysis") as app:
    gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #FF6B35, #F7931E); border-radius: 10px; color: white;">
            <h1>🧬 Dirghayu</h1>
            <p>India-First Longevity Genomics Platform</p>
        </div>
    """)
    
    with gr.Row():
        vcf_input = gr.File(label="Upload VCF File", file_types=[".vcf"])
        analyze_btn = gr.Button("🔍 Analyze", variant="primary")
    
    output = gr.HTML(label="Results")
    
    analyze_btn.click(fn=analyze_vcf, inputs=vcf_input, outputs=output)
    
    gr.Markdown("""
    ### About
    - 🇮🇳 India-focused genomic analysis
    - ⚡ Fast VCF parsing
    - 🎯 Actionable health insights
    
    ### Privacy
    - All analysis runs on this server
    - Your data is not stored
    """)

if __name__ == "__main__":
    app.launch()
