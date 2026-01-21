#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dirghayu Web UI

Interactive web interface for genomic analysis using Gradio.
Upload a VCF file and get personalized genetic insights.
"""

import sys
import io
from pathlib import Path
from typing import Dict, Tuple
import tempfile

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import gradio as gr
except ImportError:
    print("Installing Gradio...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio", "pandas"])
    import gradio as gr

import pandas as pd
from data import parse_vcf_file


def analyze_vcf(vcf_file) -> Tuple[str, str, str]:
    """
    Analyze uploaded VCF file and return results
    
    Returns:
        Tuple of (summary_html, variants_table_html, insights_html)
    """
    try:
        # Save uploaded file temporarily
        if vcf_file is None:
            return "❌ No file uploaded", "", ""
        
        # Parse VCF
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as tmp:
            # Read uploaded file content
            if hasattr(vcf_file, 'name'):
                vcf_path = Path(vcf_file.name)
            else:
                # If it's file content, write it
                tmp.write(vcf_file)
                tmp.flush()
                vcf_path = Path(tmp.name)
        
        variants_df = parse_vcf_file(vcf_path)
        
        if len(variants_df) == 0:
            return "❌ No variants found in VCF file", "", ""
        
        # Generate summary
        summary_html = f"""
        <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; color: white; margin-bottom: 20px;">
            <h2 style="margin-top: 0;">📊 Analysis Summary</h2>
            <p style="font-size: 18px; margin: 10px 0;">
                ✅ <strong>{len(variants_df)}</strong> variants parsed successfully
            </p>
            <p style="margin: 5px 0;">File: {vcf_path.name}</p>
        </div>
        """
        
        # Generate variants table
        variants_html = variants_df[['chrom', 'pos', 'rsid', 'ref', 'alt', 'genotype']].to_html(
            index=False,
            classes=['table', 'table-striped', 'table-hover'],
            border=0
        )
        
        # Add styling to table
        variants_table_html = f"""
        <style>
            .table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 14px;
            }}
            .table th {{
                background-color: #667eea;
                color: white;
                padding: 12px;
                text-align: left;
            }}
            .table td {{
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }}
            .table tr:hover {{
                background-color: #f5f5f5;
            }}
        </style>
        {variants_html}
        """
        
        # Generate genetic insights
        insights_html = generate_insights_html(variants_df)
        
        return summary_html, variants_table_html, insights_html
        
    except Exception as e:
        error_html = f"""
        <div style="padding: 20px; background-color: #fee; border-left: 4px solid #f44; 
                    border-radius: 5px; color: #c00;">
            <h3>❌ Error</h3>
            <p>{str(e)}</p>
        </div>
        """
        return error_html, "", ""


def generate_insights_html(variants_df: pd.DataFrame) -> str:
    """Generate HTML for genetic insights"""
    
    # Key variants database
    key_variants = {
        'rs1801133': {
            'gene': 'MTHFR',
            'name': 'C677T',
            'emoji': '🧬',
            'color': '#e74c3c',
            'impact': 'Folate metabolism - Higher homocysteine levels',
            'recommendation': 'Consider methylfolate supplementation (800 mcg/day)',
            'risk_level': 'HIGH'
        },
        'rs429358': {
            'gene': 'APOE',
            'name': 'ε4 allele',
            'emoji': '🧠',
            'color': '#f39c12',
            'impact': "Increased Alzheimer's disease risk (3-4x)",
            'recommendation': 'Focus on cardiovascular health, Mediterranean diet',
            'risk_level': 'MODERATE'
        },
        'rs1801131': {
            'gene': 'MTHFR',
            'name': 'A1298C',
            'emoji': '🧬',
            'color': '#3498db',
            'impact': 'Folate metabolism - Combined with C677T increases risk',
            'recommendation': 'Monitor homocysteine levels, B-vitamin supplementation',
            'risk_level': 'MODERATE'
        },
        'rs1333049': {
            'gene': 'CDKN2B-AS1',
            'name': '9p21.3 locus',
            'emoji': '❤️',
            'color': '#c0392b',
            'impact': 'Coronary artery disease risk marker',
            'recommendation': 'Regular cardiovascular screening, healthy lifestyle',
            'risk_level': 'HIGH'
        },
        'rs713598': {
            'gene': 'TAS2R38',
            'name': 'PTC taster',
            'emoji': '👅',
            'color': '#27ae60',
            'impact': 'Bitter taste perception - affects vegetable preferences',
            'recommendation': 'Ensure varied vegetable intake',
            'risk_level': 'LOW'
        },
        'rs601338': {
            'gene': 'FUT2',
            'name': 'Secretor status',
            'emoji': '💊',
            'color': '#9b59b6',
            'impact': 'Affects vitamin B12 absorption',
            'recommendation': 'Monitor B12 levels, consider supplementation',
            'risk_level': 'MODERATE'
        },
        'rs2228570': {
            'gene': 'VDR',
            'name': 'FokI',
            'emoji': '☀️',
            'color': '#f1c40f',
            'impact': 'Affects vitamin D receptor function',
            'recommendation': 'Monitor vitamin D levels, consider higher supplementation',
            'risk_level': 'MODERATE'
        }
    }
    
    insights_cards = []
    found_any = False
    
    for _, variant in variants_df.iterrows():
        rsid = variant['rsid']
        if rsid in key_variants:
            found_any = True
            info = key_variants[rsid]
            
            # Risk level badge
            risk_badges = {
                'HIGH': '<span style="background: #e74c3c; color: white; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold;">⚠️ HIGH RISK</span>',
                'MODERATE': '<span style="background: #f39c12; color: white; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold;">⚡ MODERATE</span>',
                'LOW': '<span style="background: #27ae60; color: white; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold;">✓ LOW</span>'
            }
            
            card = f"""
            <div style="background: white; border-left: 5px solid {info['color']}; 
                        padding: 20px; margin: 15px 0; border-radius: 8px; 
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h3 style="margin: 0; color: {info['color']};">
                        {info['emoji']} {rsid} - {info['name']}
                    </h3>
                    {risk_badges.get(info['risk_level'], '')}
                </div>
                
                <div style="margin: 10px 0;">
                    <strong>Gene:</strong> {info['gene']} | 
                    <strong>Genotype:</strong> {variant['genotype']} | 
                    <strong>Position:</strong> chr{variant['chrom']}:{variant['pos']}
                </div>
                
                <div style="background: #f8f9fa; padding: 12px; border-radius: 5px; margin: 10px 0;">
                    <strong>Impact:</strong><br>
                    {info['impact']}
                </div>
                
                <div style="background: #e8f5e9; padding: 12px; border-radius: 5px; margin: 10px 0;">
                    <strong>💡 Recommendation:</strong><br>
                    {info['recommendation']}
                </div>
            </div>
            """
            insights_cards.append(card)
    
    if not found_any:
        return """
        <div style="padding: 30px; text-align: center; background: #f8f9fa; 
                    border-radius: 10px; color: #666;">
            <h3>ℹ️ No High-Impact Variants Detected</h3>
            <p>Your VCF doesn't contain the common clinical variants in our database.</p>
            <p>This doesn't mean you have no genetic risks - upload a full genome VCF for comprehensive analysis.</p>
        </div>
        """
    
    header = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
        <h2 style="margin: 0;">🧬 Genetic Insights</h2>
        <p style="margin: 10px 0 0 0;">
            Found {len(insights_cards)} clinically significant variant(s)
        </p>
    </div>
    """
    
    return header + "\n".join(insights_cards)


def load_demo_vcf():
    """Load demo VCF file"""
    demo_path = Path("data/clinvar_sample.vcf")
    if demo_path.exists():
        return demo_path
    return None


# Create Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="blue",
    ),
    title="Dirghayu - Genomic Analysis",
    css="""
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            margin-bottom: 20px;
        }
    """
) as app:
    
    # Header
    gr.HTML("""
        <div class="header" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 30px; border-radius: 15px; color: white; text-align: center;">
            <h1 style="margin: 0; font-size: 42px;">🧬 Dirghayu</h1>
            <p style="font-size: 20px; margin: 10px 0 0 0;">
                India-First Longevity Genomics Platform
            </p>
            <p style="margin: 10px 0 0 0; opacity: 0.9;">
                Upload your VCF file to discover personalized health insights
            </p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # File upload
            gr.Markdown("### 📤 Upload VCF File")
            vcf_input = gr.File(
                label="Select VCF File",
                file_types=[".vcf", ".vcf.gz"],
                type="filepath"
            )
            
            analyze_btn = gr.Button(
                "🔍 Analyze Genome",
                variant="primary",
                size="lg"
            )
            
            gr.Markdown("### 🎯 Try Demo")
            demo_btn = gr.Button(
                "📊 Load Sample Data",
                variant="secondary"
            )
            
            gr.Markdown("""
            ---
            **About Dirghayu:**
            - 🇮🇳 India-first genomic analysis
            - ⚡ Fast VCF parsing (6 seconds)
            - 🎯 Actionable health insights
            - 🔬 Evidence-based recommendations
            
            **Supported Files:**
            - VCF 4.x format
            - GRCh37/GRCh38 reference
            - Whole genome or targeted panels
            
            **Privacy:**
            - All analysis runs locally
            - No data sent to external servers
            - Your genome stays private
            """)
    
        with gr.Column(scale=2):
            # Results sections
            summary_output = gr.HTML(label="Summary")
            
            with gr.Accordion("📋 Variants Table", open=False):
                variants_output = gr.HTML()
            
            insights_output = gr.HTML(label="Genetic Insights")
    
    # Event handlers
    def load_demo():
        demo_file = load_demo_vcf()
        if demo_file:
            return str(demo_file)
        return None
    
    demo_btn.click(
        fn=load_demo,
        outputs=vcf_input
    )
    
    analyze_btn.click(
        fn=analyze_vcf,
        inputs=vcf_input,
        outputs=[summary_output, variants_output, insights_output]
    )
    
    # Footer
    gr.HTML("""
        <div style="text-align: center; padding: 20px; color: #666; margin-top: 30px; 
                    border-top: 1px solid #ddd;">
            <p>
                <strong>Dirghayu v0.1.0</strong> | 
                India-First Longevity Genomics | 
                Open Source (Research Use)
            </p>
            <p style="font-size: 12px; color: #999;">
                ⚠️ For research and educational purposes only. 
                Not for clinical diagnosis or treatment decisions.
                Consult a healthcare provider before acting on genetic results.
            </p>
        </div>
    """)


if __name__ == "__main__":
    print("=" * 80)
    print("🧬 Starting Dirghayu Web Interface")
    print("=" * 80)
    print("\n✅ Server will open automatically in your browser")
    print("📍 Manual access: http://localhost:7860")
    print("🛑 Press Ctrl+C to stop the server\n")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create public link
        show_error=True
    )
