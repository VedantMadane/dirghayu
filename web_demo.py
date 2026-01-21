#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dirghayu Web Demo - Simple HTML UI

Generates an interactive HTML report with genetic insights.
No external dependencies required - pure Python + HTML.
"""

import sys
import io
from pathlib import Path
import webbrowser
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import parse_vcf_file


def generate_html_report(vcf_path: Path) -> str:
    """Generate complete HTML report"""
    
    # Parse VCF
    variants_df = parse_vcf_file(vcf_path)
    
    # Key variants database
    key_variants = {
        'rs1801133': {
            'gene': 'MTHFR',
            'name': 'C677T',
            'emoji': '🧬',
            'color': '#e74c3c',
            'impact': 'Folate metabolism - Higher homocysteine levels',
            'recommendation': 'Consider methylfolate supplementation (800 mcg/day)',
            'risk_level': 'HIGH',
            'details': '''
                <strong>What this means:</strong><br>
                - Reduced MTHFR enzyme activity (~30% of normal)<br>
                - Higher homocysteine levels<br>
                - Increased cardiovascular disease risk<br>
                - Higher frequency in Indian populations (~25-35%)<br><br>
                <strong>Action items:</strong><br>
                ✅ Methylfolate supplementation (800 mcg/day)<br>
                ✅ Monitor homocysteine levels every 6 months<br>
                ⚠️ Avoid folic acid; use methylfolate form
            '''
        },
        'rs429358': {
            'gene': 'APOE',
            'name': 'ε4 allele',
            'emoji': '🧠',
            'color': '#f39c12',
            'impact': "Increased Alzheimer's disease risk (3-4x)",
            'recommendation': 'Focus on cardiovascular health, Mediterranean diet',
            'risk_level': 'MODERATE',
            'details': '''
                <strong>What this means:</strong><br>
                - 3-4x increased Alzheimer's disease risk<br>
                - Earlier onset of cognitive decline<br>
                - Cardiovascular disease association<br><br>
                <strong>Action items:</strong><br>
                ✅ Mediterranean diet rich in omega-3<br>
                ✅ Regular cognitive assessments after age 50<br>
                ✅ Control blood pressure and cholesterol
            '''
        },
        'rs1801131': {
            'gene': 'MTHFR',
            'name': 'A1298C',
            'emoji': '🧬',
            'color': '#3498db',
            'impact': 'Folate metabolism - Combined with C677T increases risk',
            'recommendation': 'Monitor homocysteine levels, B-vitamin supplementation',
            'risk_level': 'MODERATE',
            'details': '''
                <strong>What this means:</strong><br>
                - Compound heterozygote with C677T increases risk<br>
                - Combined effect on folate metabolism<br>
                - May affect neurotransmitter synthesis<br><br>
                <strong>Action items:</strong><br>
                ✅ B-vitamin supplementation<br>
                ✅ Monitor homocysteine (especially with C677T)
            '''
        },
        'rs1333049': {
            'gene': 'CDKN2B-AS1',
            'name': '9p21.3 locus',
            'emoji': '❤️',
            'color': '#c0392b',
            'impact': 'Coronary artery disease risk marker',
            'recommendation': 'Regular cardiovascular screening, healthy lifestyle',
            'risk_level': 'HIGH',
            'details': '''
                <strong>What this means:</strong><br>
                - Strongest genetic marker for CAD<br>
                - 1.5-2x increased risk of heart disease<br>
                - Higher prevalence in South Asian populations<br><br>
                <strong>Action items:</strong><br>
                ✅ Regular cardiovascular screening (start age 30-35)<br>
                ✅ Lipid panel, ECG, stress test as recommended<br>
                ⚠️ Consider early statin therapy if other risks present
            '''
        },
        'rs713598': {
            'gene': 'TAS2R38',
            'name': 'PTC taster',
            'emoji': '👅',
            'color': '#27ae60',
            'impact': 'Bitter taste perception - affects vegetable preferences',
            'recommendation': 'Ensure varied vegetable intake',
            'risk_level': 'LOW',
            'details': '''
                <strong>What this means:</strong><br>
                - Affects taste perception of bitter compounds<br>
                - May influence dietary choices<br>
                - Common genetic variant<br><br>
                <strong>Action items:</strong><br>
                ℹ️ Awareness of taste-driven dietary limitations<br>
                ✅ Try different preparation methods for vegetables
            '''
        },
    }
    
    # Build variants cards
    variant_cards = []
    risk_summary = {'HIGH': 0, 'MODERATE': 0, 'LOW': 0}
    
    for _, variant in variants_df.iterrows():
        rsid = variant['rsid']
        if rsid in key_variants:
            info = key_variants[rsid]
            risk_summary[info['risk_level']] += 1
            
            card = f'''
            <div class="variant-card" style="border-left: 5px solid {info['color']};">
                <div class="variant-header">
                    <h3 style="color: {info['color']}; margin: 0;">
                        <span style="font-size: 32px;">{info['emoji']}</span> {rsid} - {info['name']}
                    </h3>
                    <span class="badge badge-{info['risk_level'].lower()}">{info['risk_level']}</span>
                </div>
                
                <div class="variant-meta">
                    <strong>Gene:</strong> {info['gene']} | 
                    <strong>Genotype:</strong> <code>{variant['genotype']}</code> | 
                    <strong>Position:</strong> chr{variant['chrom']}:{variant['pos']}
                </div>
                
                <div class="impact-box">
                    <strong>Impact:</strong><br>
                    {info['impact']}
                </div>
                
                <div class="recommendation-box">
                    <strong>💡 Recommendation:</strong><br>
                    {info['recommendation']}
                </div>
                
                <details class="details-box">
                    <summary>📖 Learn More</summary>
                    <div style="padding: 10px;">
                        {info['details']}
                    </div>
                </details>
            </div>
            '''
            variant_cards.append(card)
    
    # Build HTML
    html = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dirghayu - Genomic Analysis Report</title>
        <style>
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                line-height: 1.6;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            
            .header {{
                background: white;
                padding: 40px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                margin-bottom: 30px;
            }}
            
            .header h1 {{
                font-size: 48px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 10px;
            }}
            
            .header p {{
                font-size: 18px;
                color: #666;
            }}
            
            .summary-box {{
                background: white;
                padding: 30px;
                border-radius: 15px;
                margin-bottom: 30px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            }}
            
            .summary-stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            
            .stat-card {{
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }}
            
            .stat-card.high {{
                background: #fee;
                border: 2px solid #e74c3c;
            }}
            
            .stat-card.moderate {{
                background: #fff3cd;
                border: 2px solid #f39c12;
            }}
            
            .stat-card.low {{
                background: #d4edda;
                border: 2px solid #27ae60;
            }}
            
            .stat-number {{
                font-size: 48px;
                font-weight: bold;
                margin: 10px 0;
            }}
            
            .variant-card {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 3px 15px rgba(0,0,0,0.1);
                transition: transform 0.3s, box-shadow 0.3s;
            }}
            
            .variant-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            }}
            
            .variant-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }}
            
            .variant-meta {{
                background: #f8f9fa;
                padding: 12px;
                border-radius: 8px;
                margin: 15px 0;
                font-size: 14px;
            }}
            
            .impact-box {{
                background: #fff8e1;
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
                border-left: 4px solid #ffc107;
            }}
            
            .recommendation-box {{
                background: #e8f5e9;
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
                border-left: 4px solid #4caf50;
            }}
            
            .details-box {{
                background: #f5f5f5;
                padding: 15px;
                border-radius: 8px;
                margin-top: 15px;
                cursor: pointer;
            }}
            
            .details-box summary {{
                font-weight: bold;
                user-select: none;
            }}
            
            .badge {{
                padding: 6px 16px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: bold;
                text-transform: uppercase;
            }}
            
            .badge-high {{
                background: #e74c3c;
                color: white;
            }}
            
            .badge-moderate {{
                background: #f39c12;
                color: white;
            }}
            
            .badge-low {{
                background: #27ae60;
                color: white;
            }}
            
            code {{
                background: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            
            .footer {{
                background: white;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                margin-top: 30px;
                color: #666;
            }}
            
            .disclaimer {{
                background: #fff3cd;
                border: 2px solid #ffc107;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }}
            
            @media print {{
                body {{
                    background: white;
                    padding: 0;
                }}
                
                .variant-card {{
                    page-break-inside: avoid;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧬 Dirghayu</h1>
                <p>India-First Longevity Genomics Platform</p>
                <p style="margin-top: 10px; font-size: 14px; color: #999;">
                    Report generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
                </p>
            </div>
            
            <div class="summary-box">
                <h2>📊 Analysis Summary</h2>
                <p><strong>VCF File:</strong> {vcf_path.name}</p>
                <p><strong>Variants Analyzed:</strong> {len(variants_df)} variants parsed</p>
                <p><strong>Clinically Significant Variants Found:</strong> {len(variant_cards)}</p>
                
                <div class="summary-stats">
                    <div class="stat-card high">
                        <div>⚠️ HIGH RISK</div>
                        <div class="stat-number">{risk_summary['HIGH']}</div>
                        <div>variant(s)</div>
                    </div>
                    <div class="stat-card moderate">
                        <div>⚡ MODERATE RISK</div>
                        <div class="stat-number">{risk_summary['MODERATE']}</div>
                        <div>variant(s)</div>
                    </div>
                    <div class="stat-card low">
                        <div>✓ LOW RISK</div>
                        <div class="stat-number">{risk_summary['LOW']}</div>
                        <div>variant(s)</div>
                    </div>
                </div>
            </div>
            
            <div class="summary-box">
                <h2>🧬 Genetic Insights</h2>
                <p style="margin-top: 10px; color: #666;">
                    Below are the clinically significant variants detected in your genome. 
                    Each variant includes its impact, risk level, and personalized recommendations.
                </p>
            </div>
            
            {''.join(variant_cards) if variant_cards else '<div class="summary-box"><p>No clinically significant variants found in the analyzed regions.</p></div>'}
            
            <div class="disclaimer">
                <h3 style="margin-bottom: 10px;">⚠️ Important Disclaimer</h3>
                <p><strong>This report is for research and educational purposes only.</strong></p>
                <ul style="text-align: left; margin: 10px 0 10px 30px;">
                    <li>NOT for clinical diagnosis or treatment decisions</li>
                    <li>Consult a healthcare provider before acting on genetic results</li>
                    <li>Genetic risk ≠ disease certainty</li>
                    <li>Lifestyle and environment are also critical factors</li>
                </ul>
                <p><strong>For clinical use:</strong> Validate with a certified genetic counselor and CLIA-certified lab testing.</p>
            </div>
            
            <div class="footer">
                <p><strong>Dirghayu v0.1.0</strong></p>
                <p>India-First Longevity Genomics Platform</p>
                <p style="margin-top: 10px; font-size: 12px;">
                    Open Source (Research Use) | Built with ❤️ for Indian population health
                </p>
            </div>
        </div>
    </body>
    </html>
    '''
    
    return html


def main():
    """Main function"""
    if len(sys.argv) > 1:
        vcf_path = Path(sys.argv[1])
    else:
        vcf_path = Path("data/clinvar_sample.vcf")
    
    if not vcf_path.exists():
        print(f"❌ Error: VCF file not found: {vcf_path}")
        print("\nUsage:")
        print("  python web_demo.py <path_to_vcf_file>")
        print("\nOr use demo data:")
        print("  python web_demo.py data/clinvar_sample.vcf")
        return
    
    print("=" * 80)
    print("🧬 Dirghayu Web Report Generator")
    print("=" * 80)
    print(f"\n📄 Processing: {vcf_path}")
    
    # Generate report
    html = generate_html_report(vcf_path)
    
    # Save report
    output_path = Path("dirghayu_report.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Report generated: {output_path.absolute()}")
    print("\n🌐 Opening in browser...")
    
    # Open in browser
    webbrowser.open(f'file://{output_path.absolute()}')
    
    print("\n" + "=" * 80)
    print("✅ Done! Your genomic analysis report is now open in your browser.")
    print("=" * 80)


if __name__ == "__main__":
    main()
