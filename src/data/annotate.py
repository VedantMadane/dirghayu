"""
Variant Annotation Pipeline

Enriches variants with:
1. Population frequencies (gnomAD, GenomeIndia)
2. Pathogenicity predictions (AlphaMissense, CADD)
3. Gene/transcript information
4. Functional consequences
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
from pathlib import Path
import requests
import time
import pandas as pd
from functools import lru_cache


@dataclass
class VariantAnnotation:
    """Enriched variant annotation"""
    # Basic info
    variant_id: str
    chrom: str
    pos: int
    ref: str
    alt: str
    
    # Gene/transcript
    gene_symbol: Optional[str] = None
    gene_id: Optional[str] = None
    transcript_id: Optional[str] = None
    
    # Functional consequence
    consequence: Optional[str] = None  # missense, synonymous, etc.
    protein_change: Optional[str] = None  # p.Ala222Val
    
    # Population frequencies
    gnomad_af: Optional[float] = None  # Global
    gnomad_af_south_asian: Optional[float] = None
    genome_india_af: Optional[float] = None
    
    # Pathogenicity scores
    alphamissense_score: Optional[float] = None
    alphamissense_class: Optional[str] = None  # benign, ambiguous, pathogenic
    cadd_score: Optional[float] = None
    
    # Protein structure
    uniprot_id: Optional[str] = None
    alphafold_confident: Optional[bool] = None


class VariantAnnotator:
    """Annotate variants using public APIs and databases"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.last_api_call = 0
        self.min_interval = 0.2  # 200ms between API calls
    
    def _rate_limit(self):
        """Simple rate limiting"""
        elapsed = time.time() - self.last_api_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_api_call = time.time()
    
    @lru_cache(maxsize=10000)
    def annotate_variant(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str
    ) -> VariantAnnotation:
        """
        Annotate a single variant using multiple sources
        
        Args:
            chrom: Chromosome (e.g., "1", "chr1")
            pos: Position
            ref: Reference allele
            alt: Alternate allele
        
        Returns:
            VariantAnnotation with enriched data
        """
        # Normalize chromosome
        chrom = chrom.replace("chr", "")
        variant_id = f"{chrom}:{pos}:{ref}:{alt}"
        
        annotation = VariantAnnotation(
            variant_id=variant_id,
            chrom=chrom,
            pos=pos,
            ref=ref,
            alt=alt
        )
        
        # Fetch from various sources
        self._annotate_with_ensembl(annotation)
        self._annotate_with_gnomad(annotation)
        # AlphaMissense and CADD require local databases (too large for API)
        
        return annotation
    
    def _annotate_with_ensembl(self, annotation: VariantAnnotation):
        """
        Use Ensembl VEP REST API for gene/consequence annotation
        https://rest.ensembl.org/
        """
        self._rate_limit()
        
        # Format for VEP API
        region = f"{annotation.chrom}:{annotation.pos}-{annotation.pos}"
        alleles = f"{annotation.ref}/{annotation.alt}"
        
        url = f"https://rest.ensembl.org/vep/human/region/{region}/{alleles}"
        
        try:
            response = requests.get(
                url,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data:
                    # Take most severe consequence
                    result = data[0]
                    
                    # Extract transcript consequences
                    if 'transcript_consequences' in result and result['transcript_consequences']:
                        tc = result['transcript_consequences'][0]  # Most severe
                        
                        annotation.gene_symbol = tc.get('gene_symbol')
                        annotation.gene_id = tc.get('gene_id')
                        annotation.transcript_id = tc.get('transcript_id')
                        annotation.consequence = ','.join(tc.get('consequence_terms', []))
                        annotation.protein_change = tc.get('protein_start')
                        
                        # UniProt ID
                        if 'swissprot' in tc:
                            annotation.uniprot_id = tc['swissprot'][0] if tc['swissprot'] else None
        
        except Exception as e:
            print(f"⚠ Ensembl API error for {annotation.variant_id}: {e}")
    
    def _annotate_with_gnomad(self, annotation: VariantAnnotation):
        """
        Fetch gnomAD population frequencies
        Note: gnomAD API has rate limits, consider local database for production
        """
        self._rate_limit()
        
        # gnomAD GraphQL API
        query = """
        query VariantQuery($variantId: String!) {
          variant(variantId: $variantId, dataset: gnomad_r4) {
            variant_id
            genome {
              ac
              an
              af
              populations {
                id
                ac
                an
                af
              }
            }
          }
        }
        """
        
        # Format variant ID for gnomAD: "1-55051215-G-A"
        gnomad_id = f"{annotation.chrom}-{annotation.pos}-{annotation.ref}-{annotation.alt}"
        
        try:
            response = requests.post(
                "https://gnomad.broadinstitute.org/api",
                json={
                    "query": query,
                    "variables": {"variantId": gnomad_id}
                },
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and data['data']['variant']:
                    genome = data['data']['variant'].get('genome', {})
                    
                    # Global allele frequency
                    annotation.gnomad_af = genome.get('af')
                    
                    # South Asian frequency
                    populations = genome.get('populations', [])
                    for pop in populations:
                        if pop['id'] == 'sas':  # South Asian
                            annotation.gnomad_af_south_asian = pop.get('af')
        
        except Exception as e:
            print(f"⚠ gnomAD API error for {annotation.variant_id}: {e}")
    
    def annotate_dataframe(self, variants_df: pd.DataFrame) -> pd.DataFrame:
        """
        Annotate a DataFrame of variants
        
        Args:
            variants_df: DataFrame with columns: chrom, pos, ref, alt
        
        Returns:
            DataFrame with annotation columns added
        """
        print(f"Annotating {len(variants_df)} variants...")
        
        annotations = []
        
        for idx, row in variants_df.iterrows():
            if idx % 10 == 0:
                print(f"  Progress: {idx}/{len(variants_df)}")
            
            ann = self.annotate_variant(
                chrom=str(row['chrom']),
                pos=int(row['pos']),
                ref=str(row['ref']),
                alt=str(row['alt'])
            )
            
            annotations.append({
                'gene_symbol': ann.gene_symbol,
                'gene_id': ann.gene_id,
                'transcript_id': ann.transcript_id,
                'consequence': ann.consequence,
                'protein_change': ann.protein_change,
                'gnomad_af': ann.gnomad_af,
                'gnomad_af_south_asian': ann.gnomad_af_south_asian,
                'genome_india_af': ann.genome_india_af,
                'alphamissense_score': ann.alphamissense_score,
                'cadd_score': ann.cadd_score,
                'uniprot_id': ann.uniprot_id
            })
        
        # Merge with original DataFrame
        ann_df = pd.DataFrame(annotations)
        result = pd.concat([variants_df.reset_index(drop=True), ann_df], axis=1)
        
        print(f"✓ Annotation complete!")
        return result


# Local AlphaMissense database (requires download)
class AlphaMissenseDB:
    """
    Local AlphaMissense database for pathogenicity scores
    Requires downloading AlphaMissense_hg38.tsv.gz (~900MB)
    """
    
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._index = None
    
    def load_index(self):
        """Load AlphaMissense database into memory (indexed by variant)"""
        import gzip
        
        if not self.db_path.exists():
            print(f"⚠ AlphaMissense DB not found at {self.db_path}")
            print("  Download from: https://github.com/google-deepmind/alphamissense")
            return
        
        print("Loading AlphaMissense database...")
        
        # Read compressed TSV
        with gzip.open(self.db_path, 'rt') as f:
            df = pd.read_csv(f, sep='\t', comment='#')
        
        # Create index: "GENE|PROTEIN_CHANGE" -> score
        self._index = {}
        for _, row in df.iterrows():
            key = f"{row['#CHROM']}:{row['POS']}:{row['REF']}:{row['ALT']}"
            self._index[key] = {
                'score': row['am_pathogenicity'],
                'class': row['am_class']
            }
        
        print(f"✓ Loaded {len(self._index)} AlphaMissense predictions")
    
    def get_score(self, chrom: str, pos: int, ref: str, alt: str) -> Optional[Dict]:
        """Get AlphaMissense score for variant"""
        if self._index is None:
            return None
        
        key = f"{chrom}:{pos}:{ref}:{alt}"
        return self._index.get(key)


# Example usage
if __name__ == "__main__":
    # Example: Annotate MTHFR C677T (rs1801133)
    annotator = VariantAnnotator()
    
    print("Annotating MTHFR C677T (rs1801133)...")
    annotation = annotator.annotate_variant(
        chrom="1",
        pos=11856378,
        ref="C",
        alt="T"
    )
    
    print("\n" + "="*60)
    print("Annotation Results:")
    print("="*60)
    print(f"Variant: {annotation.variant_id}")
    print(f"Gene: {annotation.gene_symbol}")
    print(f"Consequence: {annotation.consequence}")
    print(f"Protein change: {annotation.protein_change}")
    print(f"gnomAD AF (global): {annotation.gnomad_af}")
    print(f"gnomAD AF (South Asian): {annotation.gnomad_af_south_asian}")
    print(f"UniProt: {annotation.uniprot_id}")
