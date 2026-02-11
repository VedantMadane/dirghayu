"""
VCF Parser for Dirghayu

Parses VCF files and extracts features for ML models.
Uses cyvcf2 for fast parsing (C++ backend).
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Iterator
from pathlib import Path
import pandas as pd

try:
    from cyvcf2 import VCF

    CYVCF2_AVAILABLE = True
except ImportError:
    CYVCF2_AVAILABLE = False
    import sys

    # Only print if not in a test environment
    if sys.stdout.encoding and "utf" in sys.stdout.encoding.lower():
        print("⚠ cyvcf2 not available, falling back to basic parser")
    else:
        print("[!] cyvcf2 not available, falling back to basic parser")


@dataclass
class Variant:
    """Single genetic variant"""

    chrom: str
    pos: int
    ref: str
    alt: str
    qual: float
    filter: str
    info: Dict[str, any]
    genotype: str  # e.g., "0/1", "1/1"
    rsid: Optional[str] = None
    gene: Optional[str] = None
    consequence: Optional[str] = None

    @property
    def variant_id(self) -> str:
        """Unique variant identifier: chr:pos:ref:alt"""
        return f"{self.chrom}:{self.pos}:{self.ref}:{self.alt}"

    @property
    def is_het(self) -> bool:
        """Is heterozygous (0/1 or 1/0)"""
        return self.genotype in ["0/1", "1/0"]

    @property
    def is_hom_alt(self) -> bool:
        """Is homozygous alternate (1/1)"""
        return self.genotype == "1/1"

    @property
    def allele_count(self) -> int:
        """Number of alternate alleles (0, 1, or 2)"""
        if self.genotype == "0/0":
            return 0
        elif self.is_het:
            return 1
        elif self.is_hom_alt:
            return 2
        return 0


class VCFParser:
    """Fast VCF parser using cyvcf2"""

    def __init__(self, vcf_path: Path):
        self.vcf_path = Path(vcf_path)

        if not self.vcf_path.exists():
            raise FileNotFoundError(f"VCF file not found: {vcf_path}")

    def parse(self, sample_id: Optional[str] = None) -> Iterator[Variant]:
        """
        Parse VCF file and yield Variant objects

        Args:
            sample_id: Which sample to extract genotypes for (default: first sample)

        Yields:
            Variant objects
        """
        if CYVCF2_AVAILABLE:
            yield from self._parse_with_cyvcf2(sample_id)
        else:
            yield from self._parse_basic(sample_id)

    def parse_chunks(
        self, sample_id: Optional[str] = None, chunk_size: int = 10000
    ) -> Iterator[pd.DataFrame]:
        """
        Parse VCF file and yield pandas DataFrames in chunks.
        Efficient for processing large WGS files.

        Args:
            sample_id: Which sample to extract genotypes for
            chunk_size: Number of variants per chunk

        Yields:
            DataFrame chunks
        """
        buffer = []

        for variant in self.parse(sample_id):
            buffer.append(variant)

            if len(buffer) >= chunk_size:
                yield self._variants_to_df(buffer)
                buffer = []

        # Yield remaining
        if buffer:
            yield self._variants_to_df(buffer)

    def _variants_to_df(self, variants: List[Variant]) -> pd.DataFrame:
        """Convert list of variants to DataFrame"""
        if not variants:
            return pd.DataFrame()

        data = {
            "chrom": [v.chrom for v in variants],
            "pos": [v.pos for v in variants],
            "rsid": [v.rsid for v in variants],
            "ref": [v.ref for v in variants],
            "alt": [v.alt for v in variants],
            "genotype": [v.genotype for v in variants],
            "allele_count": [v.allele_count for v in variants],
            "qual": [v.qual for v in variants],
            "filter": [v.filter for v in variants],
        }

        # Add INFO fields as separate columns (sparse)
        # We check the first variant for keys, which is imperfect but fast
        if variants[0].info:
            for key in variants[0].info.keys():
                data[f"info_{key}"] = [v.info.get(key) for v in variants]

        return pd.DataFrame(data)

    def _parse_with_cyvcf2(self, sample_id: Optional[str]) -> Iterator[Variant]:
        """Fast parsing with cyvcf2"""
        vcf = VCF(str(self.vcf_path))

        # Determine which sample to use
        samples = vcf.samples
        if not samples:
            raise ValueError("VCF has no samples")

        if sample_id:
            if sample_id not in samples:
                raise ValueError(f"Sample {sample_id} not found. Available: {samples}")
            sample_idx = samples.index(sample_id)
        else:
            sample_idx = 0  # Use first sample

        for variant in vcf:
            # Extract genotype for this sample
            gt = variant.gt_types[sample_idx]  # 0=HOM_REF, 1=HET, 2=HOM_ALT, 3=UNKNOWN

            genotype_map = {0: "0/0", 1: "0/1", 2: "1/1", 3: "./."}
            genotype = genotype_map.get(gt, "./.")

            # Parse INFO field
            info_dict = {}
            if variant.INFO:
                try:
                    for key in variant.INFO:
                        try:
                            val = variant.INFO.get(key)
                            info_dict[key] = val
                        except Exception:
                            # Skip fields that cause parsing errors
                            pass
                except Exception:
                    pass

            yield Variant(
                chrom=variant.CHROM,
                pos=variant.POS,
                ref=variant.REF,
                alt=variant.ALT[0] if variant.ALT else ".",
                qual=variant.QUAL if variant.QUAL else 0.0,
                filter=variant.FILTER if variant.FILTER else "PASS",
                info=info_dict,
                genotype=genotype,
                rsid=variant.ID if variant.ID else None,
            )

    def _parse_basic(self, sample_id: Optional[str]) -> Iterator[Variant]:
        """Basic text parsing fallback (slower)"""
        with open(self.vcf_path, "r") as f:
            header_cols = None
            sample_idx = 0

            for line in f:
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Meta-information lines
                if line.startswith("##"):
                    continue

                # Header line
                if line.startswith("#CHROM"):
                    header_cols = line[1:].split("\t")
                    # Sample columns start after FORMAT column
                    if "FORMAT" in header_cols:
                        format_idx = header_cols.index("FORMAT")
                        samples = header_cols[format_idx + 1 :]

                        if sample_id and sample_id in samples:
                            sample_idx = samples.index(sample_id)
                        elif samples:
                            sample_idx = 0
                    continue

                # Data lines
                cols = line.split("\t")

                if len(cols) < 8:
                    continue

                chrom, pos, rsid, ref, alt, qual, filt, info_str = cols[:8]

                # Parse INFO
                info_dict = {}
                if info_str != ".":
                    for item in info_str.split(";"):
                        if "=" in item:
                            key, value = item.split("=", 1)
                            info_dict[key] = value
                        else:
                            info_dict[item] = True

                # Extract genotype
                genotype = "0/0"
                if len(cols) > 9:  # Has FORMAT and sample columns
                    format_fields = cols[8].split(":")
                    sample_data = cols[9 + sample_idx].split(":")

                    if "GT" in format_fields:
                        gt_idx = format_fields.index("GT")
                        if gt_idx < len(sample_data):
                            genotype = sample_data[gt_idx]

                yield Variant(
                    chrom=chrom,
                    pos=int(pos),
                    ref=ref,
                    alt=alt,
                    qual=float(qual) if qual != "." else 0.0,
                    filter=filt,
                    info=info_dict,
                    genotype=genotype,
                    rsid=rsid if rsid != "." else None,
                )

    def to_dataframe(self, sample_id: Optional[str] = None) -> pd.DataFrame:
        """
        Parse VCF and return as pandas DataFrame (loads all into memory).
        Use parse_chunks() for large files.

        Returns:
            DataFrame with columns: chrom, pos, rsid, ref, alt, genotype, etc.
        """
        variants = list(self.parse(sample_id))
        return self._variants_to_df(variants)


def parse_vcf_file(vcf_path: Path, sample_id: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to parse VCF file to DataFrame

    Args:
        vcf_path: Path to VCF file
        sample_id: Sample to extract (default: first sample)

    Returns:
        DataFrame with variant data
    """
    parser = VCFParser(vcf_path)
    return parser.to_dataframe(sample_id)


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vcf_parser.py <vcf_file> [sample_id]")
        sys.exit(1)

    vcf_file = Path(sys.argv[1])
    sample_id = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Parsing VCF: {vcf_file}")

    # Test streaming
    parser = VCFParser(vcf_file)
    chunk_count = 0
    total_variants = 0

    print("Streaming chunks...")
    for chunk in parser.parse_chunks(sample_id, chunk_size=10):
        chunk_count += 1
        total_variants += len(chunk)
        print(f"  Chunk {chunk_count}: {len(chunk)} variants")
        if chunk_count >= 5:
            print("  (Stopping demo after 5 chunks)")
            break

    print(f"\nTotal variants processed: {total_variants}")
