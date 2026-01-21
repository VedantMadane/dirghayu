# Advanced VCF Analyses for Dirghayu

Beyond basic variant annotation, VCF files enable sophisticated genomics analyses for personalized health insights.

## 1. Pharmacogenomics (Drug Response Prediction)

### What It Does
Predicts how you metabolize and respond to medications based on genetic variants.

### Key Genes & Variants

**CYP450 Enzymes (Drug Metabolism):**
```python
pharmacogenomic_variants = {
    # Warfarin dosing (blood thinner)
    "CYP2C9": {
        "rs1799853": "*2 allele - reduced activity",
        "rs1057910": "*3 allele - reduced activity",
    },
    "VKORC1": {
        "rs9923231": "Affects warfarin sensitivity"
    },
    
    # Clopidogrel (anti-platelet) - common in India
    "CYP2C19": {
        "rs4244285": "*2 allele - poor metabolizer",
        "rs4986893": "*3 allele - poor metabolizer",
        "rs12248560": "*17 allele - ultra-rapid metabolizer"
    },
    
    # Statins (cholesterol drugs)
    "SLCO1B1": {
        "rs4149056": "Increased statin side effects"
    },
    
    # Codeine/morphine metabolism
    "CYP2D6": {
        "multiple_variants": "Poor/intermediate/ultra-rapid metabolizer"
    },
    
    # Metformin (diabetes drug - very common in India)
    "SLC22A1": {
        "rs622342": "Reduced metformin response"
    }
}
```

**Clinical Applications:**
- Personalized drug dosing
- Avoid adverse drug reactions
- Predict treatment effectiveness
- Choose optimal medications

**Indian Context:**
- 30% of Indians are CYP2C19 poor metabolizers (vs 2-5% Europeans)
- Critical for clopidogrel after heart stents (very common procedure)

---

## 2. Polygenic Risk Scores (PRS)

### What It Does
Combines effects of thousands of genetic variants to predict disease risk.

### Implementation
```python
def calculate_prs(vcf_variants, disease_weights_file):
    """
    Calculate polygenic risk score
    
    PRS = Σ (beta_i * dosage_i)
    where beta_i is effect size from GWAS
    """
    
    prs = 0
    for variant in vcf_variants:
        if variant.rsid in gwas_weights:
            beta = gwas_weights[variant.rsid]['beta']
            dosage = variant.allele_count  # 0, 1, or 2
            prs += beta * dosage
    
    # Normalize to percentile
    percentile = norm.cdf(prs, mean=population_mean, std=population_std)
    
    return {
        "raw_score": prs,
        "percentile": percentile,
        "risk_category": categorize_risk(percentile)
    }
```

**Diseases with Good PRS:**
- Coronary artery disease (CAD)
- Type 2 diabetes
- Breast cancer
- Prostate cancer
- Schizophrenia
- Inflammatory bowel disease

**Example Output:**
```python
{
    "coronary_artery_disease": {
        "prs": 2.3,  # Standard deviations above mean
        "percentile": 98.9,  # 98.9th percentile
        "risk": "very_high",
        "relative_risk": 3.5,  # 3.5x average risk
        "recommendations": [
            "Aggressive cholesterol management",
            "Aspirin prophylaxis consideration",
            "Annual cardiac stress test after age 40"
        ]
    }
}
```

---

## 3. Ancestry & Admixture Analysis

### What It Does
Determines your genetic ancestry and population mixtures.

### How It Works
```python
def infer_ancestry(vcf_file):
    """
    Use Principal Component Analysis (PCA) on ancestry-informative markers
    """
    
    # Select ancestry-informative SNPs
    aim_snps = load_ancestry_markers()  # ~10,000 SNPs
    
    # Extract genotypes
    genotypes = extract_genotypes(vcf_file, aim_snps)
    
    # Project onto reference populations
    pca = PCA(n_components=10)
    patient_coords = pca.transform(genotypes)
    
    # Compare to reference populations
    ancestry = {
        "Indo-Aryan": 0.45,
        "Dravidian": 0.35,
        "Tibeto-Burman": 0.15,
        "Austroasiatic": 0.05
    }
    
    return ancestry
```

**Applications:**
- Adjust PRS for ancestry (critical!)
- Understand disease risk patterns
- Optimize drug dosing
- Identify rare disease carrier status

**Indian Specificity:**
- India has 4,000+ distinct populations
- Endogamy creates unique genetic signatures
- Affects disease risk interpretation

---

## 4. Carrier Screening (Rare Diseases)

### What It Does
Identifies if you carry variants for recessive genetic diseases.

### Indian-Prevalent Conditions
```python
indian_carrier_screening = {
    # Blood disorders (very common in India)
    "beta_thalassemia": {
        "gene": "HBB",
        "carrier_frequency": "3-17% depending on region",
        "risk_areas": "Punjab, Gujarat, Tamil Nadu"
    },
    "sickle_cell": {
        "gene": "HBB",
        "carrier_frequency": "10-40% in tribal populations",
        "risk_areas": "Central India tribal belt"
    },
    
    # Metabolic disorders
    "g6pd_deficiency": {
        "gene": "G6PD",
        "carrier_frequency": "2-15%",
        "clinical_note": "Avoid certain medications (antimalarials)"
    },
    
    # Genetic deafness
    "gjb2_deafness": {
        "gene": "GJB2",
        "carrier_frequency": "1-2%",
        "note": "Common cause of congenital deafness"
    },
    
    # Mucopolysaccharidosis (MPS)
    "mps_disorders": {
        "genes": ["IDS", "IDUA", "GALNS"],
        "carrier_frequency": "Variable",
        "note": "Higher in South India"
    }
}
```

**Couple Screening:**
```python
def couple_carrier_screening(vcf_person1, vcf_person2):
    """
    Check if both partners carry same recessive variant
    """
    
    p1_carriers = identify_carrier_variants(vcf_person1)
    p2_carriers = identify_carrier_variants(vcf_person2)
    
    # Find overlapping genes
    shared_risk = set(p1_carriers.keys()) & set(p2_carriers.keys())
    
    for gene in shared_risk:
        risk = 0.25  # 25% chance of affected child
        print(f"⚠ Both carriers for {gene}: {risk:.0%} risk to offspring")
```

---

## 5. Athletic Performance Genetics

### What It Does
Predicts genetic predisposition for athletic traits.

### Key Variants
```python
athletic_genetics = {
    # Endurance vs Power
    "ACTN3": {
        "rs1815739": {
            "CC": "Power/sprint advantage (intact alpha-actinin-3)",
            "CT": "Balanced",
            "TT": "Endurance advantage (no alpha-actinin-3)"
        }
    },
    
    # VO2 max (aerobic capacity)
    "ACE": {
        "rs4340": {
            "II": "Higher VO2 max, endurance",
            "ID": "Intermediate",
            "DD": "Power/strength, lower VO2 max"
        }
    },
    
    # Muscle fiber composition
    "PPARA": {
        "rs4253778": "Affects slow-twitch vs fast-twitch fibers"
    },
    
    # Recovery rate
    "IL6": {
        "rs1800795": "Affects inflammation and recovery"
    },
    
    # Bone density (injury risk)
    "VDR": {
        "rs2228570": "Affects bone strength and stress fracture risk"
    }
}
```

**Applications:**
- Optimize training programs
- Injury prevention
- Sport selection guidance
- Recovery protocols

---

## 6. Longevity & Healthy Aging Analysis

### What It Does
Analyzes variants associated with exceptional longevity.

### Longevity Genes
```python
longevity_variants = {
    # The "longevity gene"
    "FOXO3A": {
        "rs2802292": "Associated with living to 100+",
        "rs13217795": "Centenarian variant",
        "effect": "Enhanced cellular stress resistance"
    },
    
    # Alzheimer's protection/risk
    "APOE": {
        "rs429358": {
            "ε2/ε2": "Protective against Alzheimer's",
            "ε3/ε3": "Neutral (most common)",
            "ε4/ε4": "5-15x increased AD risk"
        }
    },
    
    # DNA repair
    "SIRT6": {
        "variants": "DNA repair efficiency, cancer protection"
    },
    
    # Telomere maintenance
    "TERT": {
        "rs2736100": "Telomere length regulation"
    },
    
    # Growth hormone / IGF-1 pathway
    "IGF1R": {
        "variants": "Lower IGF-1 = longer life in some studies"
    },
    
    # mTOR pathway
    "MTOR": {
        "variants": "Nutrient sensing and aging"
    }
}
```

**Aging Clock:**
```python
def calculate_biological_age(vcf_variants, clinical_data):
    """
    Estimate biological age vs chronological age
    """
    
    genetic_score = calculate_longevity_prs(vcf_variants)
    epigenetic_age = estimate_methylation_age(clinical_data)
    
    biological_age = (
        0.3 * genetic_score +
        0.4 * epigenetic_age +
        0.3 * clinical_biomarkers_age
    )
    
    return {
        "chronological_age": 45,
        "biological_age": 38,  # 7 years younger!
        "aging_rate": "slower_than_average",
        "interventions": recommend_anti_aging_strategies()
    }
```

---

## 7. Microbiome-Genetics Interactions

### What It Does
Predicts how your genetics affects gut microbiome composition.

### Gene-Microbiome Links
```python
microbiome_genetics = {
    # Lactose tolerance
    "LCT": {
        "effect": "Affects Bifidobacterium levels",
        "mechanism": "Lactose availability in gut"
    },
    
    # Vitamin D receptor
    "VDR": {
        "effect": "Influences gut microbiome diversity",
        "mechanism": "Immune modulation"
    },
    
    # FUT2 (secretor status)
    "FUT2": {
        "effect": "Major determinant of microbiome composition",
        "mechanism": "Glycan availability for bacteria"
    },
    
    # NOD2 (Crohn's disease gene)
    "NOD2": {
        "effect": "Alters bacterial recognition",
        "clinical": "IBD risk via microbiome dysbiosis"
    }
}
```

**Personalized Probiotic Selection:**
```python
def recommend_probiotics(genetics, microbiome_test=None):
    """
    Recommend probiotics based on genetic predisposition
    """
    
    if genetics["FUT2"] == "non_secretor":
        return ["Bifidobacterium longum", "Lactobacillus rhamnosus"]
    
    if genetics["VDR"] == "poor_vitamin_d_response":
        return ["Strains that enhance vitamin D metabolism"]
```

---

## 8. Fitness & Exercise Response

### What It Does
Predicts how you respond to different types of exercise.

### Exercise Response Genetics
```python
exercise_genetics = {
    # Cardio response
    "aerobic_capacity": {
        "genes": ["ACE", "PPARGC1A", "VEGFA"],
        "prediction": "High/medium/low responder to cardio training"
    },
    
    # Strength training response
    "muscle_growth": {
        "genes": ["ACTN3", "MSTN", "IGF1"],
        "prediction": "Muscle building potential"
    },
    
    # Injury risk
    "injury_susceptibility": {
        "COL1A1": "Soft tissue injury risk",
        "COL5A1": "Tendon injury risk",
        "GDF5": "Joint injury risk"
    },
    
    # Exercise-induced inflammation
    "recovery_speed": {
        "genes": ["IL6", "TNF", "CRP"],
        "prediction": "Fast/slow recovery"
    }
}
```

**Training Optimization:**
```python
def personalized_training_plan(genetics, goals):
    """
    Create genetics-optimized training plan
    """
    
    if genetics["ACTN3"] == "RR":  # Power genotype
        return {
            "focus": "Strength and power training",
            "cardio": "HIIT > steady-state",
            "sports": "Sprinting, weightlifting, combat sports"
        }
    
    elif genetics["ACTN3"] == "XX":  # Endurance genotype
        return {
            "focus": "Endurance training",
            "cardio": "Long-distance > sprints",
            "sports": "Marathon, cycling, triathlon"
        }
```

---

## 9. Mental Health & Cognitive Traits

### What It Does
Analyzes genetic factors affecting mental health and cognition.

### Psychiatric Genetics
```python
mental_health_variants = {
    # Depression risk
    "SERT/5-HTTLPR": {
        "effect": "Stress sensitivity",
        "short_allele": "Higher depression risk with stress"
    },
    
    # ADHD
    "DRD4": {
        "7R_allele": "Associated with ADHD, novelty-seeking"
    },
    
    # Anxiety
    "COMT": {
        "Val158Met": {
            "Val/Val": "Faster dopamine breakdown, stress resilient",
            "Met/Met": "Slower breakdown, anxiety-prone"
        }
    },
    
    # Alzheimer's disease
    "APOE": {
        "ε4": "3-15x increased risk depending on copies"
    },
    
    # Schizophrenia
    "multiple_genes": {
        "PRS": "Polygenic risk score most useful"
    }
}
```

**Cognitive Traits:**
```python
cognitive_genetics = {
    # Memory
    "BDNF": {
        "rs6265": "Affects memory and learning"
    },
    
    # Intelligence (controversial)
    "polygenic_score": {
        "note": "Thousands of variants, small effects each",
        "educational_attainment_PRS": "More validated than IQ"
    },
    
    # Caffeine sensitivity
    "CYP1A2": {
        "rs762551": {
            "fast": "Can drink coffee late without sleep issues",
            "slow": "Coffee affects sleep for 8+ hours"
        }
    }
}
```

---

## 10. Cancer Risk & Screening

### What It Does
Identifies high-risk cancer predisposition variants.

### Hereditary Cancer Syndromes
```python
cancer_screening = {
    # Breast/ovarian cancer
    "BRCA1_BRCA2": {
        "prevalence_india": "Higher in certain communities",
        "risk": "Up to 80% lifetime breast cancer risk",
        "action": "Enhanced screening, prophylactic surgery option"
    },
    
    # Lynch syndrome (colon cancer)
    "MLH1_MSH2_MSH6_PMS2": {
        "risk": "80% lifetime colorectal cancer risk",
        "action": "Colonoscopy every 1-2 years starting age 25"
    },
    
    # Li-Fraumeni syndrome
    "TP53": {
        "risk": "Multiple cancer types",
        "action": "Whole-body MRI screening"
    },
    
    # Thyroid cancer (common in India)
    "RET": {
        "mutations": "Medullary thyroid carcinoma",
        "action": "Prophylactic thyroidectomy consideration"
    }
}
```

---

## 11. Circadian Rhythm & Sleep Genetics

### What It Does
Analyzes genetic variants affecting sleep patterns.

### Chronotype Genetics
```python
circadian_genetics = {
    # Are you a morning person?
    "PER3": {
        "5/5": "Extreme morning person",
        "4/4": "Evening person, delayed sleep phase"
    },
    
    # Clock genes
    "CLOCK": {
        "variants": "Sleep duration needs (short vs long sleeper)"
    },
    
    # Melatonin response
    "MTNR1B": {
        "variants": "Melatonin sensitivity, glucose metabolism"
    }
}
```

**Personalized Sleep Recommendations:**
```python
def optimize_sleep(genetics):
    if genetics["PER3"] == "4/4":  # Evening chronotype
        return {
            "natural_sleep_time": "1-2 AM",
            "wake_time": "9-10 AM",
            "productivity_peak": "Evening",
            "recommendation": "Night shift work may suit you better"
        }
```

---

## 12. Immune System & Infection Response

### What It Does
Predicts immune system characteristics and disease susceptibility.

### HLA Typing
```python
hla_analysis = {
    # Major histocompatibility complex
    "HLA-B27": {
        "diseases": "Ankylosing spondylitis (common in India)",
        "prevalence_india": "2-8% depending on region"
    },
    
    # Celiac disease
    "HLA-DQ2_DQ8": {
        "risk": "Required but not sufficient for celiac",
        "negative_predictive": "Absence rules out celiac 99%"
    },
    
    # HIV progression
    "HLA-B57": {
        "effect": "Slower HIV progression (elite controllers)"
    },
    
    # COVID-19 severity
    "various": {
        "ABO": "Blood type A = slightly higher risk",
        "TLR7": "Affects viral recognition"
    }
}
```

### Autoimmune Disease Risk
```python
autoimmune_genetics = {
    "rheumatoid_arthritis": {
        "HLA-DRB1": "Shared epitope",
        "PTPN22": "Risk variant"
    },
    
    "type_1_diabetes": {
        "HLA-DQ/DR": "Major risk factors",
        "INS": "Insulin gene variants"
    },
    
    "lupus": {
        "higher_in_women": True,
        "IRF5": "Risk variant"
    }
}
```

---

## 13. Quality Control & Relatedness

### What It Does
Validates VCF quality and checks for sample mix-ups.

### QC Metrics
```python
def vcf_quality_control(vcf_file):
    """
    Check VCF file quality
    """
    
    metrics = {
        # Basic stats
        "total_variants": count_variants(vcf_file),
        "ti_tv_ratio": calculate_titv(vcf_file),  # Should be ~2.0-2.1
        "het_hom_ratio": calculate_het_hom(vcf_file),  # Should be ~1.5-2.0
        
        # Quality metrics
        "mean_depth": calculate_depth(vcf_file),  # Should be >30x
        "mean_quality": calculate_qual(vcf_file),  # Should be >30
        
        # Contamination detection
        "contamination": estimate_contamination(vcf_file),  # Should be <2%
        
        # Sex check
        "reported_sex": "Male",
        "genetic_sex": infer_sex(vcf_file),  # Check X/Y coverage
        "sex_match": "Match" if reported == genetic else "MISMATCH!"
    }
    
    return metrics
```

### Relatedness Testing
```python
def check_relatedness(vcf1, vcf2):
    """
    Check if two VCF files are related
    """
    
    # Calculate identity-by-descent (IBD)
    ibd_score = calculate_ibd(vcf1, vcf2)
    
    relationships = {
        (0.9, 1.0): "Identical twins or same person",
        (0.35, 0.6): "Parent-child or full siblings",
        (0.15, 0.35): "Half-siblings or avuncular",
        (0.05, 0.15): "First cousins",
        (0.0, 0.05): "Unrelated"
    }
    
    return classify_relationship(ibd_score, relationships)
```

---

## 14. Structural Variants & CNVs

### What It Does
Detects large-scale genomic changes (deletions, duplications, inversions).

### Copy Number Variations
```python
def detect_cnvs(vcf_file):
    """
    Identify copy number variations
    """
    
    # Common pathogenic CNVs
    known_cnvs = {
        "22q11.2_deletion": {
            "disease": "DiGeorge syndrome",
            "features": "Heart defects, immune issues"
        },
        "15q13.3_deletion": {
            "disease": "Epilepsy, developmental delay"
        },
        "16p11.2_deletion": {
            "disease": "Autism, obesity"
        },
        
        # Indian-specific
        "alpha_thalassemia_deletion": {
            "common_in": "Tribal populations",
            "clinical": "Anemia"
        }
    }
    
    # Detect from read depth changes
    cnvs = call_cnvs_from_depth(vcf_file)
    
    return match_to_known(cnvs, known_cnvs)
```

---

## Summary: Advanced Analyses Priority for Dirghayu

**High Priority (Implement Soon):**
1. ✅ **Pharmacogenomics** - Critical for Indian population (CYP2C19 for clopidogrel)
2. ✅ **Polygenic Risk Scores** - CAD, T2D (major killers in India)
3. ✅ **Carrier screening** - Thalassemia, sickle cell (very common)
4. ✅ **Ancestry analysis** - Essential for PRS calibration

**Medium Priority:**
5. Athletic performance (fitness industry)
6. Longevity analysis (extends current work)
7. Microbiome interactions
8. Cancer screening (BRCA testing important)

**Lower Priority (Nice to Have):**
9. Mental health genetics (complex, ethical issues)
10. Sleep optimization
11. Immune system profiling
12. Structural variants (requires WGS, not just SNP arrays)

**Implementation Roadmap:**
1. Start with pharmacogenomics module (biggest immediate impact)
2. Add PRS calculators for Indian-prevalent diseases
3. Integrate carrier screening for regional diseases
4. Build ancestry adjustment algorithms

All of these can be implemented with the same VCF input we're already using!
