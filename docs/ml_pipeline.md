# Dirghayu ML Pipeline Architecture

## Overview

Genomics is fundamentally a **natural language processing problem** where DNA/protein sequences are treated as text with semantic meaning. This document outlines the ML models and pipelines for various prediction tasks in the Dirghayu platform.

---

## 1. Input Data Formats

### Primary Inputs

#### A. Genomic Data
```
Format: VCF (Variant Call Format)
Example:
#CHROM  POS     ID          REF  ALT    QUAL  FILTER  INFO
chr1    69511   rs75062661  A    G      100   PASS    AC=1;AF=0.000199681
chr1    865628  rs1278270   G    A      100   PASS    AC=1;AF=0.000199681
```

**Structure:**
- Chromosome position
- Reference/alternate alleles  
- Quality scores
- Annotations (gene, consequence, frequencies)

#### B. Clinical Phenotype Data
```json
{
  "patient_id": "IND_12345",
  "age": 45,
  "sex": "M",
  "ethnicity": "South_Indian",
  "height_cm": 170,
  "weight_kg": 75,
  "bmi": 25.9,
  "blood_pressure": {"systolic": 120, "diastolic": 80},
  "lipid_profile": {
    "total_cholesterol": 180,
    "ldl": 100,
    "hdl": 50,
    "triglycerides": 150
  },
  "family_history": {
    "diabetes": true,
    "cvd": false,
    "cancer": ["breast", "colon"]
  }
}
```

#### C. Lifestyle & Environmental Data
```json
{
  "diet": {
    "type": "vegetarian",
    "cuisine": "South_Indian",
    "calories_per_day": 2000,
    "macros": {"protein": 60, "carbs": 250, "fat": 70}
  },
  "exercise": {
    "frequency_per_week": 4,
    "intensity": "moderate",
    "type": ["walking", "yoga"]
  },
  "sleep_hours": 7,
  "stress_level": "moderate",
  "smoking": false,
  "alcohol": "occasional"
}
```

#### D. Protein Structures (Optional)
```
Format: PDB or AlphaFold2 predictions
Input: Protein sequences from coding variants
Output: 3D structural coordinates
```

---

## 2. Data Enrichment Pipeline

### Stage 1: Annotation Enrichment
```python
# Pseudocode pipeline
variant = parse_vcf(input_vcf)

# Add functional annotations
variant.gene = get_gene_symbol(variant)
variant.transcript = get_transcript_id(variant)
variant.consequence = predict_consequence(variant)  # missense, nonsense, etc.
variant.protein_change = map_to_protein(variant)

# Add population frequencies
variant.freq_gnomad = query_gnomad(variant)
variant.freq_genome_india = query_genome_india(variant)
variant.freq_regional = query_regional_db(variant, patient.ethnicity)

# Add pathogenicity scores
variant.cadd = query_cadd(variant)
variant.alphamissense = predict_alphamissense(variant)
variant.spliceai = predict_splicing(variant)
```

### Stage 2: Protein-Level Enrichment
```python
if variant.is_coding():
    # Get protein structure
    structure = fetch_alphafold(variant.protein_id)
    
    # Predict stability change
    wild_type_energy = calculate_folding_energy(structure)
    mutant_structure = apply_mutation(structure, variant)
    mutant_energy = calculate_folding_energy(mutant_structure)
    
    variant.delta_ddG = mutant_energy - wild_type_energy
    variant.stability_impact = classify_stability(variant.delta_ddG)
    
    # Predict aggregation propensity
    variant.aggregation_score = predict_aggregation(mutant_structure)
```

### Stage 3: Pathway Enrichment
```python
# Map variants to biological pathways
affected_genes = [v.gene for v in variants]

pathway_enrichment = {
    "mTOR_pathway": check_pathway_genes(affected_genes, MTOR_GENES),
    "insulin_signaling": check_pathway_genes(affected_genes, INSULIN_GENES),
    "autophagy": check_pathway_genes(affected_genes, AUTOPHAGY_GENES),
    "dna_repair": check_pathway_genes(affected_genes, DNA_REPAIR_GENES)
}
```

---

## 3. Why Genomics is an NLP Problem

### DNA as Language

**Genomic sequences have linguistic properties:**

| Genomics | NLP Analogy |
|----------|-------------|
| Nucleotides (A,C,G,T) | Characters/Letters |
| Codons (3-base units) | Words |
| Genes | Sentences |
| Pathways | Paragraphs/Documents |
| Regulatory grammar | Syntax |
| Mutation effects | Semantic meaning |

### Sequence Modeling Approaches

#### A. Transformer Models for Genomics
```python
# DNA sequence as tokens
sequence = "ATCGATCGATCG..."
tokens = tokenize_kmers(sequence, k=6)  # 6-mers as vocabulary

# Use BERT-like model
model = GenomicBERT(
    vocab_size=4**6,  # All possible 6-mers
    hidden_size=768,
    num_layers=12,
    num_heads=12
)

# Predict masked positions (variant effect prediction)
masked_sequence = mask_variant_position(sequence, variant_pos)
prediction = model(masked_sequence)
pathogenicity = classify_prediction(prediction)
```

#### B. Foundation Models
- **Nucleotide Transformer**: Pre-trained on entire human genome
- **Enformer**: Predicts gene expression from sequence
- **ESM (Evolutionary Scale Modeling)**: Protein language models
- **AlphaFold**: Structure prediction from sequence

---

## 4. ML Models for Prediction Tasks

### Task 1: Life Expectancy Prediction

**Model Architecture: Multi-Modal Ensemble**

```python
class LongevityPredictor(nn.Module):
    def __init__(self):
        # Genomic encoder
        self.genomic_encoder = TransformerEncoder(
            input_dim=variant_features,
            hidden_dim=512,
            num_layers=6
        )
        
        # Clinical encoder
        self.clinical_encoder = MLPEncoder(
            input_dim=clinical_features,
            hidden_dim=256
        )
        
        # Lifestyle encoder
        self.lifestyle_encoder = MLPEncoder(
            input_dim=lifestyle_features,
            hidden_dim=128
        )
        
        # Fusion layer
        self.fusion = nn.Linear(512 + 256 + 128, 256)
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Years of life expectancy
        )
    
    def forward(self, genomic_data, clinical_data, lifestyle_data):
        g_emb = self.genomic_encoder(genomic_data)
        c_emb = self.clinical_encoder(clinical_data)
        l_emb = self.lifestyle_encoder(lifestyle_data)
        
        fused = self.fusion(torch.cat([g_emb, c_emb, l_emb], dim=-1))
        life_expectancy = self.predictor(fused)
        
        return life_expectancy
```

**Features Used:**
- **Genomic**: Polygenic risk scores for aging-related diseases
- **Clinical**: Current health markers, family history
- **Lifestyle**: Diet, exercise, stress, sleep
- **Longevity variants**: FOXO3A, APOE, CETP variants

**Training Data:**
- Centenarian cohorts (100+ years)
- UK Biobank mortality data
- Indian longevity studies

---

### Task 2: Disease Vulnerability Prediction

**Model: Disease-Specific Risk Scorers**

```python
class DiseaseRiskModel(nn.Module):
    def __init__(self, disease_name):
        self.disease = disease_name
        
        # Variant aggregation (PRS-like)
        self.variant_aggregator = WeightedSum(
            weights=load_gwas_weights(disease_name)
        )
        
        # Clinical risk factors
        self.clinical_mlp = nn.Sequential(
            nn.Linear(clinical_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Protein vulnerability score
        self.protein_scorer = ProteinVulnerabilityNet()
        
        # Final risk predictor
        self.risk_head = nn.Linear(64 + protein_features, 1)
    
    def forward(self, variants, clinical, protein_data):
        prs = self.variant_aggregator(variants)
        clinical_emb = self.clinical_mlp(clinical)
        protein_vuln = self.protein_scorer(protein_data)
        
        risk_score = torch.sigmoid(
            self.risk_head(torch.cat([clinical_emb, protein_vuln], dim=-1))
        )
        
        return {
            "risk_score": risk_score,
            "prs": prs,
            "protein_vulnerability": protein_vuln
        }
```

**Disease-Specific Models:**

1. **Cardiovascular Disease (CVD)**
   - Input: Variants in APOE, LDLR, PCSK9
   - Features: Lipid levels, BP, age, BMI
   - Output: 10-year CVD risk

2. **Type 2 Diabetes (T2D)**
   - Input: Variants in TCF7L2, PPARG, KCNJ11
   - Features: HbA1c, fasting glucose, insulin resistance
   - Output: T2D risk score

3. **Cancer Predisposition**
   - Input: Variants in BRCA1/2, TP53, MLH1
   - Features: Family history, age, hormonal factors
   - Output: Cancer-specific risk (breast, colon, etc.)

4. **Alzheimer's Disease**
   - Input: APOE genotype, TREM2, CLU variants
   - Features: Cognitive scores, age, education
   - Output: Age-specific AD risk

---

### Task 3: Nutrient Metabolism & Food Suitability

**Model: Nutrigenomics Predictor**

```python
class NutrientMetabolismModel(nn.Module):
    def __init__(self):
        # Metabolic gene encoder
        self.metabolic_genes = [
            "MTHFR",  # Folate metabolism
            "FUT2",   # Vitamin B12 absorption
            "VDR",    # Vitamin D receptor
            "CYP1A2", # Caffeine metabolism
            "ALDH2",  # Alcohol metabolism
            "LCT",    # Lactose tolerance
            "AMY1",   # Starch digestion
        ]
        
        self.gene_encoder = nn.Embedding(len(self.metabolic_genes), 128)
        
        # Nutrient-gene interaction network
        self.interaction_net = nn.TransformerEncoder(
            d_model=128,
            nhead=8,
            num_layers=4
        )
        
        # Prediction heads
        self.absorption_head = nn.Linear(128, 1)  # 0-1 scale
        self.deficiency_head = nn.Linear(128, 1)  # Risk score
        self.suitability_head = nn.Linear(128, num_nutrients)
    
    def forward(self, genotype_data, dietary_intake):
        gene_embeddings = self.encode_genotypes(genotype_data)
        interactions = self.interaction_net(gene_embeddings)
        
        return {
            "absorption_rates": self.absorption_head(interactions),
            "deficiency_risk": self.deficiency_head(interactions),
            "nutrient_suitability": self.suitability_head(interactions)
        }
```

**Prediction Outputs:**

#### A. Absorption Rates
```python
predictions = {
    "vitamin_b12": {
        "absorption_rate": 0.65,  # 65% efficiency
        "reasoning": "FUT2 non-secretor genotype reduces B12 absorption",
        "recommendation": "Consider B12 supplementation"
    },
    "iron": {
        "absorption_rate": 0.85,
        "reasoning": "Normal HFE genotype, good iron metabolism",
        "recommendation": "Dietary iron sufficient"
    },
    "vitamin_d": {
        "absorption_rate": 0.40,
        "reasoning": "VDR variants reduce receptor sensitivity",
        "recommendation": "Higher vitamin D intake recommended"
    }
}
```

#### B. Food Suitability Scores
```python
food_suitability = {
    "dairy": {
        "score": 0.95,
        "reasoning": "LCT lactase persistence allele present",
        "recommendation": "Well-tolerated"
    },
    "caffeine": {
        "score": 0.30,
        "reasoning": "CYP1A2 slow metabolizer genotype",
        "recommendation": "Limit to 1 cup/day"
    },
    "alcohol": {
        "score": 0.10,
        "reasoning": "ALDH2 deficiency variant detected",
        "recommendation": "Avoid alcohol"
    },
    "gluten": {
        "score": 0.70,
        "reasoning": "No celiac-associated HLA variants",
        "recommendation": "Generally tolerated"
    }
}
```

#### C. Nutrient Deficiency Risk
```python
deficiency_risks = {
    "folate": {
        "risk_score": 0.75,  # High risk
        "gene": "MTHFR C677T homozygous",
        "reasoning": "Reduced enzyme activity affects folate metabolism",
        "recommendation": "Methylfolate supplementation",
        "dietary_sources": ["leafy greens", "legumes", "fortified grains"]
    },
    "omega3": {
        "risk_score": 0.60,
        "gene": "FADS1/FADS2 variants",
        "reasoning": "Reduced conversion of ALA to EPA/DHA",
        "recommendation": "Direct EPA/DHA from fish/algae",
        "dietary_sources": ["fatty fish", "algae oil"]
    }
}
```

---

## 5. Multi-Modal Learning Architecture

### Integration Strategy

```python
class DirgayuMultiModalModel(nn.Module):
    """
    Combines genomics, clinical, lifestyle, and environmental data
    for comprehensive health prediction
    """
    
    def __init__(self):
        # Modality-specific encoders
        self.dna_encoder = DNATransformer()
        self.protein_encoder = ProteinStructureNet()
        self.clinical_encoder = ClinicalDataEncoder()
        self.lifestyle_encoder = LifestyleEncoder()
        
        # Cross-modal attention
        self.cross_attention = MultiHeadCrossAttention(
            num_heads=8,
            hidden_dim=512
        )
        
        # Task-specific heads
        self.longevity_head = LongevityPredictor()
        self.disease_heads = nn.ModuleDict({
            "cvd": DiseaseHead("cardiovascular"),
            "t2d": DiseaseHead("diabetes"),
            "cancer": DiseaseHead("cancer"),
            "alzheimers": DiseaseHead("alzheimers")
        })
        self.nutrient_head = NutrientMetabolismHead()
        
    def forward(self, batch):
        # Encode each modality
        dna_emb = self.dna_encoder(batch['variants'])
        protein_emb = self.protein_encoder(batch['protein_structures'])
        clinical_emb = self.clinical_encoder(batch['clinical_data'])
        lifestyle_emb = self.lifestyle_encoder(batch['lifestyle_data'])
        
        # Cross-modal fusion
        fused_representation = self.cross_attention([
            dna_emb,
            protein_emb,
            clinical_emb,
            lifestyle_emb
        ])
        
        # Multi-task predictions
        outputs = {
            "longevity": self.longevity_head(fused_representation),
            "disease_risks": {
                disease: head(fused_representation)
                for disease, head in self.disease_heads.items()
            },
            "nutrient_metabolism": self.nutrient_head(fused_representation)
        }
        
        return outputs
```

---

## 6. Training Strategy

### Pre-training Phase
```python
# Step 1: Genomic foundation model (unsupervised)
genomic_model = pre_train_genomic_bert(
    data=human_reference_genome,
    task="masked_language_modeling",
    epochs=100,
    batch_size=256
)

# Step 2: Protein structure pre-training
protein_model = pre_train_protein_model(
    data=alphafold_structures,
    task="structure_prediction",
    epochs=50
)
```

### Fine-tuning Phase
```python
# Task-specific fine-tuning with Indian population data
model = fine_tune_on_indian_data(
    base_model=genomic_model,
    training_data={
        "genome_india": genome_india_cohort,
        "indigen": indigen_cohort,
        "centenarians": indian_centenarians
    },
    tasks=[
        "longevity_prediction",
        "disease_risk",
        "nutrient_metabolism"
    ]
)
```

### Multi-Task Learning
```python
# Joint training with shared representations
loss = (
    alpha * longevity_loss +
    beta * disease_risk_loss +
    gamma * nutrient_metabolism_loss +
    delta * protein_stability_loss
)
```

---

## 7. Evaluation Metrics

### Longevity Prediction
- **MAE (Mean Absolute Error)**: Years difference
- **Calibration**: Predicted vs actual survival curves
- **C-index**: Concordance for time-to-event

### Disease Risk
- **AUROC**: Area under ROC curve
- **Precision-Recall**: At different risk thresholds
- **Calibration plots**: Predicted vs observed risk

### Nutrient Metabolism
- **Correlation**: Predicted vs measured absorption
- **Classification accuracy**: Deficiency detection
- **Recommendation validation**: Intervention outcomes

---

## 8. Indian Population-Specific Considerations

### Genetic Diversity
```python
# Ancestry-aware modeling
ancestry_weights = {
    "Indo-Aryan": 0.40,
    "Dravidian": 0.35,
    "Tibeto-Burman": 0.15,
    "Austroasiatic": 0.10
}

# Adjust PRS based on ancestry
prs_adjusted = prs_base * ancestry_weights[patient.ancestry]
```

### Diet-Specific Modeling
```python
indian_dietary_patterns = {
    "North_Indian": {
        "primary_grains": ["wheat", "rice"],
        "protein_sources": ["dairy", "legumes"],
        "cooking_oils": ["ghee", "mustard_oil"]
    },
    "South_Indian": {
        "primary_grains": ["rice", "millets"],
        "protein_sources": ["lentils", "coconut"],
        "cooking_oils": ["coconut_oil", "sesame_oil"]
    }
}

# Nutrient predictions consider regional dietary context
```

### Common Indian Genetic Variants
```python
indian_specific_variants = {
    "APOL1": "Kidney disease risk",
    "HBB": "Sickle cell / thalassemia",
    "TCF7L2": "High T2D prevalence variant",
    "CETP": "Altered lipid metabolism",
    "CYP2C19": "Clopidogrel response (CVD treatment)"
}
```

---

## Next Steps

1. **Data Collection**: Gather training data from Indian cohorts
2. **Model Development**: Implement baseline models for each task
3. **Validation**: Clinical validation studies
4. **Deployment**: API endpoints for real-time predictions
5. **Continuous Learning**: Update models with new data

---

## References

- **Genomic NLP**: Nucleotide Transformer, Enformer
- **Protein Models**: AlphaFold2, ESM-2
- **PRS Methods**: PRSice-2, LDpred
- **Indian Genomics**: GenomeIndia, IndiGen projects
