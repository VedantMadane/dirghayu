# Dirghayu Model Specifications

## Model Inventory

### 1. Genomic Foundation Models

#### A. Dirghayu-DNA-BERT
**Purpose:** Pre-trained genomic sequence understanding
**Architecture:** BERT-style transformer
**Training:** Masked language modeling on human genome + Indian variants

```python
Model Specs:
- Vocabulary: 4096 tokens (6-mer k-mers)
- Hidden size: 768
- Layers: 12
- Attention heads: 12
- Parameters: ~110M
- Context window: 512 tokens (~3kb genomic sequence)
- Training data: GRCh38 + GenomeIndia variants
```

**Pre-training Tasks:**
1. Masked nucleotide prediction
2. Next sequence prediction
3. Variant effect prediction

#### B. Dirghayu-Protein-ESM
**Purpose:** Protein sequence understanding and stability prediction
**Architecture:** ESM-2 style protein language model

```python
Model Specs:
- Vocabulary: 20 amino acids + special tokens
- Hidden size: 1280
- Layers: 33
- Parameters: ~650M
- Context: 1024 amino acids
- Training: UniProt + AlphaFold structures
```

---

### 2. Life Expectancy Prediction Models

#### Model: LifespanNet-India

**Input Features (Total: ~500 features)**

```python
genomic_features = {
    "longevity_prs": 1,  # Polygenic risk score
    "disease_prs": {
        "cvd": 1,
        "t2d": 1,
        "cancer": 1,
        "alzheimers": 1
    },
    "key_variants": {
        "APOE_e4_count": 1,      # 0, 1, or 2 copies
        "FOXO3A_protective": 1,   # Binary
        "CETP_longevity": 1,      # Binary
        "IGF1R_variants": 1,      # Count
        "mTOR_variants": 1        # Count
    },
    "telomere_length_prs": 1,
    "dna_repair_capacity": 1
}

clinical_features = {
    "age": 1,
    "sex": 1,
    "bmi": 1,
    "blood_pressure": 2,  # systolic, diastolic
    "lipid_profile": 4,   # TC, LDL, HDL, TG
    "glucose_markers": 3, # fasting, HbA1c, insulin
    "inflammatory": 2,    # CRP, IL-6
    "kidney_function": 2, # creatinine, GFR
    "liver_function": 3,  # ALT, AST, bilirubin
    "complete_blood_count": 10
}

lifestyle_features = {
    "diet_quality_score": 1,
    "protein_intake_grams": 1,
    "fiber_intake_grams": 1,
    "sugar_intake_grams": 1,
    "exercise_minutes_per_week": 1,
    "sleep_hours": 1,
    "stress_score": 1,
    "smoking_pack_years": 1,
    "alcohol_units_per_week": 1
}

environmental_features = {
    "urban_vs_rural": 1,
    "air_quality_index": 1,
    "socioeconomic_status": 1,
    "healthcare_access": 1
}
```

**Architecture:**
```python
class LifespanNetIndia(nn.Module):
    def __init__(self):
        # Feature encoders
        self.genomic_net = nn.Sequential(
            nn.Linear(50, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        self.clinical_net = nn.Sequential(
            nn.Linear(30, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
        self.lifestyle_net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Attention-based fusion
        self.attention_fusion = nn.MultiheadAttention(
            embed_dim=224,  # 128 + 64 + 32
            num_heads=8
        )
        
        # Survival analysis head (inspired by Cox proportional hazards)
        self.survival_head = nn.Sequential(
            nn.Linear(224, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Log hazard ratio
        )
    
    def forward(self, genomic, clinical, lifestyle):
        g = self.genomic_net(genomic)
        c = self.clinical_net(clinical)
        l = self.lifestyle_net(lifestyle)
        
        combined = torch.cat([g, c, l], dim=-1)
        fused, _ = self.attention_fusion(combined, combined, combined)
        
        log_hazard = self.survival_head(fused)
        
        # Convert to life expectancy
        baseline_life_expectancy = 78.0  # Indian average
        relative_risk = torch.exp(log_hazard)
        predicted_lifespan = baseline_life_expectancy / relative_risk
        
        return {
            "predicted_lifespan": predicted_lifespan,
            "confidence_interval": calculate_ci(log_hazard),
            "contributing_factors": attention_weights_to_features(fused)
        }
```

**Training Details:**
- Loss: Cox partial likelihood + MSE on observed ages
- Data: Indian centenarian cohorts + mortality registries
- Validation: Time-split cross-validation
- Calibration: Isotonic regression post-processing

---

### 3. Disease Vulnerability Models

#### Model Suite: DiseaseNet-Multi

**Shared Architecture Pattern:**

```python
class DiseaseVulnerabilityModel(nn.Module):
    def __init__(self, disease_name, num_variants):
        # Variant effect aggregation (learned PRS)
        self.variant_embeddings = nn.Embedding(num_variants, 64)
        self.variant_weights = nn.Parameter(torch.randn(num_variants))
        
        # Protein vulnerability network
        self.protein_stability_encoder = nn.Sequential(
            nn.Linear(protein_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Clinical context
        self.clinical_context = nn.LSTM(
            input_size=clinical_dim,
            hidden_size=128,
            num_layers=2
        )
        
        # Disease risk predictor
        self.risk_predictor = nn.Sequential(
            nn.Linear(64 + 64 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
```

**Disease-Specific Instantiations:**

#### A. CVD Risk Model
```python
cvd_model = DiseaseVulnerabilityModel(
    disease_name="cardiovascular",
    key_genes=["APOE", "LDLR", "PCSK9", "SORT1", "PHACTR1"],
    clinical_features=[
        "age", "sex", "smoking", "diabetes",
        "total_cholesterol", "hdl", "sbp", "bp_treatment"
    ],
    protein_targets=[
        "LDL receptor stability",
        "PCSK9 binding affinity"
    ]
)
```

Output:
```python
{
    "10_year_risk": 0.15,  # 15% risk
    "risk_category": "intermediate",
    "framingham_score": 12,
    "genetic_contribution": 0.35,  # 35% from genetics
    "modifiable_factors": {
        "cholesterol_reduction": {"impact": 0.08, "priority": "high"},
        "exercise_increase": {"impact": 0.04, "priority": "medium"}
    }
}
```

#### B. Type 2 Diabetes Model
```python
t2d_model = DiseaseVulnerabilityModel(
    disease_name="type2_diabetes",
    key_genes=["TCF7L2", "PPARG", "KCNJ11", "SLC30A8"],
    clinical_features=[
        "fasting_glucose", "hba1c", "bmi", "waist_circumference",
        "family_history", "pcos", "gestational_diabetes"
    ],
    protein_targets=[
        "Insulin receptor sensitivity",
        "GLUT4 transporter function"
    ]
)
```

#### C. Cancer Predisposition Models
```python
cancer_models = {
    "breast": CancerModel(genes=["BRCA1", "BRCA2", "TP53", "PTEN"]),
    "colorectal": CancerModel(genes=["MLH1", "MSH2", "APC"]),
    "lung": CancerModel(genes=["EGFR", "TP53", "KRAS"]),
    "prostate": CancerModel(genes=["BRCA2", "HOXB13"])
}
```

---

### 4. Nutrigenomics Models

#### Model: NutriGene-India

**Architecture for Food-Gene Interactions:**

```python
class NutriGeneInteractionNet(nn.Module):
    def __init__(self):
        # Gene-food pairing encoder
        self.gene_embeddings = nn.Embedding(num_metabolic_genes, 128)
        self.food_embeddings = nn.Embedding(num_food_items, 128)
        
        # Interaction prediction
        self.interaction_net = nn.Bilinear(128, 128, 64)
        
        # Outcome predictors
        self.absorption_predictor = nn.Linear(64, 1)
        self.tolerance_predictor = nn.Linear(64, 1)
        self.benefit_predictor = nn.Linear(64, 1)
    
    def predict_food_suitability(self, genotype, food_item):
        gene_emb = self.gene_embeddings(genotype)
        food_emb = self.food_embeddings(food_item)
        
        interaction = self.interaction_net(gene_emb, food_emb)
        
        return {
            "absorption_rate": self.absorption_predictor(interaction),
            "tolerance_score": self.tolerance_predictor(interaction),
            "health_benefit": self.benefit_predictor(interaction)
        }
```

**Specific Predictions:**

#### Lactose Tolerance
```python
Input: LCT gene variants (rs4988235)
Output: {
    "lactase_persistence": True/False,
    "dairy_tolerance": 0.95,
    "recommendations": "Full dairy diet suitable"
}
```

#### Gluten Sensitivity
```python
Input: HLA-DQ2/DQ8 haplotypes
Output: {
    "celiac_risk": 0.05,  # Low risk
    "gluten_sensitivity_risk": 0.20,
    "recommendation": "Monitor symptoms, genetic risk low"
}
```

#### Caffeine Metabolism
```python
Input: CYP1A2 variants (rs762551)
Output: {
    "metabolism_rate": "slow",  # fast/normal/slow
    "half_life_hours": 8.0,     # vs 4-5 hours for fast metabolizers
    "recommended_max_cups": 1,
    "cut_off_time": "12:00 PM"
}
```

#### Alcohol Metabolism
```python
Input: ALDH2 variants (rs671)
Output: {
    "acetaldehyde_clearance": "impaired",
    "alcohol_flush_reaction": True,
    "cancer_risk_with_alcohol": 0.85,  # Highly elevated
    "recommendation": "Strict avoidance recommended"
}
```

#### Vitamin D Response
```python
Input: VDR, GC (vitamin D binding protein) variants
Output: {
    "receptor_sensitivity": 0.60,  # 60% of normal
    "binding_protein_level": "low",
    "recommended_daily_iu": 2000,  # vs 600 IU standard
    "sun_exposure_benefit": "moderate",
    "supplement_recommendation": "High-dose D3"
}
```

---

### 5. Indian Diet-Specific Models

#### Regional Food Suitability Matrix

```python
indian_foods_analyzed = {
    "grains": ["rice", "wheat", "millets", "sorghum"],
    "pulses": ["chickpeas", "lentils", "mung_beans", "pigeon_peas"],
    "dairy": ["milk", "yogurt", "paneer", "ghee"],
    "oils": ["ghee", "coconut", "mustard", "groundnut", "sesame"],
    "spices": ["turmeric", "cumin", "coriander", "fenugreek"],
    "vegetables": ["spinach", "okra", "eggplant", "bitter_gourd"]
}

# Model predicts compatibility for each food-genotype pair
for food in indian_foods_analyzed:
    suitability = model.predict(patient.genotype, food)
```

#### Nutrient Deficiency Risk Model

**Common Indian Deficiencies:**

```python
nutrient_models = {
    "vitamin_b12": {
        "genes": ["FUT2", "TCN2", "MTRR"],
        "risk_factors": ["vegetarian_diet", "age > 50"],
        "base_prevalence_india": 0.47  # 47% deficiency rate
    },
    "vitamin_d": {
        "genes": ["VDR", "GC", "CYP2R1", "CYP27B1"],
        "risk_factors": ["indoor_lifestyle", "dark_skin", "pollution"],
        "base_prevalence_india": 0.70  # 70% deficiency rate
    },
    "iron": {
        "genes": ["HFE", "TMPRSS6", "TFR2"],
        "risk_factors": ["vegetarian", "female", "pregnancy"],
        "base_prevalence_india": 0.53  # 53% anemia rate
    },
    "folate": {
        "genes": ["MTHFR", "MTR", "MTRR"],
        "risk_factors": ["low_green_vegetable_intake"],
        "base_prevalence_india": 0.15
    },
    "calcium": {
        "genes": ["VDR", "CASR"],
        "risk_factors": ["lactose_intolerance", "low_dairy"],
        "base_prevalence_india": 0.50
    }
}

class NutrientDeficiencyPredictor(nn.Module):
    def __init__(self):
        self.nutrient_specific_nets = nn.ModuleDict({
            nutrient: NutrientRiskNet(config)
            for nutrient, config in nutrient_models.items()
        })
    
    def predict(self, patient):
        deficiency_risks = {}
        
        for nutrient, model in self.nutrient_specific_nets.items():
            genetic_risk = model.genetic_component(patient.variants)
            dietary_risk = model.dietary_component(patient.diet)
            clinical_risk = model.clinical_component(patient.biomarkers)
            
            # Bayesian combination
            combined_risk = combine_bayesian(
                genetic_risk,
                dietary_risk,
                clinical_risk,
                base_prevalence=nutrient_models[nutrient]["base_prevalence_india"]
            )
            
            deficiency_risks[nutrient] = {
                "risk_score": combined_risk,
                "genetic_contribution": genetic_risk,
                "dietary_contribution": dietary_risk,
                "recommendation": generate_recommendation(combined_risk, nutrient)
            }
        
        return deficiency_risks
```

**Output Example:**
```python
{
    "vitamin_b12": {
        "risk_score": 0.82,  # 82% risk
        "category": "very_high",
        "genetic_contribution": 0.65,  # FUT2 non-secretor
        "dietary_contribution": 0.90,  # Strict vegetarian
        "clinical_markers": {
            "serum_b12": "not_measured",
            "homocysteine": "not_measured"
        },
        "recommendations": {
            "supplement": "Methylcobalamin 1000 mcg/day",
            "foods": ["fortified_cereals", "nutritional_yeast"],
            "monitoring": "Check serum B12 every 6 months"
        }
    },
    "vitamin_d": {
        "risk_score": 0.75,
        "category": "high",
        "genetic_contribution": 0.55,  # VDR variants
        "environmental_contribution": 0.70,  # Indoor lifestyle + pollution
        "recommendations": {
            "supplement": "Vitamin D3 2000 IU/day",
            "lifestyle": "15 min sun exposure daily (10am-12pm)",
            "monitoring": "Check 25(OH)D levels quarterly"
        }
    }
}
```

---

### 6. Protein Vulnerability Scoring

#### Model: ProteinStability-Net

**Purpose:** Predict how variants affect protein structure and function

```python
class ProteinVulnerabilityModel(nn.Module):
    def __init__(self):
        # Structure encoder (from AlphaFold coordinates)
        self.structure_encoder = ProteinGNN(
            node_features=20,  # amino acid type
            edge_features=4,   # distance, angles
            hidden_channels=256,
            num_layers=6
        )
        
        # Sequence context (ESM embeddings)
        self.sequence_context = ESM2Encoder(pretrained=True)
        
        # Mutation effect predictor
        self.stability_predictor = nn.Sequential(
            nn.Linear(256 + 1280, 512),  # GNN + ESM
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # ΔΔG, aggregation, pathogenicity
        )
    
    def forward(self, protein_structure, sequence, mutation_pos):
        # Encode wild-type structure
        struct_emb = self.structure_encoder(protein_structure)
        seq_emb = self.sequence_context(sequence)
        
        # Predict mutation effect
        combined = torch.cat([struct_emb, seq_emb], dim=-1)
        predictions = self.stability_predictor(combined)
        
        return {
            "delta_delta_g": predictions[0],  # kcal/mol
            "aggregation_propensity": predictions[1],
            "pathogenicity_score": predictions[2]
        }
```

**Longevity-Critical Proteins:**

```python
priority_proteins = {
    "DNA_repair": [
        "BRCA1", "BRCA2", "TP53", "ATM",
        "WRN", "BLM", "RECQL4"  # Werner/Bloom syndromes
    ],
    "metabolic": [
        "MTOR", "RPTOR", "IGF1R", "INSR",
        "AMPK", "SIRT1", "SIRT3"
    ],
    "autophagy": [
        "ATG5", "ATG7", "BECN1", "ULK1"
    ],
    "proteostasis": [
        "HSP70", "HSP90", "DNAJA1",
        "PSMC1", "PSMD1"  # Proteasome subunits
    ],
    "mitochondrial": [
        "POLG", "TFAM", "MT-CO1", "MT-ND1"
    ]
}
```

---

### 7. Food-Gene Interaction Prediction

#### Model: IndianDiet-GenomicsNet

**Training Data Sources:**
- UK Biobank dietary questionnaires + genomics
- Indian dietary surveys (NNMB, NFHS)
- Metabolomics studies
- Clinical intervention trials

```python
class FoodGeneInteractionModel(nn.Module):
    def __init__(self):
        # Food composition encoder
        self.food_encoder = nn.Sequential(
            nn.Linear(macro_micro_nutrients, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Metabolic gene encoder
        self.gene_encoder = nn.Sequential(
            nn.Linear(num_metabolic_variants, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Interaction predictor
        self.interaction = nn.Bilinear(64, 64, 128)
        
        # Multi-task heads
        self.absorption_head = nn.Linear(128, 1)
        self.glycemic_response_head = nn.Linear(128, 1)
        self.inflammatory_response_head = nn.Linear(128, 1)
        self.satiety_head = nn.Linear(128, 1)
```

**Prediction Tasks:**

#### Task A: Glycemic Response to Indian Foods
```python
foods_tested = [
    "white_rice", "brown_rice", "chapati", "dosa", "idli",
    "poha", "upma", "paratha", "biryani"
]

genes_analyzed = [
    "TCF7L2",  # Diabetes risk
    "GCK",     # Glucokinase
    "AMY1"     # Amylase (starch digestion)
]

output = {
    "food": "white_rice",
    "predicted_glucose_spike": 45,  # mg/dL rise
    "genetic_modifier": 1.3,  # 30% higher due to TCF7L2 variant
    "recommendation": "Replace with brown rice or millets",
    "alternative_foods": ["brown_rice", "foxtail_millet", "quinoa"]
}
```

#### Task B: Protein Digestion & Utilization
```python
Input:
- Genes: ACE, ACTN3 (muscle protein utilization)
- Diet: Vegetarian vs non-vegetarian protein sources

Output: {
    "protein_utilization_efficiency": 0.75,
    "optimal_protein_timing": "post_workout",
    "recommended_sources": {
        "plant": ["lentils", "chickpeas", "tofu"],
        "animal": ["fish", "eggs", "chicken"],
        "combined_strategy": "Mix plant proteins for complete amino acid profile"
    },
    "daily_requirement_grams": 85  # Personalized based on genetics + activity
}
```

#### Task C: Fat Metabolism & Cardiovascular Health
```python
Input:
- Genes: APOE, CETP, LIPC, FADS1/FADS2
- Diet: Ghee, coconut oil, mustard oil consumption

Output: {
    "apoe_genotype": "e3/e4",
    "saturated_fat_tolerance": "low",
    "omega3_conversion_efficiency": 0.45,  # FADS variants
    "recommendations": {
        "cooking_oils": {
            "avoid": ["coconut_oil", "palm_oil"],
            "prefer": ["olive_oil", "mustard_oil", "flaxseed_oil"],
            "ghee": "limit to 1 tsp/day"
        },
        "omega3_source": "Direct EPA/DHA from fish/algae (poor ALA conversion)",
        "cholesterol_intake": "< 200mg/day due to APOE4"
    }
}
```

---

## 8. Model Training Pipeline

### Data Preparation

```python
# Step 1: Create training dataset
training_data = {
    "genomics": load_vcf_files(genome_india_cohort),
    "phenotypes": load_clinical_data(),
    "outcomes": {
        "lifespan": load_mortality_data(),
        "disease_incidence": load_disease_registry(),
        "nutrient_levels": load_biomarker_data()
    }
}

# Step 2: Feature engineering
features = engineer_features(
    variants=training_data["genomics"],
    calculate_prs=True,
    encode_proteins=True,
    pathway_analysis=True
)

# Step 3: Train-validation-test split
# Stratified by age, sex, ethnicity
train, val, test = stratified_split(features, outcomes, ratios=[0.7, 0.15, 0.15])
```

### Training Loop

```python
# Multi-task training with uncertainty weighting
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        predictions = model(batch)
        
        # Multi-task loss
        losses = {
            "longevity": mse_loss(predictions["longevity"], batch["lifespan"]),
            "cvd_risk": bce_loss(predictions["cvd"], batch["cvd_label"]),
            "t2d_risk": bce_loss(predictions["t2d"], batch["t2d_label"]),
            "nutrient": mse_loss(predictions["nutrients"], batch["nutrient_levels"])
        }
        
        # Uncertainty weighting (learn task importance)
        total_loss = sum(
            losses[task] / (2 * task_uncertainties[task]**2) + torch.log(task_uncertainties[task])
            for task in losses
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

---

## 9. Inference Pipeline

### Real-Time Prediction API

```python
@app.post("/api/v1/predict/comprehensive")
async def predict_comprehensive_health(patient_data: PatientInput):
    # Load patient genomic data
    variants = load_variants(patient_data.vcf_file)
    
    # Enrich with annotations
    enriched_variants = enrich_pipeline(variants)
    
    # Run all prediction models
    predictions = {
        "longevity": longevity_model.predict(enriched_variants, patient_data.clinical),
        "disease_risks": {
            disease: model.predict(enriched_variants, patient_data.clinical)
            for disease, model in disease_models.items()
        },
        "nutrient_metabolism": nutrient_model.predict(enriched_variants, patient_data.diet),
        "food_suitability": food_model.predict(enriched_variants, indian_foods),
        "personalized_recommendations": generate_recommendations(all_predictions)
    }
    
    return predictions
```

### Batch Processing Pipeline

```python
# For research/population studies
batch_predictions = process_cohort(
    cohort=genome_india_samples,
    models=[longevity_model, disease_models, nutrient_models],
    output_format="parquet",
    partitions=["ethnicity", "age_group", "region"]
)
```

---

## 10. Model Interpretability

### SHAP Values for Genomic Predictions

```python
import shap

# Explain longevity prediction
explainer = shap.TreeExplainer(longevity_model)
shap_values = explainer.shap_values(patient_features)

# Top contributing factors
contributions = {
    "FOXO3A_protective_variant": +5.2,  # years
    "APOE_e4_allele": -3.8,
    "smoking_history": -4.5,
    "exercise_regular": +2.1,
    "mediterranean_diet": +1.8
}
```

### Attention Visualization

```python
# Show which genomic regions the model focuses on
attention_weights = model.get_attention_weights(sequence)
visualize_genome_browser(sequence, attention_weights)
```

---

## Model Performance Targets

| Task | Metric | Target | Baseline |
|------|--------|--------|----------|
| Life expectancy | MAE | < 5 years | 8-10 years |
| CVD risk (10yr) | AUROC | > 0.80 | 0.75 (Framingham) |
| T2D risk | AUROC | > 0.85 | 0.78 |
| Nutrient deficiency | F1-score | > 0.75 | 0.60 |
| Food tolerance | Accuracy | > 0.90 | N/A |

---

## Deployment Strategy

1. **Model Serving**: TorchServe / TensorFlow Serving / BentoML
2. **Batch Inference**: Spark + GPU for population analyses
3. **Real-time**: FastAPI with model caching
4. **Edge Deployment**: Quantized models for offline clinics
5. **Continuous Learning**: Federated learning from clinical outcomes

---

## Ethical Considerations

### Model Bias
- Ensure representation of all Indian ethnic groups
- Monitor performance across subpopulations
- Avoid discrimination based on genetic predictions

### Clinical Validation
- Prospective validation studies required
- Regulatory approval before clinical deployment
- Genetic counselor review of high-risk predictions

### Transparency
- Explainable AI for all clinical predictions
- Clear communication of uncertainty
- Right to refuse genetic testing
