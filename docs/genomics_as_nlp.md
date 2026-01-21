# Genomics as NLP: Language Models for DNA & Proteins

## Why Genomics IS Natural Language Processing

DNA and proteins exhibit all properties of natural languages:

| Property | Natural Language | Genomics |
|----------|-----------------|----------|
| **Alphabet** | 26 letters (English) | 4 nucleotides (A,C,G,T) / 20 amino acids |
| **Words** | Combinations of letters | Codons (3-base), k-mers (n-base) |
| **Grammar** | Syntax rules | Reading frames, splice sites, promoters |
| **Semantics** | Meaning | Protein function, gene regulation |
| **Context** | Surrounding words matter | Flanking sequences affect function |
| **Mutations** | Typos change meaning | Variants change protein function |
| **Evolution** | Language drift | Natural selection on sequences |
| **Dialects** | Regional variations | Population-specific variants |

---

## 1. DNA Sequence as Text

### Tokenization Strategies

#### A. K-mer Tokenization (Most Common)
```python
def tokenize_sequence(dna_sequence, k=6):
    """
    Convert DNA sequence to k-mer tokens
    
    Example:
    ATCGATCGATCG with k=3:
    ['ATC', 'TCG', 'CGA', 'GAT', 'ATC', 'TCG', 'CGA', 'GAT', 'TCG']
    """
    tokens = []
    for i in range(len(dna_sequence) - k + 1):
        kmer = dna_sequence[i:i+k]
        tokens.append(kmer)
    return tokens

# Vocabulary size for k-mers
vocab_size = 4**k  # k=6 → 4096 tokens

# Example encoding
sequence = "ATCGATCGATCG"
tokens = tokenize_sequence(sequence, k=6)
# ['ATCGAT', 'TCGATC', 'CGATCG', 'GATCGA', 'ATCGAT', 'TCGATC', 'CGATCG']
```

**Why K=6?**
- Captures codon context (2 codons)
- Manageable vocabulary size (4096)
- Good balance between specificity and generalization

#### B. BPE (Byte Pair Encoding) for Genomics
```python
from tokenizers import Tokenizer, models, trainers

# Train BPE on human genome
tokenizer = Tokenizer(models.BPE())
trainer = trainers.BpeTrainer(vocab_size=8000)
tokenizer.train_from_iterator(genome_sequences, trainer)

# Learns common genomic patterns
# Example learned tokens: "ATG" (start codon), "TAA" (stop), "GC-rich regions"
```

#### C. Codon-Aware Tokenization
```python
def tokenize_codons(coding_sequence):
    """
    Tokenize by codons (biological words)
    """
    assert len(coding_sequence) % 3 == 0, "Must be in-frame"
    
    codons = [coding_sequence[i:i+3] for i in range(0, len(coding_sequence), 3)]
    # ['ATG', 'GCT', 'AAA', 'TGA']  → [Start, Ala, Lys, Stop]
    
    # Map to amino acids (semantic translation)
    amino_acids = [genetic_code[codon] for codon in codons]
    # ['M', 'A', 'K', '*']
    
    return codons, amino_acids
```

---

## 2. Transformer Models for Genomics

### Architecture: Genomic BERT

```python
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

class GenomicBERT(nn.Module):
    """
    BERT-style model for genomic sequence understanding
    """
    
    def __init__(self, vocab_size=4096, hidden_size=768):
        super().__init__()
        
        config = BertConfig(
            vocab_size=vocab_size,      # 4^k k-mers
            hidden_size=hidden_size,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512
        )
        
        self.bert = BertModel(config)
        
        # Task-specific heads
        self.mlm_head = nn.Linear(hidden_size, vocab_size)  # Masked LM
        self.variant_effect_head = nn.Linear(hidden_size, 3)  # [benign, uncertain, pathogenic]
        self.expression_head = nn.Linear(hidden_size, 1)  # Gene expression prediction
    
    def forward(self, input_ids, attention_mask=None, task="mlm"):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        if task == "mlm":
            return self.mlm_head(sequence_output)
        elif task == "variant_effect":
            return self.variant_effect_head(sequence_output[:, 0])  # CLS token
        elif task == "expression":
            return self.expression_head(sequence_output[:, 0])
```

### Pre-training Tasks

#### Task 1: Masked Language Modeling
```python
# Like BERT for text
original:  ATCG[MASK][MASK]TCGATCG
predict:   ATCGATCGTCGATCG
         # Model learns: AT context predicts CG
```

#### Task 2: Variant Effect Prediction (NSP-like)
```python
# Predict if variant is disruptive
reference:  ATCGATCGATCG
variant:    ATCTATCGATCG  # G→T mutation
label:      pathogenic / benign
```

#### Task 3: Gene Expression Prediction
```python
# Predict expression from promoter sequence
promoter_sequence → gene_expression_level
# Learns: TATA box, CpG islands, TF binding sites
```

---

## 3. Protein Language Models

### ESM-2 Style Architecture

```python
class ProteinLanguageModel(nn.Module):
    """
    ESM-2 inspired model for protein sequences
    Vocabulary: 20 amino acids + special tokens
    """
    
    def __init__(self):
        self.embedding = nn.Embedding(
            num_embeddings=25,  # 20 AA + <pad>, <mask>, <cls>, <sep>, <unk>
            embedding_dim=1280
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=1280,
                nhead=20,
                dim_feedforward=5120
            ),
            num_layers=33
        )
        
        # Task heads
        self.structure_head = nn.Linear(1280, 3)  # x, y, z coordinates
        self.function_head = nn.Linear(1280, num_go_terms)
        self.stability_head = nn.Linear(1280, 1)
    
    def forward(self, protein_sequence):
        # protein_sequence: "MKVLWAALLVTFLAG..."
        tokens = self.tokenize_protein(protein_sequence)
        embeddings = self.embedding(tokens)
        
        # Self-attention over sequence
        encoded = self.transformer(embeddings)
        
        return {
            "structure": self.structure_head(encoded),
            "function": self.function_head(encoded[:, 0]),  # CLS token
            "stability": self.stability_head(encoded[:, 0])
        }
    
    def predict_mutation_effect(self, wild_type, mutant, position):
        """
        Core capability: predict how a mutation affects protein
        """
        wt_embedding = self.forward(wild_type)
        mut_embedding = self.forward(mutant)
        
        # Compare embeddings
        delta = mut_embedding - wt_embedding
        
        # Predict pathogenicity from embedding change
        pathogenicity = self.classify_delta(delta)
        
        return {
            "pathogenicity_score": pathogenicity,
            "confidence": calculate_confidence(delta),
            "affected_regions": identify_affected_regions(delta, position)
        }
```

### Zero-Shot Variant Effect Prediction

```python
def predict_variant_effect_zeroshot(protein_model, variant):
    """
    Predict effect WITHOUT seeing this specific variant before
    Uses learned sequence→structure→function mapping
    """
    
    # Wild-type sequence
    wt_sequence = get_protein_sequence(variant.gene)
    
    # Apply mutation
    mut_sequence = apply_mutation(wt_sequence, variant.aa_change)
    
    # Get embeddings (zero-shot)
    wt_logits = protein_model(wt_sequence)
    mut_logits = protein_model(mut_sequence)
    
    # Calculate log-likelihood ratio
    llr = (wt_logits - mut_logits).sum()
    
    # High LLR = mutation disrupts "normal" sequence → likely pathogenic
    pathogenicity = sigmoid(llr)
    
    return pathogenicity
```

---

## 4. Attention Mechanisms for Variant Interpretation

### Multi-Head Attention on Genomic Context

```python
class GenomicAttention(nn.Module):
    """
    Learn which genomic regions interact to determine variant effects
    """
    
    def __init__(self):
        self.attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=12
        )
    
    def forward(self, sequence_embeddings):
        # Query: variant position
        # Key/Value: surrounding genomic context
        
        attn_output, attn_weights = self.attention(
            query=sequence_embeddings,
            key=sequence_embeddings,
            value=sequence_embeddings
        )
        
        return attn_output, attn_weights

# Interpretation: 
# High attention weights = these regions influence variant effect
# Example: Promoter region attending to enhancer 50kb away
```

### Example: APOE Variant Interpretation

```python
# Input: rs429358 (APOE ε4 allele)
sequence_context = get_sequence_window(chromosome=19, position=44908684, window=10000)

attention_map = model.get_attention(sequence_context, variant_position=5000)

# Model learns:
# - Nearby regulatory elements
# - Linkage disequilibrium with other variants
# - Protein domain affected (lipid binding region)
# - Downstream pathway genes

interpretation = {
    "variant": "APOE ε4 (rs429358)",
    "attention_highlights": [
        "Lipid binding domain (direct effect)",
        "Regulatory enhancer 2kb upstream (expression effect)",
        "Linked variant affecting TOMM40 (independent effect)"
    ],
    "phenotypic_effect": "Increased Alzheimer's risk, altered lipid metabolism"
}
```

---

## 5. Transfer Learning Strategy

### Step 1: Foundation Model (Public Data)
```python
# Pre-train on massive public datasets
foundation_model = train_on_datasets([
    "1000_genomes",        # 2,504 individuals
    "gnomAD",              # 140,000 genomes
    "UK_biobank",          # 500,000 genomes
    "AlphaFold_structures" # 200M+ proteins
])
```

### Step 2: Indian Population Adaptation
```python
# Fine-tune on Indian-specific data
indian_model = fine_tune(
    base_model=foundation_model,
    indian_data=[
        "genome_india",  # 1,000 genomes
        "indigen",       # 1,000 genomes
        "igvdb"          # Indian variant database
    ],
    # Learn Indian-specific variant effects
    learning_rate=1e-5,
    epochs=10
)
```

### Step 3: Task-Specific Adaptation
```python
# Further fine-tune for specific tasks
longevity_model = fine_tune_task(
    base_model=indian_model,
    task="longevity_prediction",
    data=indian_centenarian_cohort
)
```

---

## 6. Sequence-to-Sequence Models

### Genomic Variant → Clinical Outcome

```python
class GenomeToPhenotypeModel(nn.Module):
    """
    Seq2Seq model: Genomic variants → Clinical phenotypes
    Like neural machine translation but for biology
    """
    
    def __init__(self):
        self.encoder = nn.TransformerEncoder(...)  # Encode variants
        self.decoder = nn.TransformerDecoder(...)  # Decode phenotypes
    
    def forward(self, variant_sequence):
        # Encoder: variant sequence → latent representation
        encoded = self.encoder(variant_sequence)
        
        # Decoder: latent → phenotype predictions
        # Output: "High CVD risk, Low T2D risk, Normal cognition, ..."
        phenotypes = self.decoder(encoded)
        
        return phenotypes

# Example:
input:  [APOE_e4, TCF7L2_risk, FOXO3A_protective, ...]
output: ["AD_risk_high", "T2D_risk_elevated", "longevity_favorable"]
```

---

## 7. Graph Neural Networks for Pathways

### Protein-Protein Interaction Networks

```python
class PathwayGNN(nn.Module):
    """
    Model biological pathways as graphs
    Nodes: Genes/Proteins
    Edges: Interactions
    """
    
    def __init__(self):
        self.node_encoder = nn.Linear(node_features, 128)
        
        # Graph convolutional layers
        self.gcn_layers = nn.ModuleList([
            GCNConv(128, 128) for _ in range(4)
        ])
        
        # Pathway vulnerability predictor
        self.pathway_head = nn.Linear(128, 1)
    
    def forward(self, protein_network, variant_effects):
        # Encode nodes (proteins) with variant effects
        node_features = self.encode_proteins_with_variants(
            protein_network,
            variant_effects
        )
        
        x = self.node_encoder(node_features)
        
        # Message passing through pathway
        for gcn in self.gcn_layers:
            x = gcn(x, protein_network.edges)
            x = F.relu(x)
        
        # Aggregate pathway vulnerability
        pathway_vulnerability = self.pathway_head(x).mean()
        
        return pathway_vulnerability

# Example: mTOR pathway vulnerability
mtor_genes = ["MTOR", "RPTOR", "RICTOR", "AKT1", "PIK3CA", "TSC1", "TSC2"]
variants_in_pathway = get_variants_in_genes(patient_vcf, mtor_genes)
pathway_vulnerability = pathway_gnn(mtor_network, variants_in_pathway)
```

---

## 8. Embedding-Based Similarity

### Genomic Embeddings for Population Clustering

```python
# Each genome gets a dense vector representation
genome_embeddings = genomic_bert.encode(patient_variants)
# Shape: [batch_size, 768]

# Find similar individuals
def find_similar_genomes(query_genome, database, top_k=10):
    query_emb = model.encode(query_genome)
    
    similarities = cosine_similarity(query_emb, database_embeddings)
    similar_indices = torch.topk(similarities, k=top_k)
    
    return database[similar_indices]

# Use case: Match to similar individuals for phenotype prediction
similar_patients = find_similar_genomes(new_patient, indian_cohort_db)
predicted_lifespan = similar_patients.mean_lifespan
```

---

## 9. Few-Shot Learning for Rare Variants

### Problem: Most Variants Are Rare

```python
# 99% of variants have frequency < 1%
# Not enough data for supervised learning

# Solution: Few-shot learning (like GPT few-shot prompting)

class FewShotVariantClassifier(nn.Module):
    def __init__(self, protein_lm):
        self.protein_lm = protein_lm  # Pre-trained ESM-2
    
    def classify_rare_variant(self, variant, support_set):
        """
        support_set: Few examples of known pathogenic/benign variants
        variant: New unseen variant to classify
        """
        
        # Embed support examples
        support_embeddings = [
            self.protein_lm.encode(v.protein_sequence)
            for v in support_set
        ]
        support_labels = [v.label for v in support_set]
        
        # Embed query variant
        query_embedding = self.protein_lm.encode(variant.protein_sequence)
        
        # Prototypical network: classify based on nearest prototype
        distances = [
            cosine_distance(query_embedding, supp_emb)
            for supp_emb in support_embeddings
        ]
        
        # Weighted vote
        prediction = weighted_vote(distances, support_labels)
        
        return prediction
```

---

## 10. Practical Examples

### Example 1: Predict MTHFR C677T Effect (Folate Metabolism)

```python
# Gene: MTHFR (methylenetetrahydrofolate reductase)
# Variant: C677T (rs1801133)
# Effect: Reduced enzyme activity → folate metabolism issues

# Input to model
variant_context = {
    "gene_sequence": "ATGC...GGAGGAGCTGACCAGTGAAGGGTACCTGGGCTCCCACCTC...TGAA",  # MTHFR gene
    "variant_position": 677,
    "ref": "C",
    "alt": "T",
    "protein_change": "p.Ala222Val",
    "population_frequency_india": 0.30  # 30% in Indians vs 10-12% Europeans
}

# Model processing (like NLP sentiment analysis)
sequence_embedding = genomic_bert.encode(variant_context["gene_sequence"])
variant_embedding = apply_variant(sequence_embedding, variant_context["variant_position"])

# Predict effect (classification task)
effect_prediction = variant_classifier(variant_embedding)

output = {
    "enzyme_activity": 0.65,  # 65% of normal (predicted from sequence)
    "folate_metabolism_impairment": "moderate",
    "homocysteine_elevation_risk": 0.75,
    "clinical_recommendations": [
        "Folate supplementation (methylfolate preferred)",
        "Monitor homocysteine levels",
        "Ensure adequate B6 and B12 intake"
    ]
}
```

### Example 2: Personalized Diet Recommendation

```python
# Patient profile
patient = {
    "variants": {
        "LCT": "lactase_persistent",      # Can digest milk
        "CYP1A2": "slow_caffeine",         # Slow caffeine metabolism
        "ALDH2": "deficient",              # Cannot metabolize alcohol
        "FADS1": "low_converter",          # Poor omega-3 conversion
        "AMY1": "high_copy",               # Excellent starch digestion
        "FTO": "obesity_risk"              # Weight gain susceptibility
    },
    "ethnicity": "South_Indian",
    "lifestyle": "sedentary",
    "current_diet": "vegetarian"
}

# Model generates personalized nutrition plan
nutrition_model = NutriGenomicsModel()
recommendations = nutrition_model.generate_diet_plan(patient)

output = {
    "macronutrient_targets": {
        "protein": "80g/day (high due to vegetarian + FTO variant)",
        "carbs": "200g/day (tolerated well due to high AMY1)",
        "fats": "60g/day (emphasize omega-3)"
    },
    "foods_to_emphasize": [
        "Lentils/legumes (protein, no lactose issues)",
        "Fatty fish OR algae oil (FADS1 → need direct EPA/DHA)",
        "Millets (complex carbs, weight management)",
        "Leafy greens (general health)"
    ],
    "foods_to_limit": [
        "Coffee: 1 cup max before noon (CYP1A2 slow)",
        "Alcohol: STRICTLY AVOID (ALDH2 deficiency + cancer risk)",
        "Refined carbs: minimize (FTO risk)"
    ],
    "foods_well_tolerated": [
        "Dairy products (LCT persistence)",
        "Rice/wheat (no gluten sensitivity variants)",
        "Most spices (no adverse metabolism variants)"
    ],
    "meal_timing": {
        "breakfast": "Protein-rich (satiety for FTO)",
        "lunch": "Largest meal",
        "dinner": "Light, early (< 8 PM)"
    }
}
```

### Example 3: Cardiovascular Risk with Dietary Modifiers

```python
# Patient: 45-year-old male, South Indian
genetic_profile = {
    "APOE": "e3/e4",           # Moderate AD + CVD risk
    "CETP": "I405V",            # Increased HDL
    "PCSK9": "loss_of_function", # Lower LDL (protective!)
    "TCF7L2": "risk_allele"     # T2D risk
}

clinical_data = {
    "ldl": 140,  # mg/dL
    "hdl": 45,
    "triglycerides": 180,
    "fasting_glucose": 105,
    "bp": "130/85"
}

# Model predicts
cvd_model = CVDRiskModel()
predictions = cvd_model.predict(genetic_profile, clinical_data)

output = {
    "10_year_cvd_risk": 0.18,  # 18%
    "genetic_component": {
        "favorable": ["PCSK9 LoF protects via low LDL"],
        "unfavorable": ["APOE e4 increases risk"]
    },
    "diet_modifications": {
        "saturated_fat": {
            "current_tolerance": "low",  # Due to APOE e4
            "recommendation": "< 7% of calories",
            "foods_to_limit": ["ghee", "coconut_oil", "red_meat", "full_fat_dairy"]
        },
        "omega3_intake": {
            "target": "2g EPA+DHA/day",
            "reasoning": "Offset APOE e4 inflammation",
            "sources": ["fatty_fish_2x_week", "fish_oil_supplement"]
        },
        "fiber": {
            "target": "30g/day",
            "reasoning": "Improve lipid profile + glucose control (TCF7L2)",
            "sources": ["whole_grains", "legumes", "vegetables"]
        }
    },
    "expected_risk_reduction": {
        "with_diet_changes": 0.12,  # 12% (from 18%)
        "with_statin_if_needed": 0.08  # 8%
    }
}
```

---

## 11. Model Serving Architecture

### Real-Time Inference API

```python
from fastapi import FastAPI
from transformers import AutoModel
import torch

app = FastAPI()

# Load models (cached in memory)
genomic_model = AutoModel.from_pretrained("dirghayu/genomic-bert-india")
protein_model = AutoModel.from_pretrained("dirghayu/protein-esm-india")
longevity_model = torch.load("models/longevity_net.pth")

@app.post("/predict/comprehensive")
async def comprehensive_analysis(patient_vcf: UploadFile):
    # Parse VCF
    variants = parse_vcf(await patient_vcf.read())
    
    # Extract features using NLP models
    genomic_features = genomic_model.encode(variants)
    protein_features = protein_model.encode(coding_variants(variants))
    
    # Run prediction models
    results = {
        "longevity": longevity_model(genomic_features, protein_features),
        "disease_risks": disease_suite(genomic_features),
        "nutrient_profile": nutrient_model(genomic_features)
    }
    
    return results
```

---

## 12. Why This Approach Works for Dirghayu

### Advantages for Indian Context

1. **Transfer Learning Handles Small Datasets**
   - Pre-train on large public datasets
   - Fine-tune on limited Indian data
   - Still get good performance

2. **Zero-Shot Capabilities**
   - Can predict effects of novel Indian-specific variants
   - No need to have seen exact variant before

3. **Multi-Modal Integration Natural**
   - Language models excel at combining different data types
   - Genomics + clinical + diet = multi-modal NLP problem

4. **Interpretable Predictions**
   - Attention mechanisms show which variants matter
   - Critical for clinical acceptance

5. **Continuous Improvement**
   - Models improve as more Indian data collected
   - Transfer learning accelerates this

---

## Implementation Priorities

### Phase 1: Foundation (Month 1-3)
- [ ] Train/adapt genomic BERT on Indian variants
- [ ] Fine-tune ESM-2 for Indian protein variants
- [ ] Build basic PRS calculators

### Phase 2: Clinical Models (Month 3-6)
- [ ] Longevity prediction model
- [ ] Disease risk models (CVD, T2D, cancer)
- [ ] Nutrient metabolism models

### Phase 3: Integration (Month 6-9)
- [ ] Multi-modal fusion architecture
- [ ] API deployment
- [ ] Clinical validation studies

### Phase 4: Optimization (Month 9-12)
- [ ] Model compression for edge deployment
- [ ] Federated learning setup
- [ ] Continuous learning pipeline

---

## Conclusion

By treating genomics as an NLP problem, Dirghayu can:
- Leverage powerful transformer architectures
- Transfer knowledge from public → Indian populations
- Make zero-shot predictions for rare variants
- Provide interpretable, clinically actionable insights
- Continuously improve with new data

The key insight: **DNA/protein sequences are languages that evolution has written, and modern NLP techniques can help us read them.**
