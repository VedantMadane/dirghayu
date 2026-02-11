"""
Drug-Gene Interaction GNN

Predicts personalized drug response using a Graph Neural Network.
Models the complex interplay between Drugs, Genes, and Protein interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class DrugGeneGNN(nn.Module):
    def __init__(self, num_genes: int = 1000, num_drugs: int = 500, embedding_dim: int = 64):
        super().__init__()

        # Embeddings for nodes
        self.gene_embedding = nn.Embedding(num_genes, embedding_dim)
        self.drug_embedding = nn.Embedding(num_drugs, embedding_dim)

        # Message Passing Layers (Simplified GCN logic)
        # In a full implementation, we'd use torch_geometric.
        # Here we simulate the aggregation:
        # H_next = ReLU(Weights * (H_self + Sum(H_neighbors)))

        self.interaction_layer1 = nn.Linear(embedding_dim, embedding_dim)
        self.interaction_layer2 = nn.Linear(embedding_dim, embedding_dim)

        # Prediction Heads
        # 1. Efficacy (0-1)
        self.efficacy_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 2. Toxicity / Adverse Event Probability (0-1)
        self.toxicity_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, gene_indices: torch.Tensor, drug_indices: torch.Tensor, adjacency_matrix: torch.Tensor = None):
        """
        Args:
            gene_indices: [batch_size] IDs of relevant genes (e.g., CYP2C19)
            drug_indices: [batch_size] IDs of drugs (e.g., Clopidogrel)
            adjacency_matrix: Optional [nodes, nodes] graph structure for message passing
        """

        # Get initial embeddings
        g_emb = self.gene_embedding(gene_indices)
        d_emb = self.drug_embedding(drug_indices)

        # Simulate Graph Convolution (if adj provided)
        # For this demo, we assume direct interaction or simple aggregation
        # H_drug_updated = H_drug + Interaction(H_gene)

        # Simple interaction: Drug affected by Gene
        interaction = self.interaction_layer1(g_emb)
        d_emb_updated = d_emb + F.relu(interaction)

        # Combine for prediction
        combined = torch.cat([g_emb, d_emb_updated], dim=-1)

        return {
            "efficacy": self.efficacy_head(combined),
            "toxicity_risk": self.toxicity_head(combined)
        }

# Knowledge Base for Demo (Indices)
DRUG_MAP = {
    "Clopidogrel": 0,
    "Warfarin": 1,
    "Simvastatin": 2,
    "Metformin": 3,
    "Codeine": 4,
    "Aspirin": 5,
    "Ibuprofen": 6,
    "Caffeine": 7
}

GENE_MAP = {
    "CYP2C19": 0,
    "CYP2C9": 1,
    "VKORC1": 2,
    "SLCO1B1": 3,
    "SLC22A1": 4,
    "CYP2D6": 5,
    "CYP1A2": 6
}

def predict_drug_response(drug_name: str, key_gene: str, variant_impact: float = 1.0) -> Dict[str, float]:
    """
    Wrapper to use the GNN for specific pairs.
    variant_impact: Modifier based on patient's specific genotype (e.g., 0.5 for poor metabolizer).
    """
    model = DrugGeneGNN()
    # Load pretrained weights ideally
    # model.load_state_dict(...)
    model.eval()

    if drug_name not in DRUG_MAP or key_gene not in GENE_MAP:
        return {"efficacy": 0.5, "toxicity_risk": 0.1, "note": "Unknown drug/gene pair"}

    d_idx = torch.tensor([DRUG_MAP[drug_name]])
    g_idx = torch.tensor([GENE_MAP[key_gene]])

    with torch.no_grad():
        out = model(g_idx, d_idx)

    # Adjust based on variant impact (rule-based overlay on GNN output)
    # If variant_impact is low (poor metabolizer), efficacy drops or toxicity rises depending on drug type
    base_efficacy = out["efficacy"].item()
    base_toxicity = out["toxicity_risk"].item()

    # Logic: Prodrugs (Clopidogrel, Codeine) need metabolism -> Low impact = Low efficacy
    prodrugs = ["Clopidogrel", "Codeine"]

    if drug_name in prodrugs:
        final_efficacy = base_efficacy * variant_impact
        final_toxicity = base_toxicity # Toxicity might be lower if not activated
    else:
        # Active drugs (Warfarin) -> Low metabolism = High accumulation = High Toxicity
        final_efficacy = base_efficacy # Works fine
        final_toxicity = base_toxicity + (1.0 - variant_impact) * 0.5 # Increases risk

    return {
        "efficacy": min(max(final_efficacy, 0.0), 1.0),
        "toxicity_risk": min(max(final_toxicity, 0.0), 1.0)
    }
