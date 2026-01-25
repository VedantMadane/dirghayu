"""
DiseaseNet-Multi

Multi-task learning model for predicting risks of:
1. Cardiovascular Disease (CVD)
2. Type 2 Diabetes (T2D)
3. Cancers (Breast, Colorectal)
"""

import torch
import torch.nn as nn
from typing import Dict

class DiseaseNetMulti(nn.Module):
    def __init__(
        self,
        genomic_dim: int = 100,  # PRS scores + key variants
        clinical_dim: int = 100,  # Updated to 100 biomarkers
        hidden_dim: int = 256
    ):
        super().__init__()

        # Shared Encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(genomic_dim + clinical_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, hidden_dim),
            nn.ReLU()
        )

        # Task-Specific Heads

        # 1. CVD Head
        self.cvd_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 2. T2D Head
        self.t2d_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 3. Cancer Head (Multi-label: Breast, Colorectal, Prostate, Lung)
        self.cancer_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4), # 4 major types
            nn.Sigmoid()
        )

    def forward(self, genomic: torch.Tensor, clinical: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Concatenate inputs
        x = torch.cat([genomic, clinical], dim=-1)

        # Shared representation
        embedding = self.shared_encoder(x)

        # Predictions
        return {
            "cvd_risk": self.cvd_head(embedding),
            "t2d_risk": self.t2d_head(embedding),
            "cancer_risks": self.cancer_head(embedding) # [breast, colorectal, prostate, lung]
        }

def load_disease_model(path: str = "models/disease_net.pth") -> DiseaseNetMulti:
    model = DiseaseNetMulti()
    try:
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
    except Exception as e:
        print(f"Warning: Could not load model from {path}. Using random weights.")
    return model
