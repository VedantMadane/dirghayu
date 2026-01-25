"""
Train Models Script

Generates synthetic data and trains the Dirghayu AI models.
Produces .pth files for the Streamlit app.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.lifespan_net import LifespanNetIndia
from src.models.disease_net import DiseaseNetMulti

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def train_lifespan_model():
    print("Training LifespanNet-India...")

    # Hyperparams
    N_SAMPLES = 1000
    GENOMIC_DIM = 50
    CLINICAL_DIM = 30
    LIFESTYLE_DIM = 10
    EPOCHS = 50

    # 1. Generate Synthetic Data
    # Genomic: random 0, 1, 2
    genomic = torch.randint(0, 3, (N_SAMPLES, GENOMIC_DIM)).float()

    # Clinical: random normal
    clinical = torch.randn(N_SAMPLES, CLINICAL_DIM)

    # Lifestyle: random 0-1
    lifestyle = torch.rand(N_SAMPLES, LIFESTYLE_DIM)

    # Generate Targets (Logic: more "good" genes/lifestyle = longer life)
    # Simple linear combination + noise
    base_score = (
        genomic.mean(dim=1) * 0.5 +
        clinical.mean(dim=1) * -0.5 + # Assume some clinical vars are "bad" like cholesterol
        lifestyle.mean(dim=1) * 2.0
    )
    lifespan_target = 78.0 + (base_score * 5.0) + torch.randn(N_SAMPLES)

    # 2. Train
    model = LifespanNetIndia(GENOMIC_DIM, CLINICAL_DIM, LIFESTYLE_DIM)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(genomic, clinical, lifestyle)
        loss = criterion(outputs["predicted_lifespan"].squeeze(), lifespan_target)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

    # 3. Save
    torch.save(model.state_dict(), MODELS_DIR / "lifespan_net.pth")
    print("✓ Saved lifespan_net.pth\n")

def train_disease_model():
    print("Training DiseaseNet-Multi...")

    # Hyperparams
    N_SAMPLES = 1000
    GENOMIC_DIM = 100
    CLINICAL_DIM = 20
    EPOCHS = 50

    # 1. Generate Synthetic Data
    genomic = torch.randint(0, 3, (N_SAMPLES, GENOMIC_DIM)).float()
    clinical = torch.randn(N_SAMPLES, CLINICAL_DIM)

    # Targets: Binary (0 or 1)
    # Logic: some features correlate with disease
    risk_score = (genomic[:, :10].sum(dim=1) + clinical[:, :5].sum(dim=1))
    prob = torch.sigmoid(risk_score)

    cvd_target = (torch.rand(N_SAMPLES) < prob).float().unsqueeze(1)
    t2d_target = (torch.rand(N_SAMPLES) < prob * 0.8).float().unsqueeze(1)
    cancer_target = (torch.rand(N_SAMPLES, 4) < 0.1).float() # 4 types

    # 2. Train
    model = DiseaseNetMulti(GENOMIC_DIM, CLINICAL_DIM)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(genomic, clinical)

        loss_cvd = criterion(outputs["cvd_risk"], cvd_target)
        loss_t2d = criterion(outputs["t2d_risk"], t2d_target)
        loss_cancer = criterion(outputs["cancer_risks"], cancer_target)

        total_loss = loss_cvd + loss_t2d + loss_cancer

        total_loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss.item():.4f}")

    # 3. Save
    torch.save(model.state_dict(), MODELS_DIR / "disease_net.pth")
    print("✓ Saved disease_net.pth\n")

if __name__ == "__main__":
    train_lifespan_model()
    train_disease_model()
    print("All models trained and saved!")
