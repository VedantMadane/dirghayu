"""
Train Models Script

Generates synthetic data OR loads real data to train the Dirghayu AI models.
Produces .pth files for the Streamlit app.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.biomarkers import generate_synthetic_clinical_data, get_biomarker_names
from src.data.dataset import GenomicBigDataset
from src.models.disease_net import DiseaseNetMulti
from src.models.lifespan_net import LifespanNetIndia

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def train_lifespan_model(data_dir=None):
    print("Training LifespanNet-India...")

    # Hyperparams
    GENOMIC_DIM = 50
    CLINICAL_DIM = 100  # Updated to 100
    LIFESTYLE_DIM = 10
    EPOCHS = 50
    BATCH_SIZE = 1024

    model = LifespanNetIndia(GENOMIC_DIM, CLINICAL_DIM, LIFESTYLE_DIM)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()

    if data_dir:
        print(f"Loading real data from {data_dir}...")
        # Define features mapping
        feature_cols = [f"g_{i}" for i in range(GENOMIC_DIM)]
        # We assume dataset returns dict with 'genomic', 'clinical', 'lifestyle', 'targets' keys
        dataset = GenomicBigDataset(
            data_dir, feature_cols=feature_cols, target_cols={"lifespan": "age_death"}
        )
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)

        for epoch in range(EPOCHS):
            total_loss = 0
            count = 0
            for batch in loader:
                bs = batch["genomic"].shape[0]
                genomic = batch["genomic"]

                # Mock clinical data if missing
                # In real scenario, this would come from the parquet file
                clinical = torch.randn(bs, CLINICAL_DIM)
                lifestyle = torch.rand(bs, LIFESTYLE_DIM)
                target = batch["targets"]["lifespan"]

                optimizer.zero_grad()
                outputs = model(genomic, clinical, lifestyle)
                loss = criterion(outputs["predicted_lifespan"].squeeze(), target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                count += 1

            avg_loss = total_loss / max(1, count)
            print(f"  Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    else:
        # Synthetic Data
        N_SAMPLES = 1000
        genomic = torch.randint(0, 3, (N_SAMPLES, GENOMIC_DIM)).float()

        # Use our new biomarker generator
        clinical_dict = generate_synthetic_clinical_data(N_SAMPLES)
        clinical_array = np.array([clinical_dict[m] for m in get_biomarker_names()]).T  # [N, 100]
        # Normalize simple standard scaler mock
        clinical_mean = clinical_array.mean(axis=0)
        clinical_std = clinical_array.std(axis=0) + 1e-6
        clinical_norm = (clinical_array - clinical_mean) / clinical_std
        clinical = torch.tensor(clinical_norm).float()

        lifestyle = torch.rand(N_SAMPLES, LIFESTYLE_DIM)

        base_score = (
            genomic.mean(dim=1) * 0.5 + clinical.mean(dim=1) * -0.5 + lifestyle.mean(dim=1) * 2.0
        )
        lifespan_target = 78.0 + (base_score * 5.0) + torch.randn(N_SAMPLES)

        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            outputs = model(genomic, clinical, lifestyle)
            loss = criterion(outputs["predicted_lifespan"].squeeze(), lifespan_target)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.4f}")

    # Save
    torch.save(model.state_dict(), MODELS_DIR / "lifespan_net.pth")
    print("✓ Saved lifespan_net.pth\n")


def train_disease_model(data_dir=None):
    print("Training DiseaseNet-Multi...")

    # Hyperparams
    GENOMIC_DIM = 100
    CLINICAL_DIM = 100  # Updated to 100
    EPOCHS = 50
    BATCH_SIZE = 1024

    model = DiseaseNetMulti(GENOMIC_DIM, CLINICAL_DIM)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    model.train()

    if data_dir:
        print(f"Loading real data from {data_dir}...")
        feature_cols = [f"g_{i}" for i in range(GENOMIC_DIM)]
        dataset = GenomicBigDataset(
            data_dir, feature_cols=feature_cols, target_cols={"cvd": "has_cvd", "t2d": "has_t2d"}
        )
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)

        for epoch in range(EPOCHS):
            total_loss = 0
            count = 0
            for batch in loader:
                bs = batch["genomic"].shape[0]
                genomic = batch["genomic"]
                clinical = torch.randn(bs, CLINICAL_DIM)

                # Mock targets
                cvd_target = batch["targets"]["cvd"]
                t2d_target = batch["targets"]["t2d"]
                cancer_target = torch.zeros(bs, 4)  # Placeholder

                optimizer.zero_grad()
                outputs = model(genomic, clinical)

                loss_cvd = criterion(outputs["cvd_risk"], cvd_target.unsqueeze(1))
                loss_t2d = criterion(outputs["t2d_risk"], t2d_target.unsqueeze(1))
                loss_cancer = criterion(outputs["cancer_risks"], cancer_target)

                loss = loss_cvd + loss_t2d + loss_cancer
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                count += 1

            avg_loss = total_loss / max(1, count)
            print(f"  Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    else:
        # Synthetic Data
        N_SAMPLES = 1000
        genomic = torch.randint(0, 3, (N_SAMPLES, GENOMIC_DIM)).float()

        # Use our new biomarker generator
        clinical_dict = generate_synthetic_clinical_data(N_SAMPLES)
        clinical_array = np.array([clinical_dict[m] for m in get_biomarker_names()]).T  # [N, 100]
        clinical_mean = clinical_array.mean(axis=0)
        clinical_std = clinical_array.std(axis=0) + 1e-6
        clinical_norm = (clinical_array - clinical_mean) / clinical_std
        clinical = torch.tensor(clinical_norm).float()

        risk_score = genomic[:, :10].sum(dim=1) + clinical[:, :10].sum(dim=1)
        prob = torch.sigmoid(risk_score)

        cvd_target = (torch.rand(N_SAMPLES) < prob).float().unsqueeze(1)
        t2d_target = (torch.rand(N_SAMPLES) < prob * 0.8).float().unsqueeze(1)
        cancer_target = (torch.rand(N_SAMPLES, 4) < 0.1).float()

        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            outputs = model(genomic, clinical)

            loss_cvd = criterion(outputs["cvd_risk"], cvd_target)
            loss_t2d = criterion(outputs["t2d_risk"], t2d_target)
            loss_cancer = criterion(outputs["cancer_risks"], cancer_target)

            total_loss = loss_cvd + loss_t2d + loss_cancer

            total_loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss.item():.4f}")

    # Save
    torch.save(model.state_dict(), MODELS_DIR / "disease_net.pth")
    print("✓ Saved disease_net.pth\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Path to directory containing .parquet files")
    args = parser.parse_args()

    train_lifespan_model(args.data_dir)
    train_disease_model(args.data_dir)

    print("All models trained and saved!")
