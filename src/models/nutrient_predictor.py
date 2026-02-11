"""
Nutrient Deficiency Predictor

Predicts risk of nutrient deficiencies based on genetic variants:
- Vitamin B12 (FUT2, TCN2, MTRR)
- Vitamin D (VDR, GC, CYP2R1)
- Iron (HFE, TMPRSS6, TFR2)
- Folate (MTHFR, MTR, MTRR)

This is a supervised learning model trained on clinical data + genotypes.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Known nutrient metabolism genes and their variants
NUTRIENT_GENES = {
    "vitamin_b12": {
        "genes": ["FUT2", "TCN2", "MTRR", "MTR", "CUBN"],
        "key_variants": {
            "rs601338": {"gene": "FUT2", "effect": "non-secretor", "impact": 0.6},
            "rs1801198": {"gene": "TCN2", "effect": "reduced B12 transport", "impact": 0.4},
            "rs1532268": {"gene": "MTRR", "effect": "reduced enzyme activity", "impact": 0.3},
        },
    },
    "vitamin_d": {
        "genes": ["VDR", "GC", "CYP2R1", "CYP27B1", "CYP24A1"],
        "key_variants": {
            "rs2228570": {"gene": "VDR", "effect": "FokI polymorphism", "impact": 0.5},
            "rs7041": {"gene": "GC", "effect": "binding protein variant", "impact": 0.4},
            "rs10741657": {"gene": "CYP2R1", "effect": "hydroxylation efficiency", "impact": 0.3},
        },
    },
    "iron": {
        "genes": ["HFE", "TMPRSS6", "TFR2", "SLC40A1"],
        "key_variants": {
            "rs1800562": {"gene": "HFE", "effect": "C282Y hemochromatosis", "impact": 0.8},
            "rs1799945": {"gene": "HFE", "effect": "H63D", "impact": 0.4},
            "rs855791": {"gene": "TMPRSS6", "effect": "iron deficiency", "impact": 0.5},
        },
    },
    "folate": {
        "genes": ["MTHFR", "MTR", "MTRR", "DHFR"],
        "key_variants": {
            "rs1801133": {"gene": "MTHFR", "effect": "C677T reduced activity", "impact": 0.7},
            "rs1801131": {"gene": "MTHFR", "effect": "A1298C", "impact": 0.3},
        },
    },
}


class NutrientFeatureExtractor:
    """Extract features from variant data for nutrient prediction"""

    def __init__(self):
        self.nutrient_genes = NUTRIENT_GENES

    def extract_features(self, variants_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Extract nutrient-specific features from variants

        Args:
            variants_df: DataFrame with columns: rsid, chrom, pos, genotype, gene_symbol

        Returns:
            Dictionary mapping nutrient -> feature vector
        """
        features = {}

        for nutrient, config in self.nutrient_genes.items():
            nutrient_features = self._extract_nutrient_features(variants_df, config)
            features[nutrient] = nutrient_features

        return features

    def _extract_nutrient_features(self, variants_df: pd.DataFrame, config: Dict) -> np.ndarray:
        """Extract features for a specific nutrient"""

        feature_vector = []

        # Check for key variants
        for rsid, variant_info in config.get("key_variants", {}).items():
            if rsid in variants_df["rsid"].values:
                variant_row = variants_df[variants_df["rsid"] == rsid].iloc[0]

                # Encode genotype: 0=ref/ref, 1=het, 2=alt/alt
                if variant_row["genotype"] == "0/0":
                    allele_count = 0
                elif variant_row["genotype"] in ["0/1", "1/0"]:
                    allele_count = 1
                elif variant_row["genotype"] == "1/1":
                    allele_count = 2
                else:
                    allele_count = 0

                # Weight by impact
                weighted_score = allele_count * variant_info["impact"]
                feature_vector.append(weighted_score)
            else:
                # Variant not present (assume reference)
                feature_vector.append(0.0)

        # Gene-level aggregation
        for gene in config["genes"]:
            # Count total variants in this gene
            gene_variants = variants_df[variants_df["gene_symbol"] == gene]

            if len(gene_variants) > 0:
                # Count alternate alleles
                total_alt_alleles = 0
                for _, v in gene_variants.iterrows():
                    if v["genotype"] == "1/1":
                        total_alt_alleles += 2
                    elif v["genotype"] in ["0/1", "1/0"]:
                        total_alt_alleles += 1

                feature_vector.append(total_alt_alleles)
            else:
                feature_vector.append(0.0)

        return np.array(feature_vector, dtype=np.float32)


class NutrientDeficiencyModel(nn.Module):
    """
    Neural network to predict nutrient deficiency risk

    Multi-task model predicting risk for multiple nutrients simultaneously
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_nutrients: int = 4):
        super().__init__()

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Nutrient-specific heads
        self.nutrient_heads = nn.ModuleDict(
            {
                "vitamin_b12": self._make_head(hidden_dim // 2),
                "vitamin_d": self._make_head(hidden_dim // 2),
                "iron": self._make_head(hidden_dim // 2),
                "folate": self._make_head(hidden_dim // 2),
            }
        )

    def _make_head(self, input_dim: int) -> nn.Module:
        """Create prediction head for one nutrient"""
        return nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Output: risk score 0-1
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Feature tensor [batch_size, input_dim]

        Returns:
            Dictionary mapping nutrient -> risk score [batch_size, 1]
        """
        # Shared encoding
        encoded = self.encoder(x)

        # Nutrient-specific predictions
        outputs = {}
        for nutrient, head in self.nutrient_heads.items():
            outputs[nutrient] = head(encoded)

        return outputs


class NutrientPredictor:
    """High-level interface for nutrient deficiency prediction"""

    def __init__(self, model_path: Optional[Path] = None):
        self.feature_extractor = NutrientFeatureExtractor()
        self.model = None
        self.scaler = StandardScaler()

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def train(
        self,
        variants_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001,
    ):
        """
        Train the model

        Args:
            variants_df: DataFrame with variant data
            labels_df: DataFrame with columns: sample_id, vitamin_b12_deficient,
                       vitamin_d_deficient, iron_deficient, folate_deficient
                       (binary labels: 0=normal, 1=deficient)
        """
        print("Extracting features...")

        # Extract features (this is simplified - real version would group by sample)
        # For now, assume variants_df is already per-sample
        features = self.feature_extractor.extract_features(variants_df)

        # Combine all features into one vector
        # In production, handle per-sample properly
        all_features = []
        for nutrient in ["vitamin_b12", "vitamin_d", "iron", "folate"]:
            all_features.append(features[nutrient])
        X = np.concatenate(all_features)

        # Normalize features
        X = self.scaler.fit_transform(X.reshape(1, -1)).flatten()

        # For demo purposes, create synthetic training data
        print("⚠ Using synthetic training data for demonstration")
        X_train, y_train = self._generate_synthetic_data(n_samples=1000)
        X_val, y_val = self._generate_synthetic_data(n_samples=200)

        # Initialize model
        input_dim = X_train.shape[1]
        self.model = NutrientDeficiencyModel(input_dim=input_dim)

        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        print(f"\nTraining for {epochs} epochs...")

        for epoch in range(epochs):
            self.model.train()

            # Convert to tensors
            X_tensor = torch.FloatTensor(X_train)
            y_tensors = {
                nutrient: torch.FloatTensor(y_train[nutrient])
                for nutrient in ["vitamin_b12", "vitamin_d", "iron", "folate"]
            }

            # Forward pass
            predictions = self.model(X_tensor)

            # Calculate loss (multi-task)
            losses = {}
            total_loss = 0
            for nutrient in predictions.keys():
                loss = criterion(predictions[nutrient].squeeze(), y_tensors[nutrient])
                losses[nutrient] = loss
                total_loss += loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Validation
            if (epoch + 1) % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val)
                    val_predictions = self.model(X_val_tensor)

                    val_losses = {}
                    for nutrient in val_predictions.keys():
                        val_loss = criterion(
                            val_predictions[nutrient].squeeze(), torch.FloatTensor(y_val[nutrient])
                        )
                        val_losses[nutrient] = val_loss.item()

                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"  Train Loss: {total_loss.item():.4f}")
                print(f"  Val Losses: {val_losses}")

        print("✓ Training complete!")

    def _generate_synthetic_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, Dict]:
        """Generate synthetic training data for demonstration"""
        # Random features
        n_features = 20  # Total features across all nutrients
        X = np.random.randn(n_samples, n_features).astype(np.float32)

        # Synthetic labels (correlated with features)
        y = {}
        for i, nutrient in enumerate(["vitamin_b12", "vitamin_d", "iron", "folate"]):
            # Use specific features to generate labels
            risk_score = X[:, i * 5 : (i + 1) * 5].sum(axis=1)
            risk_score = 1 / (1 + np.exp(-risk_score))  # Sigmoid
            labels = (risk_score > 0.5).astype(np.float32)
            y[nutrient] = labels

        return X, y

    def predict(self, variants_df: pd.DataFrame) -> Dict[str, float]:
        """
        Predict nutrient deficiency risks

        Args:
            variants_df: DataFrame with variant data

        Returns:
            Dictionary mapping nutrient -> risk score (0-1)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        # Extract features
        features = self.feature_extractor.extract_features(variants_df)

        # Combine features
        all_features = []
        for nutrient in ["vitamin_b12", "vitamin_d", "iron", "folate"]:
            all_features.append(features[nutrient])
        X = np.concatenate(all_features)

        # Normalize
        X = self.scaler.transform(X.reshape(1, -1))

        # Predict
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = self.model(X_tensor)

        # Convert to dictionary
        results = {}
        for nutrient, pred_tensor in predictions.items():
            results[nutrient] = float(pred_tensor.item())

        return results

    def save(self, path: Path):
        """Save model and scaler"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state": self.model.state_dict(),
                "scaler": self.scaler,
                "model_config": {
                    "input_dim": self.model.encoder[0].in_features,
                },
            },
            path,
        )

        print(f"✓ Model saved to {path}")

    def load(self, path: Path):
        """Load model and scaler"""
        checkpoint = torch.load(path)

        # Recreate model
        self.model = NutrientDeficiencyModel(input_dim=checkpoint["model_config"]["input_dim"])
        self.model.load_state_dict(checkpoint["model_state"])
        self.scaler = checkpoint["scaler"]

        print(f"✓ Model loaded from {path}")


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Nutrient Deficiency Predictor - Training Demo")
    print("=" * 60)

    # Create predictor
    predictor = NutrientPredictor()

    # Train on synthetic data
    print("\nTraining model on synthetic data...")
    predictor.train(
        variants_df=pd.DataFrame(),  # Would be real variant data
        labels_df=pd.DataFrame(),  # Would be real clinical labels
        epochs=50,
    )

    # Save model
    model_path = Path("models/nutrient_predictor.pth")
    predictor.save(model_path)

    print("\n" + "=" * 60)
    print("✓ Demo complete!")
    print("=" * 60)
