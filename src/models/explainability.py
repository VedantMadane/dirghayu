"""
Explainability & Backtracking Engine

Provides:
1. SHAP-based model explanations (Feature Attribution)
2. Backtracking logic (Risk -> Precaution -> Gene Expression)
"""

import torch
import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt
from .gene_expression import BacktrackingEngine, PrecautionImpact

class ExplainabilityManager:
    def __init__(self, background_samples: int = 100):
        self.backtracker = BacktrackingEngine()
        self.background_samples = background_samples
        self.background_data = None
        self.explainer = None

    def setup_shap(self, model: torch.nn.Module, input_data: torch.Tensor):
        """
        Initialize SHAP explainer for a given model.

        Args:
            model: PyTorch model
            input_data: Representative input data (e.g. training set sample)
        """
        # We use DeepExplainer for PyTorch models
        # Ensure model is in eval mode
        model.eval()

        # Select background samples
        if len(input_data) > self.background_samples:
            background = input_data[:self.background_samples]
        else:
            background = input_data

        try:
            self.explainer = shap.DeepExplainer(model, background)
        except Exception as e:
            print(f"Error initializing DeepExplainer: {e}")
            # Fallback to GradientExplainer or KernelExplainer if Deep fails
            # For this demo, we'll try to handle it or return None
            self.explainer = None

    def explain_prediction(
        self,
        input_tensor: torch.Tensor,
        feature_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for a single prediction.
        """
        if self.explainer is None:
            return {"error": "Explainer not initialized"}

        try:
            shap_values = self.explainer.shap_values(input_tensor)

            # Handle list output (for multi-output models)
            if isinstance(shap_values, list):
                shap_values = shap_values[0] # Take first output for simplicity

            # Create summary
            explanation = {
                "shap_values": shap_values,
                "feature_names": feature_names,
                "top_features": self._get_top_features(shap_values, feature_names)
            }

            return explanation

        except Exception as e:
            return {"error": str(e)}

    def _get_top_features(self, shap_values: np.ndarray, feature_names: List[str], top_k: int = 5):
        """Extract top driving features based on absolute SHAP value"""
        if isinstance(shap_values, list):
            vals = np.abs(shap_values[0]).mean(0) if len(shap_values) > 0 else np.array([])
        else:
            vals = np.abs(shap_values).flatten()

        indices = np.argsort(vals)[::-1][:top_k]

        top_feats = []
        for idx in indices:
            name = feature_names[idx] if feature_names else f"Feature {idx}"
            score = float(vals[idx])
            top_feats.append((name, score))

        return top_feats

    def get_backtracking_insights(self, disease_risks: Dict[str, float]) -> Dict[str, List[PrecautionImpact]]:
        """
        Get backtracking insights for high-risk conditions.

        Args:
            disease_risks: Dictionary of {disease: risk_score}

        Returns:
            Dictionary mapping disease -> list of precautions/gene impacts
        """
        insights = {}
        threshold = 0.5  # Risk threshold

        for disease, risk in disease_risks.items():
            if risk > threshold:
                # Get precautions from Knowledge Base
                # Map disease names to keys in gene_expression.py
                key_map = {
                    "cvd_risk": "cvd",
                    "t2d_risk": "t2d",
                    "cancer_risks": "cancer",
                    "cardiovascular": "cvd",
                    "diabetes": "t2d"
                }

                kb_key = key_map.get(disease, disease)
                precautions = self.backtracker.backtrack_risk(kb_key)

                if precautions:
                    insights[disease] = precautions

        return insights

    def plot_shap_summary(self, shap_values, feature_names):
        """Generate SHAP summary plot (returns figure)"""
        if shap_values is None:
            return None

        plt.figure()
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        return plt.gcf()
