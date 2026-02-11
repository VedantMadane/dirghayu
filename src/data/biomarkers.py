"""
Biomarker Definitions

Defines 100 clinical biomarkers used in the Dirghayu AI models.
Includes categories and reference ranges for synthetic generation and normalization.
"""

from typing import Dict, List

BIOMARKER_CATEGORIES = {
    "Lipid Profile": [
        "Total Cholesterol",
        "LDL-C",
        "HDL-C",
        "Triglycerides",
        "VLDL",
        "Non-HDL-C",
        "ApoA1",
        "ApoB",
        "Lp(a)",
        "Oxidized LDL",
    ],
    "Glucose Metabolism": [
        "Fasting Glucose",
        "HbA1c",
        "Insulin",
        "C-Peptide",
        "HOMA-IR",
        "Proinsulin",
        "1h Post-Prandial Glucose",
        "2h Post-Prandial Glucose",
        "Fructosamine",
        "Adiponectin",
    ],
    "Inflammation": [
        "hs-CRP",
        "IL-6",
        "TNF-alpha",
        "Fibrinogen",
        "ESR",
        "Homocysteine",
        "Ferritin",
        "Procalcitonin",
        "SAA",
        "Lp-PLA2",
    ],
    "Kidney Function": [
        "Creatinine",
        "BUN",
        "eGFR",
        "Uric Acid",
        "Cystatin C",
        "Albumin/Creatinine Ratio",
        "Sodium",
        "Potassium",
        "Chloride",
        "Bicarbonate",
    ],
    "Liver Function": [
        "ALT",
        "AST",
        "ALP",
        "GGT",
        "Total Bilirubin",
        "Direct Bilirubin",
        "Albumin",
        "Globulin",
        "Total Protein",
        "PT/INR",
    ],
    "Vitamins & Minerals": [
        "Vitamin D (25-OH)",
        "Vitamin B12",
        "Folate",
        "Iron",
        "TIBC",
        "Transferrin Saturation",
        "Magnesium",
        "Calcium",
        "Zinc",
        "Selenium",
    ],
    "Hormones": [
        "TSH",
        "Free T3",
        "Free T4",
        "Cortisol",
        "Testosterone",
        "Estrogen",
        "Progesterone",
        "SHBG",
        "DHEA-S",
        "IGF-1",
    ],
    "Hematology (CBC)": [
        "Hemoglobin",
        "Hematocrit",
        "RBC Count",
        "WBC Count",
        "Platelets",
        "MCV",
        "MCH",
        "MCHC",
        "RDW",
        "Neutrophils",
    ],
    "Cardiovascular": [
        "Troponin T",
        "NT-proBNP",
        "CK-MB",
        "Myoglobin",
        "D-Dimer",
        "Renin",
        "Aldosterone",
        "Endothelin-1",
        "MMP-9",
        "Galectin-3",
    ],
    "Oxidative Stress & Others": [
        "Glutathione",
        "SOD",
        "MDA",
        "8-OHdG",
        "CoQ10",
        "Omega-3 Index",
        "Telomere Length",
        "PSA",
        "CEA",
        "CA-125",
    ],
}

# Flatten the list
BIOMARKERS_100 = []
for cat, items in BIOMARKER_CATEGORIES.items():
    BIOMARKERS_100.extend(items)

assert len(BIOMARKERS_100) == 100, f"Expected 100 biomarkers, got {len(BIOMARKERS_100)}"

# Mock reference ranges (for synthetic generation)
# Format: (mean, std_dev) for a healthy population
REFERENCE_RANGES = {
    "Total Cholesterol": (180, 25),
    "LDL-C": (100, 20),
    "HDL-C": (50, 10),
    "Triglycerides": (120, 40),
    "Fasting Glucose": (90, 10),
    "HbA1c": (5.2, 0.4),
    "hs-CRP": (1.0, 0.5),
    "Vitamin D (25-OH)": (40, 10),
    "Testosterone": (500, 150),
    "Cortisol": (12, 4),
}


def get_biomarker_names() -> List[str]:
    return BIOMARKERS_100


def generate_synthetic_clinical_data(n_samples: int) -> Dict[str, List[float]]:
    """Generate synthetic data for 100 biomarkers"""
    import numpy as np

    data = {}
    for marker in BIOMARKERS_100:
        # Use specific params if defined, else generic
        mean, std = REFERENCE_RANGES.get(marker, (0.0, 1.0))  # Default to normalized

        # Generate with some random variation
        values = np.random.normal(mean, std, n_samples)
        data[marker] = values

    return data
