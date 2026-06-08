# ChemFluor Project: Machine Learning Pipeline Summary

This document serves as a complete summary of the machine learning pipeline built to predict the Wavelength (Emission/nm) and Brightness (PLQY) of fluorescent chemicals based on their molecular structure (SMILES) and solvent.

---

## 1. Project Overview & Strategy
The goal of this project is to recreate and optimize the machine learning architecture from the ChemFluor research paper. We predict two targets:
1. **Wavelength (Color):** Predicted using a regression model (Continuous output in nm).
2. **PLQY (Brightness):** Predicted using both a Regressor (Exact decimal) and a Classifier (Binary: > 0.25 is Bright (1), <= 0.25 is Dim (0)).

---

## 2. Feature Engineering (The Clues)
The machine learning algorithms cannot read text (like "Toluene" or "CCO"). We translated the chemistry into math (`X` matrix) using three techniques:

1. **Morgan Fingerprints (The Camera):** Using RDKit, we look at atoms up to 2 bonds away (`RADIUS = 2`) and hash their shapes into an array of 2048 ones and zeros (`BITS = 2048`).
2. **One-Hot Encoding (The Solvents):** Using Pandas (`pd.get_dummies`), we converted the categorical text column of 55 different solvents into 55 separate binary math columns (1 if present, 0 if not).
3. **Physical Descriptors (The Physics Engine):** Fingerprints only see shapes. To break the Wavelength accuracy barrier, we added RDKit's built-in physics calculators to generate 6 real-number physics columns:
    * `MolWt` (Molecular Weight)
    * `MolLogP` (Water/Oil Solubility)
    * `NumHDonors` & `NumHAcceptors` (Hydrogen bonding)
    * `TPSA` (Topological Polar Surface Area / Electric Charge)
    * `RingCount` (Aromaticity)

**Final Feature Matrix (`X`) Shape:** `3090 rows × 2109 columns` (2048 Fingerprints + 55 Solvents + 6 Physics).

---

## 3. The Models & Upgrades

### A. The Wavelength "Super-Team" (Stacking Regressor)
To achieve the lowest Wavelength Error (MAE), we built a committee of algorithms. A `LinearRegression` "Boss" was put in charge of four "Experts":
* `GradientBoostingRegressor`
* `LGBMRegressor`
* `RandomForestRegressor`
* `SVR`

### B. The Brightness Experts (LightGBM)
Because PLQY is highly sensitive and chaotic, we deployed Microsoft's high-speed, highly-efficient `LightGBM` framework to predict both the exact Regressor values and the Classifier buckets.

### C. Hyperparameter Tuning (`GridSearchCV`)
We used `GridSearchCV` as an "Automated Scientist" to test hundreds of different brain settings for our algorithms (Cross-Validated 3 times to ensure no cheating). 
* **`n_estimators`**: The size of the decision tree workforce.
* **`max_depth`**: How deep the chain of "Yes/No" questions is allowed to go. (Note: `None` in Random Forest and `-1` in LightGBM mean the exact same thing: No limit).
* **`learning_rate`**: How aggressively the trees correct each other's mistakes.

### Evolution of Wavelength MAE Scores:
* **38.75 nm** (Untuned, Single GBRT, No Physics)
* **24.16 nm** (Untuned Stacking Super-Team, No Physics)
* **23.21 nm** (Tuned GBRT in Stacking, No Physics)
* **22.72 nm** (Fully Tuned Stacking, No Physics)
* **21.99 nm** (Fully Tuned Stacking, WITH Physics)

### Evolution of PLQY Scores (WITH Physics):
* **Regressor:** `0.1454` (Untuned) -> `0.1339` (Tuned)
* **Classifier:** `81.72%` (Untuned) -> `82.69%` (Tuned)

---

## 4. Key Data Science Concepts Discussed

* **`random_state=42`**: A seed for the random number generator. It ensures the computer shuffles the data and builds the trees the exact same way every time, ensuring reproducibility.
* **`X` vs `y`**: `X` (capitalized) is the 2D matrix of clues (Inputs). `y` (lowercase) is the 1D list of the actual answers (Targets).
* **Numpy vs Pandas**:
    * *Numpy (The Freight Train):* Strict, 100% mathematical arrays. Lightning fast because it uses "Vectorization" (doing math on massive chunks at once).
    * *Pandas (The Smart Filing Cabinet):* Built on top of Numpy. Allows labels, headers, mixed data types, and massive automated data cleaning (like `.dropna()`).
* **Mixing Binary and Continuous Data**: Tree-based algorithms (like LightGBM) are *scale-invariant*. They don't care that a fingerprint is `1` and a Molecular Weight is `345.2`. They simply look for the best "Yes/No" split point for each number, meaning we can mix them safely without crashing the math.
* **Overfitting**: When a model becomes too complex (e.g., too many trees, depth too high), it stops learning the actual chemistry and just memorizes the training CSV file. It will fail miserably on new real-world data.

---

## 5. The Final, Fully Optimized Code

```python
import numpy as np
import pandas as pd
import re
from pathlib import Path
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.svm import SVR
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator

DATA_PATH = Path(__file__).resolve().parent / "chemfluor_data.csv"
FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048

# Global Fingerprint Generator
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_BITS)

def smiles_to_morgan_fingerprint(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    fingerprint = mfpgen.GetFingerprintAsNumPy(mol)
    return fingerprint.astype(np.int8)

def get_physical_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0, 0, 0, 0, 0, 0]
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.RingCount(mol),
    ]

def build_feature_matrix(data_path: str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(data_path, encoding='latin1')
    df = df.dropna(subset=["SMILES", "solvent", "Emission/nm", "PLQY"])

    # 1. Fingerprints
    fingerprints = np.vstack(df["SMILES"].apply(smiles_to_morgan_fingerprint).to_numpy())
    fingerprint_columns = [f"morgan_{i}" for i in range(FINGERPRINT_BITS)]
    fingerprint_df = pd.DataFrame(fingerprints, columns=fingerprint_columns, index=df.index)

    # 2. Physics
    physical_properties = np.vstack(df["SMILES"].apply(get_physical_properties).to_numpy())
    physics_df = pd.DataFrame(
        physical_properties,
        columns=["MolWt", "LogP", "HDonors", "HAcceptors", "TPSA", "RingCount"],
        index=df.index,
    )

    # 3. Solvents (One-Hot)
    solvent_df = pd.get_dummies(df["solvent"], prefix="solvent", dtype=np.int8)

    # Assemble X
    X = pd.concat([fingerprint_df, solvent_df, physics_df], axis=1)
    X.columns = [re.sub(r"[^A-Za-z0-9_]", "_", column) for column in X.columns]
    
    y_plqy = df["PLQY"]
    y_wave = df["Emission/nm"]

    return X, y_wave, y_plqy


def train_stacked_wavelength_model(X, y_wave):
    X_train, X_test, y_train, y_test = train_test_split(X, y_wave, test_size=0.2, random_state=42)

    estimators = [
        ("gradient_boosting", GradientBoostingRegressor(n_estimators=500, max_depth=7, learning_rate=0.1, random_state=42)),
        ("lightgbm", LGBMRegressor(n_estimators=300, max_depth=-1, learning_rate=0.1, random_state=42)),
        ("random_forest", RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42)),
        ("svr", SVR()),
    ]

    model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Final Stacked Wavelength MAE: {mae:.4f}")
    return model


def train_tuned_plqy_models(X, y_plqy):
    X_train, X_test, y_train, y_test = train_test_split(X, y_plqy, test_size=0.2, random_state=42)
    
    y_train_class = (y_train > 0.25).astype(int)
    y_test_class = (y_test > 0.25).astype(int)

    # Tuned Regressor
    regressor = LGBMRegressor(learning_rate=0.05, max_depth=-1, n_estimators=500, random_state=42)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Final PLQY Regressor MAE: {mae:.4f}")

    # Tuned Classifier
    classifier = LGBMClassifier(learning_rate=0.1, max_depth=-1, n_estimators=300, random_state=42)
    classifier.fit(X_train, y_train_class)
    y_class_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test_class, y_class_pred)
    print(f"Final PLQY Classifier Accuracy: {accuracy:.4f}")

    return regressor, classifier


if __name__ == "__main__":
    X, y_wave, y_plqy = build_feature_matrix()
    print(f"Feature matrix (X) shape: {X.shape}")
    
    train_stacked_wavelength_model(X, y_wave)
    train_tuned_plqy_models(X, y_plqy)