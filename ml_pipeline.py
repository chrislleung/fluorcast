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

# Create the generator right here at the top, so the whole script can see it!
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_BITS)

def smiles_to_morgan_fingerprint(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Use the new generator to make the fingerprint
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
    # latin1 fix for the special characters!
    df = pd.read_csv(data_path, encoding='latin1')
    
    # FIXED: Using the exact column names from your CSV
    df = df.dropna(subset=["SMILES", "solvent", "Emission/nm", "PLQY"])

    fingerprints = np.vstack(
        df["SMILES"].apply(smiles_to_morgan_fingerprint).to_numpy()
    )
    fingerprint_columns = [f"morgan_{i}" for i in range(FINGERPRINT_BITS)]
    fingerprint_df = pd.DataFrame(
        fingerprints,
        columns=fingerprint_columns,
        index=df.index,
    )

    physical_properties = np.vstack(
        df["SMILES"].apply(get_physical_properties).to_numpy()
    )
    physics_df = pd.DataFrame(
        physical_properties,
        columns=["MolWt", "LogP", "HDonors", "HAcceptors", "TPSA", "RingCount"],
        index=df.index,
    )

    # FIXED: Using 'solvent' with a lowercase 's'
    solvent_df = pd.get_dummies(df["solvent"], prefix="solvent", dtype=np.int8)

    # FIXED: No data leakage! Only inputs in X.
    X = pd.concat(
        [
            fingerprint_df,
            solvent_df,
            physics_df,
        ],
        axis=1,
    )
    X.columns = [re.sub(r"[^A-Za-z0-9_]", "_", column) for column in X.columns]
    
    y_plqy = df["PLQY"]
    
    # FIXED: Using 'Emission/nm' for the wavelength target
    y_wave = df["Emission/nm"]

    return X, y_wave, y_plqy

def train_wavelength_model(X, y_wave):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_wave,
        test_size=0.2,
        random_state=42,
    )

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Wavelength MAE: {mae:.4f}")

    return model


def tune_gbrt_wavelength(X, y_wave):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_wave,
        test_size=0.2,
        random_state=42,
    )

    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
    }

    base_model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Best GBRT parameters: {grid_search.best_params_}")
    print(f"Tuned GBRT MAE: {mae:.4f}")

    return best_model


def tune_other_models(X, y_wave):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_wave,
        test_size=0.2,
        random_state=42,
    )

    lgbm_model = LGBMRegressor(random_state=42)
    lgbm_grid = {
        "n_estimators": [100, 300],
        "learning_rate": [0.05, 0.1],
        "max_depth": [5, 10, -1],
    }
    lgbm_search = GridSearchCV(
        lgbm_model,
        lgbm_grid,
        cv=3,
        n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )
    lgbm_search.fit(X_train, y_train)

    lgbm_best_model = lgbm_search.best_estimator_
    lgbm_pred = lgbm_best_model.predict(X_test)
    lgbm_mae = mean_absolute_error(y_test, lgbm_pred)
    print(f"Best LGBM parameters: {lgbm_search.best_params_}")
    print(f"Tuned LGBM MAE: {lgbm_mae:.4f}")

    rf_model = RandomForestRegressor(random_state=42)
    rf_grid = {
        "n_estimators": [100, 300],
        "max_depth": [10, 20, None],
    }
    rf_search = GridSearchCV(
        rf_model,
        rf_grid,
        cv=3,
        n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )
    rf_search.fit(X_train, y_train)

    rf_best_model = rf_search.best_estimator_
    rf_pred = rf_best_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    print(f"Best Random Forest parameters: {rf_search.best_params_}")
    print(f"Tuned Random Forest MAE: {rf_mae:.4f}")

    return lgbm_best_model, rf_best_model


def train_stacked_wavelength_model(X, y_wave):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_wave,
        test_size=0.2,
        random_state=42,
    )

    estimators = [
        # OLD ESTIMATORS
        # ("gradient_boosting", GradientBoostingRegressor(random_state=42)),
        # ("lightgbm", LGBMRegressor(random_state=42)),
        # ("random_forest", RandomForestRegressor(random_state=42)),
        # ("svr", SVR()),

        # NEW, TUNED ESTIMATORS
        # The Tuned GBRT 
        ("gradient_boosting", GradientBoostingRegressor(n_estimators=500, max_depth=7, learning_rate=0.1, random_state=42)),
        
        # The Tuned LightGBM 
        ("lightgbm", LGBMRegressor(n_estimators=300, max_depth=-1, learning_rate=0.1, random_state=42)),
        
        # The Tuned Random Forest 
        ("random_forest", RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42)),
        
        # The standard SVR (SVR tuning takes too long for standard computers, so we leave it as the baseline worker!)
        ("svr", SVR()),

    ]

    model = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Stacked Wavelength MAE: {mae:.4f}")

    return model


def train_plqy_models(X, y_plqy):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_plqy,
        test_size=0.2,
        random_state=42,
    )

    regressor = LGBMRegressor(random_state=42)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"PLQY Regressor MAE: {mae:.4f}")

    y_train_class = (y_train > 0.25).astype(int)
    y_test_class = (y_test > 0.25).astype(int)

    classifier = LGBMClassifier(random_state=42)
    classifier.fit(X_train, y_train_class)

    y_class_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test_class, y_class_pred)
    print(f"PLQY Classifier Accuracy: {accuracy:.4f}")

    return regressor, classifier


def tune_plqy_models(X, y_plqy):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_plqy,
        test_size=0.2,
        random_state=42,
    )
    y_train_class = (y_train > 0.25).astype(int)
    y_test_class = (y_test > 0.25).astype(int)

    param_grid = {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 7, -1],
    }

    regressor = LGBMRegressor(random_state=42)
    regressor_search = GridSearchCV(
        regressor,
        param_grid,
        cv=3,
        n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )
    regressor_search.fit(X_train, y_train)

    regressor_best_model = regressor_search.best_estimator_
    y_pred = regressor_best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Best PLQY Regressor parameters: {regressor_search.best_params_}")
    print(f"Tuned PLQY Regressor MAE: {mae:.4f}")

    classifier = LGBMClassifier(random_state=42)
    classifier_search = GridSearchCV(
        classifier,
        param_grid,
        cv=3,
        n_jobs=-1,
        scoring="accuracy",
    )
    classifier_search.fit(X_train, y_train_class)

    classifier_best_model = classifier_search.best_estimator_
    y_class_pred = classifier_best_model.predict(X_test)
    accuracy = accuracy_score(y_test_class, y_class_pred)
    print(f"Best PLQY Classifier parameters: {classifier_search.best_params_}")
    print(f"Tuned PLQY Classifier Accuracy: {accuracy:.4f}")

    return regressor_best_model, classifier_best_model


# Test the script
if __name__ == "__main__":
    X, y_wave, y_plqy = build_feature_matrix()
    print(f"Feature matrix (X) shape: {X.shape}")
    print(f"Wavelength Target shape: {y_wave.shape}")
    print(f"PLQY Target shape: {y_plqy.shape}")
    # train_wavelength_model(X, y_wave)
    # train_stacked_wavelength_model(X, y_wave)
    # train_plqy_models(X, y_plqy)

    # tuning GBRT
    # tune_gbrt_wavelength(X, y_wave)

    # testing the tuned GBRT in the stacker
    # train_stacked_wavelength_model(X, y_wave)

    # tuning everything else
    # tune_other_models(X, y_wave)

    # train_stacked_wavelength_model(X, y_wave)
    # train_plqy_models(X, y_plqy)
    tune_plqy_models(X, y_plqy)
