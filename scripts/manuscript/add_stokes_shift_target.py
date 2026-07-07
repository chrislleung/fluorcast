from pathlib import Path
import pandas as pd

in_path = Path("data/processed/fluodb_lite/combined_deduplicated.csv")
out_path = Path("data/processed/fluodb_lite/combined_deduplicated_with_stokes.csv")

df = pd.read_csv(in_path)

required = ["absorption_nm", "emission_nm"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing required columns: {missing}")

df["stokes_shift_nm"] = df["emission_nm"] - df["absorption_nm"]

paired = df[df["absorption_nm"].notna() & df["emission_nm"].notna()].copy()
valid = paired[paired["stokes_shift_nm"] >= 0].copy()
negative = paired[paired["stokes_shift_nm"] < 0].copy()

print("Total rows:", len(df))
print("Paired absorption/emission rows:", len(paired))
print("Physically valid Stokes rows:", len(valid))
print("Negative Stokes rows:", len(negative))
print("Mean valid Stokes shift:", valid["stokes_shift_nm"].mean())
print("Median valid Stokes shift:", valid["stokes_shift_nm"].median())

# Keep all original rows, but set invalid/negative Stokes values to NaN
df.loc[df["stokes_shift_nm"] < 0, "stokes_shift_nm"] = pd.NA

out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

print(f"\nSaved: {out_path}")
