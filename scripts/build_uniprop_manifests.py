"""Build stable UniProp molecule/row manifests and leakage-safe splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.uniprop.manifests import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    audit_split_leakage,
    build_manifests,
    make_split_assignments,
    split_statistics,
    training_normalization_statistics,
    validate_manifest_reconciliation,
    write_manifest_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create UniProp manifests from the processed FluorCast dataset."
    )
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--targets",
        default=None,
        help="Optional comma-separated target list. Defaults to known target columns present in the dataset.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--compute-inchikey",
        action="store_true",
        help="Compute RDKit InChIKeys for all unique molecules. This can be slow on the full dataset.",
    )
    parser.add_argument(
        "--compute-rdkit-properties",
        action="store_true",
        help="Compute formal charge and atom-count fields for all unique molecules.",
    )
    parser.add_argument(
        "--compute-nonisomeric",
        action="store_true",
        help="Compute RDKit non-isomeric canonical SMILES for all unique molecules.",
    )
    parser.add_argument(
        "--compute-rdkit-scaffolds",
        action="store_true",
        help="Use RDKit Bemis-Murcko scaffold groups for the scaffold split.",
    )
    return parser.parse_args()


def _targets(text: str | None) -> list[str] | None:
    if text is None:
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    try:
        bundle = build_manifests(
            args.dataset,
            target_columns=_targets(args.targets),
            compute_inchikey=args.compute_inchikey,
            compute_rdkit_properties=args.compute_rdkit_properties,
            compute_nonisomeric=args.compute_nonisomeric,
        )
        validate_manifest_reconciliation(bundle)
        splits = make_split_assignments(
            bundle.row_manifest,
            bundle.molecule_manifest,
            test_size=args.test_size,
            seed=args.seed,
            compute_rdkit_scaffolds=args.compute_rdkit_scaffolds,
        )
        leakage = audit_split_leakage(
            bundle.row_manifest,
            bundle.molecule_manifest,
            splits,
            compute_rdkit_scaffolds=args.compute_rdkit_scaffolds,
        )
        if not leakage["passed"].all():
            raise RuntimeError("Split leakage audit failed.")
        stats = split_statistics(
            bundle.row_manifest,
            bundle.molecule_manifest,
            splits,
            bundle.metadata["target_columns"],
            compute_rdkit_scaffolds=args.compute_rdkit_scaffolds,
        )
        normalization = training_normalization_statistics(
            bundle.row_manifest,
            splits,
            bundle.metadata["target_columns"],
        )
        write_manifest_outputs(args.out_dir, bundle, splits, leakage, stats, normalization)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Authoritative dataset: {bundle.source_path}")
    print(f"Molecule manifest rows: {len(bundle.molecule_manifest)}")
    print(f"Row manifest rows: {len(bundle.row_manifest)}")
    print(f"All leakage audits passed: {bool(leakage['passed'].all())}")
    print(f"Saved UniProp manifests to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
