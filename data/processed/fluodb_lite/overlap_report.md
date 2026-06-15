# FluoDB-Lite Overlap Report

## Rows By Source Before Deduplication

- FluoDB-Lite: 49602
- deep4chem: 20833
- chemfluor: 4386

## Deduplication

- Total rows before deduplication: 74821
- Rows after deduplication: 66820
- Exact duplicate rows removed: 8001
- FluoDB-Lite exact overlaps with ChemFluor: 0
- FluoDB-Lite exact overlaps with Deep4Chem: 7854
- Molecule-solvent pairs with multiple measurements: 3605

Exact duplicates are defined as rows with the same canonical chromophore SMILES, canonical solvent SMILES, absorption, emission, quantum yield, and log extinction. Source priority is ChemFluor, then Deep4Chem, then FluoDB-Lite.

## Red/Orange/NIR Coverage

- >= 550 nm: before 16547, after 14850
- >= 580 nm: before 11280, after 10085
- >= 600 nm: before 8720, after 7780
- >= 650 nm: before 4186, after 3706
- >= 700 nm: before 1946, after 1720
