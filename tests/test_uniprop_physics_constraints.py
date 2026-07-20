from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

torch = pytest.importorskip("torch")

from chemfluor.uniprop.physics_constraints import (  # noqa: E402
    HC_EV_NM,
    PHYSICS_MODEL_VARIANTS,
    PhysicsConstrainedOutputHead,
    build_stokes_masks,
    derive_rates,
    energy_ev_to_wavelength_nm,
    masked_physics_loss,
    migrate_three_target_checkpoint,
    physics_consistency_metrics,
    wavelength_nm_to_energy_ev,
)


TARGETS = ("absorption_nm", "emission_nm", "quantum_yield", "lifetime_ns", "log_extinction")


def test_wavelength_energy_round_trip_without_intermediate_rounding() -> None:
    wavelengths = torch.tensor([287.123456789, 431.987654321, 812.246813579], dtype=torch.float64)
    energies = wavelength_nm_to_energy_ev(wavelengths)
    assert torch.equal(energies, HC_EV_NM / wavelengths)
    torch.testing.assert_close(energy_ev_to_wavelength_nm(energies), wavelengths, rtol=1e-14, atol=1e-14)


def test_complete_head_outputs_are_physically_bounded_and_consistent() -> None:
    model = PhysicsConstrainedOutputHead.build(torch, input_dim=4, variant="complete")
    features = torch.tensor([[0.1, -0.2, 0.3, 0.4], [1.0, 0.5, -0.25, 0.75]], dtype=torch.float32)
    outputs = model(features)
    metrics = physics_consistency_metrics(outputs)

    assert bool(((outputs["quantum_yield"] >= 0.0) & (outputs["quantum_yield"] <= 1.0)).all())
    assert bool((outputs["lifetime_ns"] > 0.0).all())
    assert bool((outputs["stokes_energy_ev"] >= 0.0).all())
    assert metrics["ordinary_stokes_violation_count"] == 0.0
    assert metrics["quantum_yield_bounds_violation_count"] == 0.0
    assert metrics["lifetime_positive_violation_count"] == 0.0
    assert metrics["max_stokes_equation_error"] <= 1.0e-6
    assert metrics["max_quantum_yield_rate_error"] <= 1.0e-6
    assert metrics["max_lifetime_rate_error"] <= 1.0e-5


def test_rate_equations_match_closed_form_values() -> None:
    log_k_r = torch.tensor([10.0, 11.0], dtype=torch.float64)
    log_k_nr = torch.tensor([9.0, 13.0], dtype=torch.float64)
    outputs = derive_rates(log_k_r, log_k_nr)
    expected_total = torch.exp(log_k_r) + torch.exp(log_k_nr)

    torch.testing.assert_close(outputs["quantum_yield"], torch.exp(log_k_r) / expected_total)
    torch.testing.assert_close(outputs["lifetime_ns"], 1.0e9 / expected_total)


def test_anti_stokes_exception_masks_work() -> None:
    values = torch.tensor(
        [
            [400.0, 500.0, 0.2, 3.0, 4.0],
            [600.0, 500.0, 0.2, 3.0, 4.0],
            [610.0, 500.0, 0.2, 3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    mask = torch.ones_like(values, dtype=torch.bool)
    verified = torch.tensor([False, True, False])
    stokes_masks = build_stokes_masks(values, mask, TARGETS, verified)

    assert stokes_masks["negative_measured_stokes"].tolist() == [False, True, True]
    assert stokes_masks["anti_stokes_exception"].tolist() == [False, True, False]
    assert stokes_masks["ordinary_stokes_supervision"].tolist() == [True, False, True]


def test_gradients_pass_through_every_derived_equation() -> None:
    model = PhysicsConstrainedOutputHead.build(torch, input_dim=3, variant="complete")
    features = torch.randn(6, 3, requires_grad=True)
    outputs = model(features)
    loss = (
        outputs["absorption_nm"].mean()
        + outputs["emission_nm"].mean()
        + outputs["stokes_energy_ev"].mean()
        + outputs["quantum_yield"].mean()
        + outputs["lifetime_ns"].mean()
        + outputs["log_extinction"].mean()
    )
    loss.backward()

    assert features.grad is not None
    assert bool(torch.isfinite(features.grad).all())
    assert any(parameter.grad is not None and float(parameter.grad.abs().sum()) > 0 for parameter in model.parameters())


def test_missing_targets_do_not_produce_artificial_supervision() -> None:
    model = PhysicsConstrainedOutputHead.build(torch, input_dim=2, variant="complete")
    outputs = model(torch.randn(3, 2))
    target_values = torch.zeros(3, len(TARGETS), dtype=torch.float32)
    target_mask = torch.zeros_like(target_values, dtype=torch.bool)

    loss = masked_physics_loss(outputs, target_values, target_mask, TARGETS)
    assert float(loss.detach()) == 0.0


def test_extreme_numerical_inputs_do_not_overflow() -> None:
    outputs = derive_rates(torch.tensor([1000.0, -1000.0]), torch.tensor([-1000.0, 1000.0]))

    assert bool(torch.isfinite(outputs["quantum_yield"]).all())
    assert bool(torch.isfinite(outputs["lifetime_ns"]).all())
    assert bool((outputs["lifetime_ns"] > 0).all())


def test_checkpoint_migration_from_three_target_baseline_is_explicit() -> None:
    model = PhysicsConstrainedOutputHead.build(torch, input_dim=5, variant="complete")
    checkpoint = {
        "schema_version": "fluorcast_uniprop_head_smoke_v1",
        "model_state_dict": {
            "projection.weight": torch.ones_like(model.projection.weight),
            "projection.bias": torch.ones_like(model.projection.bias),
            "heads.weight": torch.ones(3, 7),
        },
    }

    report = migrate_three_target_checkpoint(checkpoint, model)

    assert report["migration"] == "three_target_baseline_to_physics_constrained"
    assert report["migrated_tensors"] == ["projection.weight", "projection.bias"]
    assert report["skipped_tensors"] == ["heads.weight"]
    assert "log_k_r" in report["new_physics_outputs"]


def test_physics_variants_are_registered_for_ablation_matrix() -> None:
    assert PHYSICS_MODEL_VARIANTS == (
        "uniprop_independent_heads",
        "uniprop_wavelength_constrained_heads",
        "uniprop_rate_constrained_heads",
        "uniprop_complete_physics_constrained",
    )
