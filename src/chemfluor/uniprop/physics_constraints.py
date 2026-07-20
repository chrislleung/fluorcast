"""Physics-constrained UniProp output heads and photophysical equations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


HC_EV_NM = 1239.8419843320026
SECONDS_PER_NS = 1.0e-9
PHYSICS_SCHEMA_VERSION = "fluorcast_uniprop_physics_constraints_v1"
PHYSICS_TARGETS = (
    "absorption_nm",
    "emission_nm",
    "quantum_yield",
    "lifetime_ns",
    "log_extinction",
)
PHYSICS_MODEL_VARIANTS = (
    "uniprop_independent_heads",
    "uniprop_wavelength_constrained_heads",
    "uniprop_rate_constrained_heads",
    "uniprop_complete_physics_constrained",
)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for UniProp physics constraints.") from exc
    return torch


def wavelength_nm_to_energy_ev(wavelength_nm: Any) -> Any:
    """Convert wavelength in nm to photon energy in eV without rounding."""
    return HC_EV_NM / wavelength_nm


def energy_ev_to_wavelength_nm(energy_ev: Any) -> Any:
    """Convert photon energy in eV to wavelength in nm without rounding."""
    return HC_EV_NM / energy_ev


def ordinary_stokes_energy_ev(absorption_energy_ev: Any, emission_energy_ev: Any) -> Any:
    """Return ordinary fluorescence Stokes energy, E_abs - E_em."""
    return absorption_energy_ev - emission_energy_ev


@dataclass(frozen=True)
class AntiStokesPolicy:
    """Policy for rows whose measured energies imply anti-Stokes fluorescence."""

    require_verified_flag: bool = True


def build_stokes_masks(
    target_values: Any,
    target_mask: Any,
    target_names: tuple[str, ...],
    verified_anti_stokes_mask: Any | None = None,
    policy: AntiStokesPolicy = AntiStokesPolicy(),
) -> dict[str, Any]:
    """Build masks that keep verified anti-Stokes rows out of ordinary-Stokes supervision."""
    torch = _require_torch()
    abs_index = target_names.index("absorption_nm")
    em_index = target_names.index("emission_nm")
    has_pair = target_mask[:, abs_index] & target_mask[:, em_index]
    absorption_energy = wavelength_nm_to_energy_ev(target_values[:, abs_index])
    emission_energy = wavelength_nm_to_energy_ev(target_values[:, em_index])
    measured_delta = ordinary_stokes_energy_ev(absorption_energy, emission_energy)
    negative_measured_delta = has_pair & (measured_delta < 0)
    if verified_anti_stokes_mask is None:
        verified = torch.zeros_like(has_pair, dtype=torch.bool)
    else:
        verified = verified_anti_stokes_mask.to(dtype=torch.bool, device=has_pair.device)
    if policy.require_verified_flag:
        anti_stokes_exception = negative_measured_delta & verified
    else:
        anti_stokes_exception = negative_measured_delta
    return {
        "has_absorption_emission_pair": has_pair,
        "negative_measured_stokes": negative_measured_delta,
        "anti_stokes_exception": anti_stokes_exception,
        "ordinary_stokes_supervision": has_pair & ~anti_stokes_exception,
    }


def derive_rates(log_k_r: Any, log_k_nr: Any) -> dict[str, Any]:
    """Derive quantum yield and fluorescence lifetime from log rates."""
    torch = _require_torch()
    log_total_rate = torch.logaddexp(log_k_r, log_k_nr)
    quantum_yield = torch.exp(log_k_r - log_total_rate)
    lifetime_ns = torch.exp(torch.clamp(-log_total_rate, min=-80.0, max=80.0)) / SECONDS_PER_NS
    return {
        "log_total_rate": log_total_rate,
        "quantum_yield": quantum_yield,
        "lifetime_ns": lifetime_ns,
    }


def physics_consistency_metrics(outputs: dict[str, Any], atol: float = 1.0e-6) -> dict[str, float]:
    """Compute numeric consistency diagnostics from already-derived tensors."""
    torch = _require_torch()
    with torch.no_grad():
        stokes = outputs["absorption_energy_ev"] - outputs["emission_energy_ev"]
        rate_values = derive_rates(outputs["log_k_r"], outputs["log_k_nr"])
        return {
            "ordinary_stokes_violation_count": float((outputs["stokes_energy_ev"] < -atol).sum().item()),
            "quantum_yield_bounds_violation_count": float(
                ((outputs["quantum_yield"] < -atol) | (outputs["quantum_yield"] > 1.0 + atol)).sum().item()
            ),
            "lifetime_positive_violation_count": float((outputs["lifetime_ns"] <= 0).sum().item()),
            "max_stokes_equation_error": float((stokes - outputs["stokes_energy_ev"]).abs().max().item()),
            "max_quantum_yield_rate_error": float((rate_values["quantum_yield"] - outputs["quantum_yield"]).abs().max().item()),
            "max_lifetime_rate_error": float((rate_values["lifetime_ns"] - outputs["lifetime_ns"]).abs().max().item()),
        }


class PhysicsConstrainedOutputHead:
    """Small differentiable output model for independent and constrained ablations."""

    @staticmethod
    def build(torch: Any, input_dim: int, variant: str = "complete_physics_constrained") -> Any:
        nn = torch.nn
        canonical_variant = {
            "independent": "independent_heads",
            "wavelength": "wavelength_constrained_heads",
            "rate": "rate_constrained_heads",
            "complete": "complete_physics_constrained",
        }.get(variant, variant)
        if canonical_variant not in {
            "independent_heads",
            "wavelength_constrained_heads",
            "rate_constrained_heads",
            "complete_physics_constrained",
        }:
            raise ValueError(f"Unknown physics head variant: {variant}")

        class Head(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.variant = canonical_variant
                self.projection = nn.Linear(input_dim, 5)

            def forward(self, features: Any) -> dict[str, Any]:
                raw = self.projection(features)
                emission_energy = torch.nn.functional.softplus(raw[:, 0]) + 1.0e-6
                stokes_energy = torch.nn.functional.softplus(raw[:, 1])
                if self.variant == "independent_heads":
                    absorption_energy = torch.nn.functional.softplus(raw[:, 1]) + 1.0e-6
                    stokes_energy = absorption_energy - emission_energy
                else:
                    absorption_energy = emission_energy + stokes_energy
                rates = derive_rates(raw[:, 2], raw[:, 3])
                return {
                    "absorption_energy_ev": absorption_energy,
                    "emission_energy_ev": emission_energy,
                    "stokes_energy_ev": stokes_energy,
                    "absorption_nm": energy_ev_to_wavelength_nm(absorption_energy),
                    "emission_nm": energy_ev_to_wavelength_nm(emission_energy),
                    "log_k_r": raw[:, 2],
                    "log_k_nr": raw[:, 3],
                    "quantum_yield": rates["quantum_yield"],
                    "lifetime_ns": rates["lifetime_ns"],
                    "log_extinction": raw[:, 4],
                }

        return Head()


def masked_physics_loss(
    outputs: dict[str, Any],
    target_values: Any,
    target_mask: Any,
    target_names: tuple[str, ...],
    verified_anti_stokes_mask: Any | None = None,
) -> Any:
    """Masked multitask MSE plus ordinary-Stokes supervision where policy allows it."""
    torch = _require_torch()
    losses = []
    for index, target in enumerate(target_names):
        if target not in outputs:
            continue
        mask = target_mask[:, index]
        if bool(mask.any().item()):
            diff = outputs[target][mask] - target_values[:, index][mask]
            losses.append((diff * diff).mean())
    masks = build_stokes_masks(target_values, target_mask, target_names, verified_anti_stokes_mask)
    ordinary_mask = masks["ordinary_stokes_supervision"]
    if bool(ordinary_mask.any().item()):
        abs_index = target_names.index("absorption_nm")
        em_index = target_names.index("emission_nm")
        target_stokes = wavelength_nm_to_energy_ev(target_values[:, abs_index]) - wavelength_nm_to_energy_ev(target_values[:, em_index])
        diff = outputs["stokes_energy_ev"][ordinary_mask] - target_stokes[ordinary_mask]
        losses.append((diff * diff).mean())
    if not losses:
        return torch.zeros((), dtype=target_values.dtype, device=target_values.device, requires_grad=True)
    total = torch.stack(losses).mean()
    if not bool(torch.isfinite(total).item()):
        raise FloatingPointError("Physics-constrained loss is NaN or infinite.")
    return total


def migrate_three_target_checkpoint(checkpoint: dict[str, Any], model: Any) -> dict[str, Any]:
    """Explicitly migrate compatible tensors from the three-target baseline checkpoint."""
    if checkpoint.get("schema_version") != "fluorcast_uniprop_head_smoke_v1":
        raise ValueError(f"Unsupported source checkpoint schema: {checkpoint.get('schema_version')}")
    source_state = checkpoint.get("model_state_dict", {})
    target_state = model.state_dict()
    migrated: list[str] = []
    skipped: list[str] = []
    for name, tensor in source_state.items():
        if name in target_state and tuple(target_state[name].shape) == tuple(tensor.shape):
            target_state[name] = tensor
            migrated.append(name)
        else:
            skipped.append(name)
    model.load_state_dict(target_state)
    return {
        "schema_version": PHYSICS_SCHEMA_VERSION,
        "source_schema_version": checkpoint["schema_version"],
        "migration": "three_target_baseline_to_physics_constrained",
        "migrated_tensors": migrated,
        "skipped_tensors": skipped,
        "new_physics_outputs": ["stokes_energy_ev", "log_k_r", "log_k_nr", "lifetime_ns", "log_extinction"],
    }
