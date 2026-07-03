"""Deterministic aggregation and reporting for ChemFluor predictions."""

from .explanation import collect_confidence_reasons, confidence_label, summarize_prediction
from .report import (
    build_hybrid_report,
    load_prediction_table,
    render_report_markdown,
    write_report_json,
)
from .meta_features import (
    add_wide_feature_aliases,
    build_meta_feature_row,
    build_meta_feature_table,
    build_meta_features,
)

__all__ = [
    "build_hybrid_report",
    "add_wide_feature_aliases",
    "build_meta_feature_row",
    "build_meta_feature_table",
    "build_meta_features",
    "collect_confidence_reasons",
    "confidence_label",
    "load_prediction_table",
    "render_report_markdown",
    "summarize_prediction",
    "write_report_json",
]
