"""Ensemble methods module."""
from .average import (
    AverageEnsemble,
    WeightedAverageEnsemble,
    StackingEnsemble,
    create_ensemble_from_paths,
    evaluate_ensemble
)

__all__ = [
    "AverageEnsemble",
    "WeightedAverageEnsemble", 
    "StackingEnsemble",
    "create_ensemble_from_paths",
    "evaluate_ensemble"
]