"""
Модуль для оценки и сравнения моделей автодополнения текста.
"""

from .baseline_comparison import DistilGPT2Baseline
from .parameter_tuning import tune_parameters, extract_rouge_score, test_temperature_sensitivity
from .model_evaluation import compare_models, generate_examples

__all__ = [
    'DistilGPT2Baseline',
    'tune_parameters', 
    'extract_rouge_score',
    'compare_models',
    'generate_examples',
    'test_temperature_sensitivity'
]