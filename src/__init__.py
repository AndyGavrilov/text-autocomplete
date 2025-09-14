"""
LSTM Text Autocomplete Package

Пакет для обучения и использования LSTM модели автодополнения текста.
"""

from .lstm_model import LSTMLanguageModel
from .lstm_train import train_model
from .eval_lstm import evaluate_rouge
from .next_token_dataset import LMTextDataset, collate_fn
from .configs import CONFIG, DATA_PATH, PROJECT_PATH

__version__ = "1.0.0"
__author__ = "Yandex School of Deep Learning team"

__all__ = [
    "LSTMLanguageModel",
    "train_model", 
    "evaluate_rouge",
    "LMTextDataset",
    "collate_fn",
    "CONFIG",
    "DATA_PATH", 
    "PROJECT_PATH"
]
