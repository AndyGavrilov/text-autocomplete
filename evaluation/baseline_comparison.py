"""
Сравнение LSTM модели с baseline моделью distilgpt2.
"""

import torch
from typing import Dict, List, Any, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from src.configs import ROUGE
from src.utils.utils import preprocess_text

class DistilGPT2Baseline:
    """Baseline модель distilgpt2 для сравнения с LSTM."""
    
    def __init__(self, device: str = None):
        """
        Инициализация baseline модели distilgpt2.
        
        Args:
            device: Устройство для вычислений. По умолчанию автоматически определяется.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.device = device
        print("Загрузка предобученной модели distilgpt2...")
        
        # Инициализация pipeline для генерации
        self.generator = pipeline(
            "text-generation", 
            model="distilgpt2", 
            device=0 if device == "cuda" else -1
        )
        
        # Загрузка токенизатора и модели
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        
        # Добавляем pad_token если его нет
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Модель загружена успешно!")
    
    def generate(self, text: str, max_new_tokens: int = 20, 
                 do_sample: bool = True, top_k: int = 50, 
                 top_p: float = 0.95, temperature: float = 1.0) -> str:
        """
        Генерация текста с помощью distilgpt2.
        
        Args:
            text: Входной контекст.
            max_new_tokens: Максимальное количество новых токенов для генерации.
            do_sample: Использовать ли сэмплирование.
            top_k: Параметр top-k сэмплирования.
            top_p: Параметр top-p (nucleus) сэмплирования.
            temperature: Температура сэмплирования.
            
        Returns:
            Сгенерированный текст.
        """
        # Подготавливаем параметры генерации
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "pad_token_id": self.tokenizer.eos_token_id,
            "truncation": True
        }
        
        # Добавляем параметры сэмплирования только если используется сэмплирование
        if do_sample:
            generation_kwargs.update({
                "top_k": top_k,
                "top_p": top_p
            })
        
        result = self.generator(text, **generation_kwargs)
        return result[0]["generated_text"]
    
    def evaluate_rouge(self, val_texts: List[str], max_samples: int = 2000, 
                      max_new_tokens: int = 20, do_sample: bool = True, 
                      top_k: int = 50, top_p: float = 0.95, 
                      temperature: float = 1.0) -> Dict[str, float]:
        """
        Оценка качества distilgpt2 с помощью метрики ROUGE.
        
        Args:
            val_texts: Список валидационных текстов (исходные строки).
            max_samples: Максимальное количество примеров для оценки.
            max_new_tokens: Максимальное количество новых токенов для генерации.
            do_sample: Использовать ли сэмплирование.
            top_k: Параметр top-k сэмплирования.
            top_p: Параметр top-p сэмплирования.
            temperature: Температура сэмплирования.
            
        Returns:
            Словарь с метриками ROUGE.
        """
        preds, refs = [], []
        processed = 0
        
        print(f"Оценка distilgpt2 на {min(len(val_texts), max_samples)} примерах...")
        
        for i, text in enumerate(tqdm(val_texts[:max_samples])):
            if processed >= max_samples:
                break
                
            # Проверяем, что text - это строка, а не список токенов
            if isinstance(text, list):
                # Если это список токенов, пропускаем этот пример
                continue
                
            # Предобработка текста - получаем строку, а не токены
            processed_text = preprocess_text(text, tokenizer=None)  # Не используем токенизатор
            
            # Теперь токенизируем с помощью distil_tokenizer
            tokens = self.tokenizer.tokenize(processed_text)
            if len(tokens) < 4:
                continue
                
            # Берем первые 75% как контекст
            cutoff = max(1, int(len(tokens) * 0.75))
            context_tokens = tokens[:cutoff]
            target_tokens = tokens[cutoff:]
            
            if len(target_tokens) == 0:
                continue
                
            # Декодируем контекст
            context_text = self.tokenizer.convert_tokens_to_string(context_tokens)
            target_text = self.tokenizer.convert_tokens_to_string(target_tokens)
            
            # Генерируем продолжение
            try:
                generated_text = self.generate(
                    context_text, 
                    max_new_tokens=len(target_tokens) + 5,  # Генерируем примерно столько же токенов + небольшой запас
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature
                )
                
                # Извлекаем только сгенерированную часть
                generated_continuation = generated_text[len(context_text):].strip()
                
                preds.append(generated_continuation)
                refs.append(target_text)
                processed += 1
                
            except Exception as e:
                print(f"Ошибка при генерации для примера {i}: {e}")
                continue
        
        print(f"Обработано {len(preds)} примеров")
        
        if len(preds) == 0:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        # Вычисляем ROUGE метрики
        rouge_scores = ROUGE.compute(predictions=preds, references=refs)
        return rouge_scores