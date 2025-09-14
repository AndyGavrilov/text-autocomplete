"""
Оценка качества и сравнение различных моделей автодополнения текста.
"""

import torch
from typing import Dict, List, Any, Tuple
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.eval_lstm import evaluate_rouge
from src.configs import TOKENIZER
from src.utils.utils import preprocess_text
from .parameter_tuning import extract_rouge_score

def compare_models(lstm_model, baseline_model, val_loader: DataLoader, 
                  val_texts: List[str], device: torch.device, 
                  best_baseline_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Сравнение LSTM модели с baseline моделью.
    
    Args:
        lstm_model: Обученная LSTM модель.
        baseline_model: Baseline модель для сравнения.
        val_loader: DataLoader с валидационными данными для LSTM.
        val_texts: Список валидационных текстов для baseline.
        device: Устройство для вычислений.
        best_baseline_config: Лучшая конфигурация параметров для baseline.
        
    Returns:
        Словарь с результатами сравнения.
    """
    print("\n=== Сравнение LSTM с baseline моделью ===")
    
    # Оценка LSTM модели
    print("Оценка LSTM модели...")
    lstm_rouge_scores = evaluate_rouge(
        model=lstm_model,
        dataloader=val_loader,
        tokenizer=TOKENIZER,
        device=device
    )
    
    # Извлекаем LSTM ROUGE scores
    lstm_rouge1 = extract_rouge_score(lstm_rouge_scores['rouge1'])
    lstm_rouge2 = extract_rouge_score(lstm_rouge_scores['rouge2'])
    lstm_rouge_l = extract_rouge_score(lstm_rouge_scores['rougeL'])
    
    # Оценка baseline модели
    print("Оценка baseline модели...")
    if best_baseline_config:
        baseline_rouge_scores = baseline_model.evaluate_rouge(
            val_texts,
            max_samples=2000,
            max_new_tokens=20,
            do_sample=True,
            **{k: v for k, v in best_baseline_config.items() if k != 'name'}
        )
    else:
        baseline_rouge_scores = baseline_model.evaluate_rouge(
            val_texts,
            max_samples=2000,
            max_new_tokens=20
        )
    
    # Извлекаем baseline ROUGE scores
    baseline_rouge1 = extract_rouge_score(baseline_rouge_scores['rouge1'])
    baseline_rouge2 = extract_rouge_score(baseline_rouge_scores['rouge2'])
    baseline_rouge_l = extract_rouge_score(baseline_rouge_scores['rougeL'])
    
    # Выводим результаты сравнения
    print(f"\nСравнение результатов:")
    print(f"{'Модель':<15} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10}")
    print("-" * 50)
    print(f"{'LSTM':<15} {lstm_rouge1:.4f}     {lstm_rouge2:.4f}     {lstm_rouge_l:.4f}")
    print(f"{'Baseline':<15} {baseline_rouge1:.4f}     {baseline_rouge2:.4f}     {baseline_rouge_l:.4f}")
    
    # Определяем лучшую модель
    if baseline_rouge_l > lstm_rouge_l:
        best_model = "Baseline"
        print(f"\nBaseline показывает лучшие результаты (ROUGE-L: {baseline_rouge_l:.4f} vs {lstm_rouge_l:.4f})")
    else:
        best_model = "LSTM"
        print(f"\nLSTM показывает лучшие результаты (ROUGE-L: {lstm_rouge_l:.4f} vs {baseline_rouge_l:.4f})")
    
    return {
        'lstm_rouge1': lstm_rouge1,
        'lstm_rouge2': lstm_rouge2,
        'lstm_rouge_l': lstm_rouge_l,
        'baseline_rouge1': baseline_rouge1,
        'baseline_rouge2': baseline_rouge2,
        'baseline_rouge_l': baseline_rouge_l,
        'best_model': best_model
    }

def generate_examples(lstm_model, baseline_model, tokenizer: AutoTokenizer, 
                     device: torch.device, sample_texts: List[str],
                     best_baseline_config: Dict[str, Any] = None) -> List[Dict[str, str]]:
    """
    Генерация примеров для демонстрации работы моделей.
    
    Args:
        lstm_model: LSTM модель.
        baseline_model: Baseline модель.
        tokenizer: Токенизатор.
        device: Устройство для вычислений.
        sample_texts: Список примеров текстов для генерации.
        best_baseline_config: Лучшая конфигурация параметров для baseline.
        
    Returns:
        Список словарей с примерами генерации.
    """
    print("\n=== Примеры генерации ===")
    
    examples = []
    
    for i, text in enumerate(sample_texts):
        print(f"\n--- Пример {i+1} ---")
        print(f"Исходный текст: {text[:100]}...")
        
        try:
            # Предобработка текста
            processed_text = preprocess_text(text, tokenizer=None)
            tokens = tokenizer.tokenize(processed_text)
            
            if len(tokens) < 4:
                print("Текст слишком короткий, пропускаем")
                continue
                
            # Разбиваем на контекст (75%) и цель (25%)
            cutoff = max(1, int(len(tokens) * 0.75))
            context_tokens = tokens[:cutoff]
            target_tokens = tokens[cutoff:]
            
            context_text = tokenizer.convert_tokens_to_string(context_tokens)
            target_text = tokenizer.convert_tokens_to_string(target_tokens)
            
            print(f"Контекст: {context_text}")
            print(f"Ожидаемое продолжение: {target_text}")
            
            # Генерация с LSTM
            try:
                input_ids = tokenizer.encode(context_text, return_tensors="pt").to(device)
                lstm_generated = lstm_model.generate(
                    input_ids, tokenizer, 
                    max_new_tokens=len(target_tokens) + 5, 
                    top_k=20
                )
                # Извлекаем только сгенерированную часть
                lstm_continuation = lstm_generated[len(context_text):].strip()
                print(f"LSTM продолжение: {lstm_continuation}")
            except Exception as e:
                print(f"Ошибка LSTM: {e}")
                lstm_continuation = "Ошибка генерации"
            
            # Генерация с baseline
            try:
                if best_baseline_config:
                    baseline_generated = baseline_model.generate(
                        context_text,
                        max_new_tokens=len(target_tokens) + 5,
                        **{k: v for k, v in best_baseline_config.items() if k != 'name'}
                    )
                else:
                    baseline_generated = baseline_model.generate(
                        context_text, 
                        max_new_tokens=len(target_tokens) + 5
                    ).strip()
                
                # Извлекаем только сгенерированную часть
                baseline_continuation = baseline_generated[len(context_text):].strip()
                print(f"Baseline продолжение: {baseline_continuation}")
            except Exception as e:
                print(f"Ошибка Baseline: {e}")
                baseline_continuation = "Ошибка генерации"
            
            examples.append({
                'original': text,
                'context': context_text,
                'target': target_text,
                'lstm': lstm_continuation,
                'baseline': baseline_continuation
            })
            
        except Exception as e:
            print(f"Критическая ошибка при обработке примера {i+1}: {e}")
            print("Пропускаем этот пример")
            continue
    
    return examples

def create_comparison_report(results: Dict[str, Any], examples: List[Dict[str, str]]) -> str:
    """
    Создание текстового отчета о сравнении моделей.
    
    Args:
        results: Результаты сравнения моделей.
        examples: Примеры генерации.
        
    Returns:
        Текстовый отчет.
    """
    report = []
    report.append("=" * 60)
    report.append("�� ОТЧЕТ О СРАВНЕНИИ МОДЕЛЕЙ АВТОДОПОЛНЕНИЯ ТЕКСТА")
    report.append("=" * 60)
    
    report.append(f"\n🏆 Лучшая модель: {results['best_model']}")
    
    report.append(f"\n�� Метрики качества:")
    report.append(f"{'Модель':<15} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10}")
    report.append("-" * 50)
    report.append(f"{'LSTM':<15} {results['lstm_rouge1']:.4f}     {results['lstm_rouge2']:.4f}     {results['lstm_rouge_l']:.4f}")
    report.append(f"{'Baseline':<15} {results['baseline_rouge1']:.4f}     {results['baseline_rouge2']:.4f}     {results['baseline_rouge_l']:.4f}")
    
    report.append(f"\n📝 Примеры генерации:")
    for i, example in enumerate(examples[:5]):  # Показываем первые 5 примеров
        report.append(f"\n--- Пример {i+1} ---")
        report.append(f"Исходный: {example['original'][:50]}...")
        report.append(f"LSTM: {example['lstm']}")
        report.append(f"Baseline: {example['baseline']}")
    
    return "\n".join(report)