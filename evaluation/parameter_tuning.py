"""
Подбор оптимальных параметров генерации для baseline моделей.
"""

from typing import Dict, List, Any, Tuple
import evaluate

def extract_rouge_score(rouge_value) -> float:
    """
    Извлекает числовое значение ROUGE метрики.
    
    Args:
        rouge_value: Значение ROUGE метрики (может быть объектом с .mid.fmeasure или просто числом).
        
    Returns:
        Числовое значение метрики.
    """
    if hasattr(rouge_value, 'mid') and hasattr(rouge_value.mid, 'fmeasure'):
        return rouge_value.mid.fmeasure
    elif hasattr(rouge_value, 'fmeasure'):
        return rouge_value.fmeasure
    else:
        return float(rouge_value)

def tune_parameters(baseline_model, val_texts: List[str], 
                   param_configs: List[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Подбор оптимальных параметров генерации для baseline модели.
    
    Args:
        baseline_model: Baseline модель для тестирования.
        val_texts: Список валидационных текстов.
        param_configs: Список конфигураций параметров для тестирования.
        
    Returns:
        Кортеж (лучшая_конфигурация, все_результаты).
    """
    if param_configs is None:
        param_configs = [
            {"top_k": 20, "top_p": 0.9, "temperature": 0.7, "name": "Консервативная"},
            {"top_k": 50, "top_p": 0.95, "temperature": 1.0, "name": "Стандартная"},
            {"top_k": 100, "top_p": 0.98, "temperature": 1.2, "name": "Креативная"},
            {"top_k": 10, "top_p": 0.8, "temperature": 0.6, "name": "Осторожная"},
            {"top_k": 10, "top_p": 0.8, "temperature": 0.3, "name": "Очень осторожная"},
            {"top_k": 200, "top_p": 0.99, "temperature": 1.5, "name": "Экспериментальная"}
        ]
    
    print("=== Подбор параметров генерации для baseline модели ===")
    
    best_rouge = 0.0
    best_config = None
    results = []
    
    for config in param_configs:
        print(f"\nТестирование конфигурации: {config['name']}")
        print(f"Параметры: top_k={config['top_k']}, top_p={config['top_p']}, temperature={config['temperature']}")
        
        # Оценка с текущими параметрами
        rouge_scores = baseline_model.evaluate_rouge(
            val_texts,
            max_samples=500,  # Меньше примеров для быстрого тестирования
            max_new_tokens=20,
            do_sample=True,
            top_k=config['top_k'],
            top_p=config['top_p'],
            temperature=config['temperature']
        )
        
        # Извлекаем ROUGE scores с помощью универсальной функции
        rouge1 = extract_rouge_score(rouge_scores["rouge1"])
        rouge2 = extract_rouge_score(rouge_scores["rouge2"])
        rouge_l = extract_rouge_score(rouge_scores["rougeL"])
        
        results.append({
            "config": config,
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rouge_l
        })
        
        print(f"ROUGE-1: {rouge1:.4f}")
        print(f"ROUGE-2: {rouge2:.4f}")
        print(f"ROUGE-L: {rouge_l:.4f}")
        
        if rouge_l > best_rouge:
            best_rouge = rouge_l
            best_config = config
    
    print(f"\nЛучшая конфигурация: {best_config['name']}")
    print(f"Лучший ROUGE-L: {best_rouge:.4f}")
    
    return best_config, results

def test_temperature_sensitivity(baseline_model, val_texts: List[str], 
                                best_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Тестирование чувствительности к температуре.
    
    Args:
        baseline_model: Baseline модель.
        val_texts: Список валидационных текстов.
        best_config: Лучшая конфигурация параметров.
        
    Returns:
        Список результатов для разных температур.
    """
    print("\n=== Тестирование чувствительности к температуре ===")
    
    temperatures = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5]
    temp_results = []
    
    for temp in temperatures:
        print(f"Тестирование температуры: {temp}")
        
        temp_rouge = baseline_model.evaluate_rouge(
            val_texts[:300], 
            max_samples=300,
            max_new_tokens=20,
            do_sample=True,
            top_k=best_config['top_k'],
            top_p=best_config['top_p'],
            temperature=temp
        )
        
        temp_rouge_l = extract_rouge_score(temp_rouge['rougeL'])
        temp_results.append({
            'temperature': temp,
            'rouge_l': temp_rouge_l
        })
        
        print(f"Температура {temp}: ROUGE-L = {temp_rouge_l:.4f}")
    
    # Найдем лучшую температуру
    best_temp_result = max(temp_results, key=lambda x: x['rouge_l'])
    print(f"\n Лучшая температура: {best_temp_result['temperature']} (ROUGE-L: {best_temp_result['rouge_l']:.4f})")
    
    return temp_results