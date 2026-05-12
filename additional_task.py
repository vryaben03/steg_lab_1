"""
Дополнительное задание: доверительные интервалы для PSNR и дисперсии битовых плоскостей
"""

import os
from dotenv import load_dotenv
from stego_utils import calculate_all_confidence_intervals, plot_confidence_intervals_results
from config import RESULTS_PATH
load_dotenv()

# ========== КОНФИГУРАЦИЯ ==========
BOSSbase_PATH = os.getenv("BOSSbase_PATH")
MEDICAL_COMBINED_PATH = os.getenv("MEDICAL_COMBINED_PATH")
TEXTURE_PATH = os.getenv("TEXTURE_PATH")

# Доступные наборы
DATASETS = {}

if BOSSbase_PATH and os.path.exists(BOSSbase_PATH):
    DATASETS["BOSSbase"] = BOSSbase_PATH

if MEDICAL_COMBINED_PATH and os.path.exists(MEDICAL_COMBINED_PATH):
    DATASETS["Medical"] = MEDICAL_COMBINED_PATH

if TEXTURE_PATH and os.path.exists(TEXTURE_PATH):
    DATASETS["Textures"] = TEXTURE_PATH


def main():
    print("="*60)
    print("ДОПОЛНИТЕЛЬНОЕ ЗАДАНИЕ")
    print("Доверительные интервалы для PSNR и дисперсии")
    print("="*60)
    
    if not DATASETS:
        print("❌ Нет доступных наборов изображений!")
        return
    
    print("\nАнализируемые наборы:")
    for name in DATASETS.keys():
        print(f"  - {name}")
    
    # Запускаем расчёт
    results = calculate_all_confidence_intervals(DATASETS, k_values=[1, 2, 3], max_images=100)
    
    # Сохраняем результаты в CSV
    import pandas as pd
    
    # PSNR результаты
    psnr_rows = []
    for dataset_name, psnr_data in results['psnr'].items():
        for k, data in psnr_data.items():
            psnr_rows.append({
                'dataset': dataset_name,
                'k': k,
                'mean_psnr': data['mean'],
                'ci_lower': data['ci_lower'],
                'ci_upper': data['ci_upper']
            })
    
    psnr_df = pd.DataFrame(psnr_rows)
    psnr_df.to_csv(os.path.join(RESULTS_PATH, "confidence_psnr.csv"), index=False)
    
    # Дисперсия результаты
    var_rows = []
    for dataset_name, var_data in results['variance'].items():
        for k, data in var_data.items():
            var_rows.append({
                'dataset': dataset_name,
                'k': k,
                'mean_variance': data['mean'],
                'ci_lower': data['ci_lower'],
                'ci_upper': data['ci_upper']
            })
    
    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(os.path.join(RESULTS_PATH, "confidence_variance.csv"), index=False)
    
    print(f"\n✅ Результаты сохранены:")
    print(f"   PSNR: {os.path.join(RESULTS_PATH, 'confidence_psnr.csv')}")
    print(f"   Дисперсия: {os.path.join(RESULTS_PATH, 'confidence_variance.csv')}")
    
    # Строим графики
    plot_confidence_intervals_results(results, save_path=os.path.join(RESULTS_PATH, "confidence_intervals.png"))
    
    print("\n" + "="*60)
    print("✅ ДОПОЛНИТЕЛЬНОЕ ЗАДАНИЕ ВЫПОЛНЕНО")
    print("="*60)


if __name__ == "__main__":
    main()