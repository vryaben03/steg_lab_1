"""
Исследовательская часть для Задания 2
Адаптивное внедрение ЦВЗ (по локальной дисперсии) на трёх наборах
Расчёт доверительных интервалов для PSNR (α = 0.05)
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from PIL import Image
import matplotlib.pyplot as plt

from watermarking import DigitalWatermark
from config import RESULTS_PATH, DEFAULT_MAX_IMAGES, WATERMARK_BLOCK_SIZE, CONFIDENCE
from stego_utils import confidence_interval

load_dotenv()

# ========== конфиг ==========
BOSSbase_PATH = os.getenv("BOSSbase_PATH")
MEDICAL_COMBINED_PATH = os.getenv("MEDICAL_COMBINED_PATH")
TEXTURE_PATH = os.getenv("TEXTURE_PATH")

# путь к логотипу (создадим тестовый, если нет)
LOGO_PATH = os.getenv("LOGO_PATH")

# Доступные наборы
DATASETS = {}

if BOSSbase_PATH and os.path.exists(BOSSbase_PATH):
    DATASETS["BOSSbase"] = BOSSbase_PATH

if MEDICAL_COMBINED_PATH and os.path.exists(MEDICAL_COMBINED_PATH):
    DATASETS["Medical"] = MEDICAL_COMBINED_PATH

if TEXTURE_PATH and os.path.exists(TEXTURE_PATH):
    DATASETS["Textures"] = TEXTURE_PATH


def create_test_logo():
    """Создаёт тестовый логотип, если его нет"""
    if os.path.exists(LOGO_PATH):
        print(f"Логотип уже существует: {LOGO_PATH}")
        return
    
    # cоздаём логотип 128x128 (чтобы заполнить половину ёмкости 512x512 при дублировании)
    logo_size = 128
    logo = np.zeros((logo_size, logo_size), dtype=np.uint8)
    # рисуем простой паттерн: квадрат и круг
    logo[logo_size//4:3*logo_size//4, logo_size//4:3*logo_size//4] = 255
    
    # круг
    for i in range(logo_size):
        for j in range(logo_size):
            if (i - logo_size//2)**2 + (j - logo_size//2)**2 < (logo_size//3)**2:
                logo[i, j] = 128
    
    Image.fromarray(logo).save(LOGO_PATH)
    print(f"Создан тестовый логотип: {LOGO_PATH} (размер {logo_size}x{logo_size})")


def process_dataset_watermarking(dataset_path, dataset_name, logo_path, output_dir):
    """
    Обрабатывает набор изображений адаптивным методом
    Возвращает список PSNR значений
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # находим PGM файлы
    pgm_files = [f for f in os.listdir(dataset_path) if f.lower().endswith('.pgm')]
    try:
        pgm_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        pgm_files.sort()
    pgm_files = pgm_files[:DEFAULT_MAX_IMAGES]
    
    print(f"\nОбработка набора {dataset_name} ({len(pgm_files)} изображений)...")
    
    psnr_values = []
    results = []
    
    for idx, img_file in enumerate(pgm_files):
        img_path = os.path.join(dataset_path, img_file)
        
        try:
            # создаём объект водяного знака
            watermark = DigitalWatermark(img_path, logo_path)
            
            # выходной путь
            output_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_watermark.png")
            
            # адаптивное внедрение
            metrics = watermark.embed_adaptive(output_path, block_size=WATERMARK_BLOCK_SIZE)
            
            psnr_values.append(metrics['PSNR'])
            results.append({
                'image': img_file,
                'dataset': dataset_name,
                'psnr': metrics['PSNR'],
                'mse': metrics['MSE'],
                'ssim': metrics['SSIM']
            })
            
            if (idx + 1) % 20 == 0:
                print(f"  Обработано {idx+1}/{len(pgm_files)}")
                
        except Exception as e:
            print(f"  Ошибка при обработке {img_file}: {e}")
    
    return psnr_values, results


def main():
    print("="*60)
    print("ИССЛЕДОВАТЕЛЬСКАЯ ЧАСТЬ")
    print("Адаптивное внедрение ЦВЗ по локальной дисперсии")
    print("Доверительные интервалы для PSNR (95%)")
    print("="*60)
    
    if not DATASETS:
        print("Нет доступных наборов изображений!")
        return
    
    # создаём логотип
    create_test_logo()
    
    # папка для результатов
    watermark_results_dir = os.path.join(RESULTS_PATH, "watermarking_adaptive")
    os.makedirs(watermark_results_dir, exist_ok=True)
    
    # собираем результаты по наборам
    all_psnr = {}
    all_results = []
    
    for dataset_name, dataset_path in DATASETS.items():
        print(f"\n{'='*50}")
        print(f"Набор: {dataset_name}")
        print(f"Путь: {dataset_path}")
        
        output_dir = os.path.join(watermark_results_dir, dataset_name)
        psnr_values, results = process_dataset_watermarking(
            dataset_path, dataset_name, LOGO_PATH, output_dir
        )
        
        all_psnr[dataset_name] = psnr_values
        all_results.extend(results)
        
        # сохраняем результаты в CSV
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(watermark_results_dir, f"{dataset_name}_psnr.csv"), index=False)
    
    # считаем доверительные интервалы
    print("\n" + "="*60)
    print("ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ ДЛЯ PSNR (95%)")
    print("="*60)
    
    confidence_data = []
    for dataset_name, psnr_values in all_psnr.items():
        mean, ci_lower, ci_upper = confidence_interval(psnr_values, CONFIDENCE)
        confidence_data.append({
            'dataset': dataset_name,
            'mean_psnr': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std': np.std(psnr_values),
            'n': len(psnr_values)
        })
        print(f"\n{dataset_name}:")
        print(f"  Среднее PSNR = {mean:.2f} дБ")
        print(f"  95% доверительный интервал = [{ci_lower:.2f}, {ci_upper:.2f}]")
        print(f"  Стандартное отклонение = {np.std(psnr_values):.2f}")
    
    # сохраняем сводку
    summary_df = pd.DataFrame(confidence_data)
    summary_df.to_csv(os.path.join(watermark_results_dir, "confidence_intervals.csv"), index=False)
    
    # строим график
    plot_confidence_intervals(confidence_data, watermark_results_dir)
    
    print("\n" + "="*60)
    print(f"Результаты сохранены в: {watermark_results_dir}")
    print("="*60)


def plot_confidence_intervals(confidence_data, output_dir):
    """Строит график доверительных интервалов для PSNR"""
    datasets = [d['dataset'] for d in confidence_data]
    means = [d['mean_psnr'] for d in confidence_data]
    errors_lower = [d['mean_psnr'] - d['ci_lower'] for d in confidence_data]
    errors_upper = [d['ci_upper'] - d['mean_psnr'] for d in confidence_data]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(datasets, means, color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.7)
    plt.errorbar(datasets, means, yerr=[errors_lower, errors_upper], 
                fmt='none', color='black', capsize=10, capthick=2, elinewidth=2)
    
    plt.xlabel('Тип контейнера', fontsize=12)
    plt.ylabel('PSNR (дБ)', fontsize=12)
    plt.title('Доверительные интервалы для PSNR (95%)\nАдаптивное внедрение ЦВЗ по локальной дисперсии', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # добавляем значения на столбцы
    for bar, mean, err_low, err_up in zip(bars, means, errors_lower, errors_upper):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_intervals_psnr.png"), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nГрафик сохранён: {os.path.join(output_dir, 'confidence_intervals_psnr.png')}")


if __name__ == "__main__":
    main()