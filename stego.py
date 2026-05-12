"""
Основной файл для выполнения исследовательской части по заданию
1. Для 5 изображений из набора визуализирует 8 битовых плоскостей
2. Для одного изображения внедряет сообщение в k=1,2,3
3. Считает PSNR, MSE, SSIM для каждого k
4. Строит гистограммы для исходного и стего-изображений
"""

import os
import pandas as pd
from dotenv import load_dotenv
from lsb_steganography import LSBSteganography
from stego_utils import (
    compute_metrics, plot_histogram, visualize_all_bit_planes_for_image
)
from config import *
load_dotenv()


# ========== КОНФИГУРАЦИЯ ==========
BOSSbase_PATH = os.getenv("BOSSbase_PATH")
MEDICAL_GLIOMA_PATH = os.getenv("MEDICAL_GLIOMA_PATH")
MEDICAL_MENIN_PATH = os.getenv("MEDICAL_MENIN_PATH")
MEDICAL_TUMOR_PATH = os.getenv("MEDICAL_TUMOR_PATH")
MEDICAL_COMBINED_PATH = os.getenv("MEDICAL_COMBINED_PATH")
TEXTURE_PATH = os.getenv("TEXTURE_PATH")


TEST_MESSAGE_PATH=os.getenv("TEST_MESSAGE_PATH")
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

# Папки для результатов
for path in [RESULTS_PATH, BIT_PLANES_PATH, STEGO_IMAGES_PATH, 
             HISTOGRAMS_PATH, METRICS_PATH]:
    os.makedirs(path, exist_ok=True)


def create_test_message(filepath=TEST_MESSAGE_PATH, target_kb=MESSAGE_SIZE_KB):
    """Создает тестовый файл сообщения"""
    if os.path.exists(filepath):
        print(f"Файл сообщения уже существует: {filepath}")
        return
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for i in range(target_kb * 14):
            f.write(f"Строка {i:04d}: Это тестовое сообщение для LSB стеганографии. ")
    
    actual_size = os.path.getsize(filepath) / 1024
    print(f"Создан файл сообщения: {filepath}")
    print(f"Размер: {actual_size:.1f} КБ")


def analyze_one_image(dataset_path, dataset_name, image_index=0):
    """
    Полный анализ одного изображения из набора по заданию:
    1. Показывает 8 битовых плоскостей (уже сделано в visualize_all_bit_planes_for_image)
    2. Внедряет сообщение в k=1,2,3
    3. ИЗВЛЕКАЕТ сообщение обратно (проверка)
    4. Считает PSNR, MSE, SSIM
    5. Строит гистограммы для исходного и всех 3 стего-изображений
    6. Сохраняет все результаты
    """
    print("\n" + "="*60)
    print(f"АНАЛИЗ ИЗОБРАЖЕНИЯ ДЛЯ НАБОРА: {dataset_name}")
    print("="*60)
    
    # Находим PGM изображения
    pgm_files = [f for f in os.listdir(dataset_path) if f.lower().endswith('.pgm')]
    if not pgm_files:
        print(f"PGM файлы не найдены в {dataset_path}!")
        return None
    
    # Сортируем для предсказуемого порядка
    try:
        pgm_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        pgm_files.sort()
    
    if image_index >= len(pgm_files):
        image_index = 0
    
    image_file = pgm_files[image_index]
    image_path = os.path.join(dataset_path, image_file)
    print(f"\nИсходное изображение: {image_file}")
    
    # Создаем папки для результатов этого набора
    dataset_bit_planes = os.path.join(BIT_PLANES_PATH, dataset_name)
    dataset_stego = os.path.join(STEGO_IMAGES_PATH, dataset_name)
    dataset_histograms = os.path.join(HISTOGRAMS_PATH, dataset_name)
    os.makedirs(dataset_bit_planes, exist_ok=True)
    os.makedirs(dataset_stego, exist_ok=True)
    os.makedirs(dataset_histograms, exist_ok=True)
    
    # Создаем тестовое сообщение
    message_path = os.path.join(RESULTS_PATH, f"test_message_{dataset_name}.txt")
    if not os.path.exists(message_path):
        create_test_message(message_path, target_kb=MESSAGE_SIZE_KB)
    else:
        print(f"✓ Используем существующий файл сообщения: {message_path}")
    
    # ========== 1. Визуализация битовых плоскостей для исходного изображения ==========
    print("\n--- Шаг 1: Визуализация 8 битовых плоскостей ---")
    planes_path = os.path.join(dataset_bit_planes, f"{image_file}_all_planes.png")
    visualize_all_bit_planes_for_image(image_path, output_path=planes_path)
    
    # ========== 2. Внедрение в k=1,2,3 и сбор метрик ==========
    print("\n--- Шаг 2: Внедрение сообщения в k=1, 2, 3 ---")
    original = LSBSteganography(image_path)
    results = []
    
    # Прочитаем исходное сообщение для сравнения
    with open(message_path, 'r', encoding='utf-8') as f:
        original_message = f.read()
    print(f"\nИсходное сообщение (первые 200 символов): {original_message[:200]}...")
    
    for k in DEFAULT_K_VALUES:
        print(f"\n>>> Внедрение в битовую плоскость k={k}")
        
        stego_filename = f"{os.path.splitext(image_file)[0]}_k{k}.png"
        stego_path = os.path.join(dataset_stego, stego_filename)
        
        # Внедряем
        original.embed_message(message_path, k=k, output_path=stego_path, verbose=False)
        
        # ========== Шаг 3: ИЗВЛЕКАЕМ сообщение ==========
        print(f"\n--- Извлечение сообщения для k={k} ---")
        stego_image = LSBSteganography(stego_path)
        extracted_message = stego_image.extract_message(k=k, verbose=True)
        
        # Проверяем, совпадает ли извлечённое сообщение с исходным
        
        print(f"  Длина исходного: {len(original_message)} символов")
        print(f"  Длина извлечённого: {len(extracted_message)} символов")
        # Покажем первые 100 символов для сравнения
        print(f"  Исходное (первые 100): {original_message[:100]}")
        print(f"  Извлечённое (первые 100): {extracted_message[:100]}")
        
        # Считаем метрики качества
        metrics = compute_metrics(image_path, stego_path)
        metrics['k'] = k
        metrics['image'] = image_file
        metrics['dataset'] = dataset_name
        metrics['extraction_ok'] = (extracted_message == original_message)
        results.append(metrics)
        
        print(f"\n  PSNR: {metrics['PSNR']:.2f} dB")
        print(f"  MSE: {metrics['MSE']:.4f}")
        print(f"  SSIM: {metrics['SSIM']:.4f}")
        
        # Строим гистограммы (исходная и стего)
        print(f"\n--- Гистограммы для k={k} ---")
        plot_histogram(image_path, 
                       title=f"Исходное: {image_file} ({dataset_name})",
                       save_path=os.path.join(dataset_histograms, f"{image_file}_original_hist.png"))
        plot_histogram(stego_path, 
                       title=f"Стего k={k}: {stego_filename} ({dataset_name})",
                       save_path=os.path.join(dataset_histograms, f"{image_file}_k{k}_hist.png"))
    
    # Сохраняем метрики в CSV
    results_df = pd.DataFrame(results)
    metrics_path = os.path.join(METRICS_PATH, f"analysis_{dataset_name}.csv")
    results_df.to_csv(metrics_path, index=False)
    print(f"\n✅ Метрики сохранены в: {metrics_path}")
    
    return results_df


def analyze_dataset(dataset_path, dataset_name, num_images_for_planes=5):
    """
    Полный анализ набора данных по заданию:
    1. Для num_images_for_planes изображений визуализирует 8 битовых плоскостей
    2. Для первого изображения внедряет в k=1,2,3 и считает метрики
    """
    print("\n" + "="*60)
    print(f"АНАЛИЗ НАБОРА: {dataset_name}")
    print("="*60)
    
    # Проверяем существование папки
    if not os.path.exists(dataset_path):
        print(f"❌ Папка не найдена: {dataset_path}")
        return None
    
    # Находим PGM изображения
    pgm_files = [f for f in os.listdir(dataset_path) if f.lower().endswith('.pgm')]
    if not pgm_files:
        print(f"❌ PGM файлы не найдены в {dataset_path}")
        return None
    
    # ✅ ДОБАВЛЯЕМ СОРТИРОВКУ — берём первые 5 по числовому порядку
    try:
        # Пробуем сортировать по числовому значению (например, "1.pgm" → 1)
        pgm_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        # Если не числа — обычная лексикографическая сортировка
        pgm_files.sort()
    
    print(f"\n📁 Найдено {len(pgm_files)} PGM файлов")
    print(f"📁 Первые {num_images_for_planes} для визуализации: {pgm_files[:num_images_for_planes]}")
    
    # ========== Шаг a: Визуализация битовых плоскостей для 5 изображений ==========
    print(f"\n{'='*50}")
    print(f"ШАГ a: Визуализация битовых плоскостей для {num_images_for_planes} изображений")
    print(f"{'='*50}")
    
    for i in range(min(num_images_for_planes, len(pgm_files))):
        image_file = pgm_files[i]
        image_path = os.path.join(dataset_path, image_file)
        print(f"\nОбработка {i+1}/{min(num_images_for_planes, len(pgm_files))}: {image_file}")
        
        # Создаём папку для битовых плоскостей этого набора
        planes_dir = os.path.join(BIT_PLANES_PATH, dataset_name)
        os.makedirs(planes_dir, exist_ok=True)
        
        output_path = os.path.join(planes_dir, f"{os.path.splitext(image_file)[0]}_planes.png")
        visualize_all_bit_planes_for_image(image_path, output_path=output_path)
    
    # ========== Шаг c: Внедрение в k=1,2,3 для одного изображения ==========
    print(f"\n{'='*50}")
    print(f"ШАГ c: Внедрение сообщения в k=1, 2, 3 для одного изображения")
    print(f"{'='*50}")
    
    results = analyze_one_image(dataset_path, dataset_name, image_index=0)
    
    return results


def visualize_all_bit_planes_for_image(image_path, output_path=None):
    """Вспомогательная функция для визуализации всех 8 плоскостей одного изображения"""
    stego = LSBSteganography(image_path)
    stego.visualize_bit_planes(output_path=output_path)


def plot_histograms_for_image(original_path, stego_paths, dataset_name, image_name):
    """Строит гистограммы для оригинального и стего-изображений"""
    dataset_histograms = os.path.join(HISTOGRAMS_PATH, dataset_name)
    os.makedirs(dataset_histograms, exist_ok=True)
    
    # Оригинал
    plot_histogram(original_path, 
                   title=f"Исходное: {image_name} ({dataset_name})",
                   save_path=os.path.join(dataset_histograms, f"{image_name}_original_hist.png"))
    
    # Стего для каждого k
    for k, stego_path in stego_paths.items():
        plot_histogram(stego_path, 
                       title=f"Стего k={k}: {image_name} ({dataset_name})",
                       save_path=os.path.join(dataset_histograms, f"{image_name}_k{k}_hist.png"))


def main():
    """Главная функция: выбор набора и запуск анализа по заданию"""
    print("="*60)
    print("ИССЛЕДОВАТЕЛЬСКАЯ РАБОТА - LSB СТЕГАНОГРАФИЯ")
    print("="*60)
    
    # Доступные наборы
    available_datasets = {}
    
    if BOSSbase_PATH and os.path.exists(BOSSbase_PATH):
        available_datasets["1"] = ("BOSSbase (естественные)", BOSSbase_PATH)
    
    if MEDICAL_COMBINED_PATH and os.path.exists(MEDICAL_COMBINED_PATH):
        available_datasets["2"] = ("Медицинские (комбинированные)", MEDICAL_COMBINED_PATH)
    elif MEDICAL_GLIOMA_PATH and os.path.exists(MEDICAL_GLIOMA_PATH):
        available_datasets["2"] = ("Медицинские - глиома", MEDICAL_GLIOMA_PATH)
    
    if TEXTURE_PATH and os.path.exists(TEXTURE_PATH):
        available_datasets["3"] = ("Текстурные изображения", TEXTURE_PATH)
    
    if not available_datasets:
        print("\n❌ Не найдено ни одного набора изображений!")
        print("Проверьте пути в .env файле")
        return
    
    print("\n📁 Доступные наборы изображений:")
    for key, (name, path) in available_datasets.items():
        try:
            pgm_count = len([f for f in os.listdir(path) if f.lower().endswith('.pgm')])
            print(f"   {key}. {name} ({pgm_count} файлов)")
            print(f"      Путь: {path}")
        except:
            print(f"   {key}. {name}")
    
    # Выбор набора
    print("\n" + "="*60)
    choice = input("Выберите набор для анализа (1-3): ").strip()
    
    if choice not in available_datasets:
        print("Неверный выбор, беру первый доступный")
        choice = list(available_datasets.keys())[0]
    
    dataset_name, dataset_path = available_datasets[choice]
    
    # Запуск анализа
    analyze_dataset(dataset_path, dataset_name, num_images_for_planes=DEFAULT_NUM_VISUALIZE)
    
    print("\n" + "="*60)
    print("✅ АНАЛИЗ ЗАВЕРШЁН")
    print(f"📁 Результаты сохранены в папке: {RESULTS_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()