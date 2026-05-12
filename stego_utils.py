"""
Вспомогательные функции для стеганографического анализа:
- метрики качества (PSNR, MSE, SSIM)
- гистограммы
- доверительные интервалы
- массовая обработка наборов изображений
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy import stats
import os
import pandas as pd
from config import RESULTS_PATH

from lsb_steganography import LSBSteganography


def compute_metrics(original_path, stego_path):
    """
    Вычисляет PSNR, MSE, SSIM между двумя изображениями
    
    Returns:
        dict: {'MSE': float, 'PSNR': float, 'SSIM': float}
    """
    orig = np.array(Image.open(original_path).convert('L'), dtype=np.float64)
    stego = np.array(Image.open(stego_path).convert('L'), dtype=np.float64)
    
    # MSE
    mse = np.mean((orig - stego) ** 2)
    
    # PSNR
    if mse == 0:
        psnr_val = float('inf')
    else:
        psnr_val = 10 * np.log10(255.0 ** 2 / mse)
    
    # SSIM
    orig_uint8 = orig.astype(np.uint8)
    stego_uint8 = stego.astype(np.uint8)
    ssim_val = ssim(orig_uint8, stego_uint8, data_range=255)
    
    return {
        'MSE': mse,
        'PSNR': psnr_val,
        'SSIM': ssim_val
    }


def plot_histogram(image_path, title="Гистограмма яркости", save_path=None):
    """
    Строит гистограмму распределения яркости изображения
    
    Args:
        image_path: путь к изображению
        title: заголовок графика
        save_path: если указан, сохраняет график в файл
    """
    img = np.array(Image.open(image_path).convert('L'))
    plt.figure(figsize=(10, 5))
    plt.hist(img.flatten(), bins=256, range=(0, 255), alpha=0.7, color='gray', edgecolor='black')
    plt.title(title)
    plt.xlabel("Яркость")
    plt.ylabel("Количество пикселей")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Сохранена гистограмма: {save_path}")
    plt.show()


def confidence_interval(data, confidence=0.95):
    """
    Вычисляет доверительный интервал для выборки
    
    Args:
        data: список или массив значений
        confidence: уровень доверия (по умолчанию 0.95)
    
    Returns:
        tuple: (mean, lower_bound, upper_bound)
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)  # стандартная ошибка среднего
    margin = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - margin, mean + margin


def process_dataset(dataset_path, message_path, k_values=[1, 2, 3], 
                    output_dir="stego_results", max_images=100, verbose=True):
    """
    Обрабатывает все изображения в папке dataset_path
    Внедряет сообщение в каждое изображение для каждого k
    
    Args:
        dataset_path: путь к папке с исходными изображениями (только чтение)
        message_path: путь к файлу с сообщением
        k_values: список битовых плоскостей для внедрения
        output_dir: папка для сохранения стего-изображений
        max_images: максимальное количество изображений для обработки
        verbose: выводить ли прогресс
    
    Returns:
        pandas.DataFrame: таблица с метриками
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    # Ищем файлы .pgm (только в исходной папке)
    image_files = []
    for ext in ['.pgm', '.bmp', '.png']:
        image_files.extend([f for f in os.listdir(dataset_path) if f.lower().endswith(ext)])
    image_files = image_files[:max_images]
    
    if not image_files:
        print(f"В папке {dataset_path} не найдено файлов изображений")
        return pd.DataFrame()
    
    print(f"Найдено {len(image_files)} изображений в {dataset_path}")
    
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(dataset_path, img_file)
        base_name = os.path.splitext(img_file)[0]
        
        if verbose:
            print(f"\nОбработка {idx+1}/{len(image_files)}: {img_file}")
        
        for k in k_values:
            output_filename = f"{base_name}_k{k}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                stego = LSBSteganography(img_path)
                stego.embed_message(message_path, k, output_path, verbose=False)
                
                metrics = compute_metrics(img_path, output_path)
                metrics['image'] = img_file
                metrics['k'] = k
                results.append(metrics)
                
                if verbose:
                    print(f"  k={k}: PSNR={metrics['PSNR']:.2f} dB, SSIM={metrics['SSIM']:.4f}")
                    
            except Exception as e:
                if verbose:
                    print(f"  Ошибка при обработке k={k}: {e}")
    
    return pd.DataFrame(results)


def analyze_bit_planes(dataset_path, num_images=5, output_dir="bit_planes_analysis"):
    """
    Для num_images изображений из набора визуализирует все 8 битовых плоскостей
    
    Args:
        dataset_path: путь к папке с исходными изображениями (только чтение)
        num_images: количество изображений для анализа
        output_dir: папка для сохранения результатов
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = []
    for ext in ['.pgm', '.bmp', '.png']:
        image_files.extend([f for f in os.listdir(dataset_path) if f.lower().endswith(ext)])
    image_files = image_files[:num_images]
    
    for img_file in image_files:
        print(f"Визуализация битовых плоскостей для {img_file}")
        img_path = os.path.join(dataset_path, img_file)
        stego = LSBSteganography(img_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_planes.png")
        stego.visualize_bit_planes(output_path=output_path)


def compare_containers(results_dict, confidence=0.95):
    """
    Сравнивает три типа контейнеров по метрикам PSNR
    
    Args:
        results_dict: словарь {container_name: DataFrame_with_metrics}
        confidence: уровень доверия для интервалов
    
    Returns:
        pandas.DataFrame: таблица с доверительными интервалами
    """
    comparison = []
    
    for container_name, df in results_dict.items():
        for k in [1, 2, 3]:
            psnr_values = df[df['k'] == k]['PSNR'].values
            mean, lower, upper = confidence_interval(psnr_values, confidence)
            
            comparison.append({
                'Container': container_name,
                'k': k,
                'Mean_PSNR': mean,
                'CI_lower': lower,
                'CI_upper': upper,
                'Std': np.std(psnr_values)
            })
    
    return pd.DataFrame(comparison)


def plot_confidence_intervals(comparison_df, save_path=None):
    """
    Визуализирует доверительные интервалы для PSNR
    """
    import matplotlib.pyplot as plt
    
    containers = comparison_df['Container'].unique()
    k_values = [1, 2, 3]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    width = 0.25
    x = np.arange(len(k_values))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for i, container in enumerate(containers):
        container_data = comparison_df[comparison_df['Container'] == container]
        means = [container_data[container_data['k'] == k]['Mean_PSNR'].values[0] for k in k_values]
        errors_lower = [container_data[container_data['k'] == k]['Mean_PSNR'].values[0] - 
                        container_data[container_data['k'] == k]['CI_lower'].values[0] for k in k_values]
        errors_upper = [container_data[container_data['k'] == k]['CI_upper'].values[0] - 
                        container_data[container_data['k'] == k]['Mean_PSNR'].values[0] for k in k_values]
        
        ax.bar(x + i * width, means, width, label=container, color=colors[i], alpha=0.7)
        ax.errorbar(x + i * width, means, yerr=[errors_lower, errors_upper], 
                   fmt='none', color='black', capsize=5)
    
    ax.set_xlabel('Bit Plane (k)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Сравнение PSNR с доверительными интервалами (95%)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def visualize_all_bit_planes_for_image(image_path, output_path=None):
    """
    Визуализирует все 8 битовых плоскостей для одного изображения
    """
    from lsb_steganography import LSBSteganography
    stego = LSBSteganography(image_path)
    stego.visualize_bit_planes(output_path=output_path)

def analyze_bit_plane_variance(dataset_path, k_values=[1, 2, 3], max_images=100):
    """
    Анализирует дисперсию значений в битовых плоскостях для набора изображений
    
    Args:
        dataset_path: путь к папке с PGM изображениями
        k_values: список битовых плоскостей для анализа (по умолчанию 1,2,3)
        max_images: максимальное количество изображений для анализа
    
    Returns:
        pandas.DataFrame: таблица с дисперсиями для каждого изображения и k
    """
    from lsb_steganography import LSBSteganography
    
    # Находим PGM файлы
    pgm_files = [f for f in os.listdir(dataset_path) if f.lower().endswith('.pgm')]
    
    # Сортируем и ограничиваем количество
    try:
        pgm_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        pgm_files.sort()
    pgm_files = pgm_files[:max_images]
    
    results = []
    
    print(f"\nАнализ дисперсии битовых плоскостей для {len(pgm_files)} изображений")
    
    for img_file in pgm_files:
        img_path = os.path.join(dataset_path, img_file)
        stego = LSBSteganography(img_path)
        
        for k in k_values:
            # Извлекаем битовую плоскость
            bit_plane = stego.extract_bit_plane(k)
            # Приводим к 0/1 (сейчас 0/255)
            bit_values = (bit_plane / 255).astype(np.uint8)
            # Считаем дисперсию
            variance = np.var(bit_values)
            
            results.append({
                'image': img_file,
                'k': k,
                'variance': variance
            })
    
    return pd.DataFrame(results)


def confidence_interval_variance(data, confidence=0.95):
    """
    Вычисляет доверительный интервал для СРЕДНЕГО ЗНАЧЕНИЯ
    (то есть: где находится истинное среднее дисперсии битовой плоскости)
    """
    from scipy import stats
    import numpy as np
    
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # стандартная ошибка среднего
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)  # t-критерий
    margin = sem * t_crit
    
    return {
        'mean': mean,
        'ci_lower': mean - margin,
        'ci_upper': mean + margin
    }

def calculate_all_confidence_intervals(datasets_dict, k_values=[1, 2, 3], max_images=100):
    """
    Рассчитывает доверительные интервалы для PSNR и дисперсии битовых плоскостей
    для всех наборов изображений
    
    Args:
        datasets_dict: словарь {название_набора: путь_к_папке}
        k_values: список битовых плоскостей
        max_images: максимальное количество изображений для анализа
    
    Returns:
        dict: результаты для PSNR и дисперсии
    """
    from stego_utils import process_dataset, compute_metrics, analyze_bit_plane_variance, confidence_interval
    
    results = {
        'psnr': {},
        'variance': {}
    }
    
    for dataset_name, dataset_path in datasets_dict.items():
        if not os.path.exists(dataset_path):
            print(f"❌ Папка не найдена: {dataset_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Обработка набора: {dataset_name}")
        print(f"{'='*60}")
        
        # 1. PSNR доверительные интервалы
        print("\n--- Расчет PSNR ---")
        
        # Создаём тестовое сообщение
        message_path = os.path.join(os.path.dirname(dataset_path), "temp_message.txt")
        if not os.path.exists(message_path):
            create_test_message(message_path, target_kb=20)
        
        # Обрабатываем набор
        psnr_data = {k: [] for k in k_values}
        
        pgm_files = [f for f in os.listdir(dataset_path) if f.lower().endswith('.pgm')]
        try:
            pgm_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        except ValueError:
            pgm_files.sort()
        pgm_files = pgm_files[:max_images]
        
        for idx, img_file in enumerate(pgm_files):
            img_path = os.path.join(dataset_path, img_file)
            
            for k in k_values:
                try:
                    stego = LSBSteganography(img_path)
                    temp_output = os.path.join(RESULTS_PATH, f"temp_k{k}.png")
                    stego.embed_message(message_path, k=k, output_path=temp_output, verbose=False)
                    
                    metrics = compute_metrics(img_path, temp_output)
                    psnr_data[k].append(metrics['PSNR'])
                    
                except Exception as e:
                    print(f"Ошибка {img_file}, k={k}: {e}")
            
            if (idx + 1) % 20 == 0:
                print(f"  Обработано {idx+1}/{len(pgm_files)} изображений")
        
        # Считаем доверительные интервалы для PSNR
        results['psnr'][dataset_name] = {}
        for k in k_values:
            ci = confidence_interval(psnr_data[k], confidence=0.95)
            results['psnr'][dataset_name][k] = {
                'mean': ci[0],
                'ci_lower': ci[1],
                'ci_upper': ci[2],
                'data': psnr_data[k]
            }
            print(f"  k={k}: PSNR = {ci[0]:.2f} дБ [{ci[1]:.2f}, {ci[2]:.2f}]")
        
        # 2. Дисперсия битовых плоскостей
        print("\n--- Расчет дисперсии битовых плоскостей ---")
        
        variance_df = analyze_bit_plane_variance(dataset_path, k_values=k_values, max_images=max_images)
        
        results['variance'][dataset_name] = {}
        for k in k_values:
            var_data = variance_df[variance_df['k'] == k]['variance'].values
            ci = confidence_interval_variance(var_data, confidence=0.95)
            results['variance'][dataset_name][k] = {
                'mean': ci['mean'],
                'ci_lower': ci['ci_lower'],
                'ci_upper': ci['ci_upper'],
                'data': var_data
            }
            print(f"  k={k}: Дисперсия = {ci['mean']:.6f} [{ci['ci_lower']:.6f}, {ci['ci_upper']:.6f}]")
    
    return results


def plot_confidence_intervals_results(results, save_path=None):
    """
    Визуализирует доверительные интервалы для PSNR и дисперсии
    """
    import matplotlib.pyplot as plt
    
    # График для PSNR
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PSNR
    ax1 = axes[0]
    k_values = [1, 2, 3]
    x = np.arange(len(k_values))
    width = 0.25
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for i, (dataset_name, psnr_data) in enumerate(results['psnr'].items()):
        means = [psnr_data[k]['mean'] for k in k_values]
        # Берём абсолютные значения ошибок (чтобы избежать отрицательных)
        errors_lower = [max(0, psnr_data[k]['mean'] - psnr_data[k]['ci_lower']) for k in k_values]
        errors_upper = [max(0, psnr_data[k]['ci_upper'] - psnr_data[k]['mean']) for k in k_values]
        
        ax1.bar(x + i * width, means, width, label=dataset_name, color=colors[i % len(colors)], alpha=0.7)
        ax1.errorbar(x + i * width, means, yerr=[errors_lower, errors_upper], 
                    fmt='none', color='black', capsize=5)
    
    ax1.set_xlabel('Битовая плоскость (k)')
    ax1.set_ylabel('PSNR (дБ)')
    ax1.set_title('Доверительные интервалы для PSNR (95%)')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([f'k={k}' for k in k_values])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Дисперсия
    ax2 = axes[1]
    
    for i, (dataset_name, variance_data) in enumerate(results['variance'].items()):
        means = [variance_data[k]['mean'] for k in k_values]
        # Берём абсолютные значения ошибок
        errors_lower = [max(0, variance_data[k]['mean'] - variance_data[k]['ci_lower']) for k in k_values]
        errors_upper = [max(0, variance_data[k]['ci_upper'] - variance_data[k]['mean']) for k in k_values]
        
        ax2.bar(x + i * width, means, width, label=dataset_name, color=colors[i % len(colors)], alpha=0.7)
        ax2.errorbar(x + i * width, means, yerr=[errors_lower, errors_upper], 
                    fmt='none', color='black', capsize=5)
    
    ax2.set_xlabel('Битовая плоскость (k)')
    ax2.set_ylabel('Дисперсия')
    ax2.set_title('Доверительные интервалы для дисперсии битовых плоскостей (95%)')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f'k={k}' for k in k_values])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранён: {save_path}")
    plt.show()


def create_test_message(filepath, target_kb=20):
    """Вспомогательная функция для создания тестового сообщения"""
    if os.path.exists(filepath):
        return
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for i in range(target_kb * 14):
            f.write(f"Строка {i:04d}: Тестовое сообщение. ")