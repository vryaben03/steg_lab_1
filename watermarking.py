"""
Модуль для внедрения цифровых водяных знаков (ЦВЗ) в изображения
Реализует два метода:
1) LSB с секретным ключом (перемешивание порядка пикселей)
2) Адаптивный метод по локальной дисперсии
"""

import numpy as np
from PIL import Image
from dotenv import load_dotenv
import os
import random
from lsb_steganography import LSBSteganography
from stego_utils import compute_metrics

load_dotenv()

class DigitalWatermark:
    """
    Класс для внедрения и извлечения цифрового водяного знака
    """
    
    def __init__(self, image_path=None, logo_path=os.getenv("LOGO_PATH")):
        """
        Инициализация
        
        Args:
            image_path: путь к изображению-контейнеру
            logo_path: путь к изображению-логотипу (ЦВЗ)
        """
        self.image_path = image_path
        self.image = None
        self.height = 0
        self.width = 0
        
        self.logo_path = logo_path
        self.logo = None
        self.logo_height = 0
        self.logo_width = 0
        
        if image_path:
            self.load_image(image_path)
        if logo_path:
            self.load_logo(logo_path)
    
    def load_image(self, image_path):
        """Загружает изображение-контейнер (8-битное серое)"""
        img = Image.open(image_path).convert('L')
        self.image = np.array(img, dtype=np.uint8)
        self.height, self.width = self.image.shape
        print(f"Загружено изображение {self.width}x{self.height} из {image_path}")
    
    def load_logo(self, logo_path):
        """Загружает изображение-логотип (ЦВЗ) и преобразует в биты"""
        logo_img = Image.open(logo_path).convert('L')
        self.logo = np.array(logo_img, dtype=np.uint8)
        self.logo_height, self.logo_width = self.logo.shape
        print(f"Загружен логотип {self.logo_width}x{self.logo_height} из {logo_path}")
    
    def logo_to_bits(self):
        """Преобразует логотип в битовую строку (1 бит на пиксель)"""
        # Преобразуем в бинарное: >127 = 1, <=127 = 0
        binary_logo = (self.logo > 127).astype(np.uint8)
        bits = binary_logo.flatten().tolist()
        print(f"Логотип преобразован в {len(bits)} бит")
        return bits
    
    def bits_to_logo(self, bits, height, width):
        """Восстанавливает логотип из битов"""
        logo_array = np.array(bits[:height*width]).reshape(height, width) * 255
        return logo_array.astype(np.uint8)
    
    def get_pixel_order(self, key, total_pixels):
        """
        Генерирует перемешанный порядок пикселей на основе секретного ключа
        """
        random.seed(key)
        indices = list(range(total_pixels))
        random.shuffle(indices)
        return indices
    
    def embed_lsb(self, output_path, key=42):
        """
        Режим 1: Внедрение в LSB с секретным ключом
        
        Args:
            output_path: путь для сохранения стего-изображения
            key: секретный ключ (определяет порядок внедрения)
        
        Returns:
            dict: метрики качества
        """
        print("\n" + "="*50)
        print("РЕЖИМ 1: LSB с секретным ключом")
        print(f"Ключ: {key}")
        print("="*50)
        
        if self.image is None:
            raise ValueError("Изображение не загружено")
        if self.logo is None:
            raise ValueError("Логотип не загружен")
        
        # Преобразуем логотип в биты
        logo_bits = self.logo_to_bits()
        total_bits = self.height * self.width
        max_bits_to_embed = total_bits // 2  # не менее половины ёмкости
        
        # Если логотип меньше половины ёмкости, дублируем его
        required_bits = max_bits_to_embed
        if len(logo_bits) < required_bits:
            repeats = (required_bits // len(logo_bits)) + 1
            logo_bits = (logo_bits * repeats)[:required_bits]
            print(f"Логотип продублирован до {len(logo_bits)} бит")
        
        print(f"Внедряется {len(logo_bits)} бит (половина ёмкости: {max_bits_to_embed})")
        
        # Создаём перемешанный порядок пикселей
        pixel_order = self.get_pixel_order(key, total_bits)
        
        # Внедряем биты в LSB (используем int32 для избежания переполнения)
        stego_image = self.image.copy().astype(np.int32)
        bit_pos = 0  # LSB
        
        for idx, bit in enumerate(logo_bits):
            if idx >= max_bits_to_embed:
                break
            pixel_idx = pixel_order[idx]
            i = pixel_idx // self.width
            j = pixel_idx % self.width
            # Заменяем LSB (работаем с int32)
            stego_image[i, j] = (stego_image[i, j] & ~1) | bit
        
        # Приводим обратно к uint8
        stego_image = np.clip(stego_image, 0, 255).astype(np.uint8)
        
        # Сохраняем
        self.save_image(stego_image, output_path)
        
        # Считаем метрики
        metrics = compute_metrics(self.image_path, output_path)
        metrics['method'] = 'LSB'
        metrics['key'] = key
        print(f"PSNR: {metrics['PSNR']:.2f} дБ, MSE: {metrics['MSE']:.4f}, SSIM: {metrics['SSIM']:.4f}")
        
        return metrics
    
    def embed_adaptive(self, output_path, block_size=16):
        """
        Режим 2: Адаптивное внедрение по локальной дисперсии
        
        Args:
            output_path: путь для сохранения стего-изображения
            block_size: размер блока для вычисления дисперсии
        
        Returns:
            dict: метрики качества
        """
        print("\n" + "="*50)
        print("РЕЖИМ 2: Адаптивное внедрение по локальной дисперсии")
        print(f"Размер блока: {block_size}x{block_size}")
        print("="*50)
        
        if self.image is None:
            raise ValueError("Изображение не загружено")
        if self.logo is None:
            raise ValueError("Логотип не загружен")
        
        # Преобразуем логотип в биты
        logo_bits = self.logo_to_bits()
        total_bits = self.height * self.width
        max_bits_to_embed = total_bits // 2
        
        if len(logo_bits) < max_bits_to_embed:
            repeats = (max_bits_to_embed // len(logo_bits)) + 1
            logo_bits = (logo_bits * repeats)[:max_bits_to_embed]
            print(f"Логотип продублирован до {len(logo_bits)} бит")
        
        print(f"Внедряется {len(logo_bits)} бит (половина ёмкости: {max_bits_to_embed})")
        
        # Разбиваем изображение на блоки
        blocks_h = self.height // block_size
        blocks_w = self.width // block_size
        
        # Вычисляем дисперсию для каждого блока
        variances = np.zeros((blocks_h, blocks_w))
        for i in range(blocks_h):
            for j in range(blocks_w):
                block = self.image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                variances[i, j] = np.var(block)
        
        # Нормализуем дисперсии
        max_variance = np.max(variances) if np.max(variances) > 0 else 1
        normalized_vars = variances / max_variance
        
        # Распределяем биты по блокам
        total_capacity = max_bits_to_embed
        block_weights = normalized_vars.flatten()
        block_weights = block_weights / np.sum(block_weights)
        bits_per_block = (block_weights * total_capacity).astype(int)
        
        # Корректируем сумму
        diff = total_capacity - np.sum(bits_per_block)
        for i in range(diff):
            bits_per_block[i % len(bits_per_block)] += 1
        
        print(f"Распределено {np.sum(bits_per_block)} бит по {len(bits_per_block)} блокам")
        
        # Внедряем биты (используем int32)
        stego_image = self.image.copy().astype(np.int32)
        bit_idx = 0
        
        for i in range(blocks_h):
            for j in range(blocks_w):
                n_bits = bits_per_block[i * blocks_w + j]
                if n_bits == 0:
                    continue
                
                # Получаем пиксели блока
                block = stego_image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                block_flat = block.flatten()
                
                # Внедряем биты в LSB
                for idx in range(min(n_bits, len(block_flat))):
                    if bit_idx >= len(logo_bits):
                        break
                    block_flat[idx] = (block_flat[idx] & ~1) | logo_bits[bit_idx]
                    bit_idx += 1
                
                # Возвращаем блок обратно
                stego_image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = block_flat.reshape(block.shape)
        
        print(f"Внедрено {bit_idx} бит")
        
        # Приводим обратно к uint8
        stego_image = np.clip(stego_image, 0, 255).astype(np.uint8)
        
        # Сохраняем
        self.save_image(stego_image, output_path)
        
        # Считаем метрики
        metrics = compute_metrics(self.image_path, output_path)
        metrics['method'] = 'Adaptive'
        metrics['block_size'] = block_size
        print(f"PSNR: {metrics['PSNR']:.2f} дБ, MSE: {metrics['MSE']:.4f}, SSIM: {metrics['SSIM']:.4f}")
        
        return metrics
    
    def extract_lsb(self, stego_path, logo_size, key=42, total_bits_to_extract=None):
        """
        Извлечение ЦВЗ из LSB-стегоизображения с использованием ключа
        """
        # Загружаем стего-изображение
        stego_img = np.array(Image.open(stego_path).convert('L'), dtype=np.uint8)
        height, width = stego_img.shape
        total_pixels = height * width
        
        if total_bits_to_extract is None:
            total_bits_to_extract = total_pixels // 2
        
        # Восстанавливаем порядок пикселей
        pixel_order = self.get_pixel_order(key, total_pixels)
        
        # Извлекаем биты
        extracted_bits = []
        for idx in range(total_bits_to_extract):
            pixel_idx = pixel_order[idx]
            i = pixel_idx // width
            j = pixel_idx % width
            bit = stego_img[i, j] & 1
            extracted_bits.append(bit)
        
        # Восстанавливаем логотип
        logo_height, logo_width = logo_size
        extracted_logo = self.bits_to_logo(extracted_bits, logo_height, logo_width)
        
        return extracted_logo
    
    def extract_adaptive(self, stego_path, logo_size, block_size=16, total_bits_to_extract=None):
        """
        Извлечение ЦВЗ из адаптивного стего-изображения
        С учётом дисперсии блоков (повторяем логику внедрения)
        """
        stego_img = np.array(Image.open(stego_path).convert('L'), dtype=np.uint8)
        height, width = stego_img.shape
        
        if total_bits_to_extract is None:
            total_bits_to_extract = (height * width) // 2
        
        # Разбиваем на блоки
        blocks_h = height // block_size
        blocks_w = width // block_size
        
        # ВЫЧИСЛЯЕМ ДИСПЕРСИЮ (как при внедрении)
        # Для этого нужно исходное изображение или его приближение.
        # Проще всего: используем само стего-изображение для расчёта дисперсии
        variances = np.zeros((blocks_h, blocks_w))
        for i in range(blocks_h):
            for j in range(blocks_w):
                block = stego_img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                variances[i, j] = np.var(block)
        
        # Нормализуем
        max_variance = np.max(variances) if np.max(variances) > 0 else 1
        normalized_vars = variances / max_variance
        
        # Распределяем биты (как при внедрении)
        block_weights = normalized_vars.flatten()
        block_weights = block_weights / np.sum(block_weights)
        bits_per_block = (block_weights * total_bits_to_extract).astype(int)
        
        # Корректируем сумму
        diff = total_bits_to_extract - np.sum(bits_per_block)
        for i in range(diff):
            bits_per_block[i % len(bits_per_block)] += 1
        
        # Извлекаем биты, читая из каждого блока ровно столько, сколько в него внедрено
        extracted_bits = []
        bit_idx = 0
        for i in range(blocks_h):
            for j in range(blocks_w):
                n_bits = bits_per_block[i * blocks_w + j]
                if n_bits == 0:
                    continue
                
                block = stego_img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                block_flat = block.flatten()
                
                for idx in range(min(n_bits, len(block_flat))):
                    if len(extracted_bits) >= total_bits_to_extract:
                        break
                    extracted_bits.append(block_flat[idx] & 1)
                
                if len(extracted_bits) >= total_bits_to_extract:
                    break
            if len(extracted_bits) >= total_bits_to_extract:
                break
        
        # Восстанавливаем логотип
        logo_height, logo_width = logo_size
        extracted_logo = self.bits_to_logo(extracted_bits, logo_height, logo_width)
        
        return extracted_logo
    
    def save_image(self, image_array, output_path):
        """Сохраняет изображение"""
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        Image.fromarray(image_array.astype(np.uint8)).save(output_path)
        print(f"Сохранено изображение: {output_path}")
    
    def save_logo(self, logo_array, output_path):
        """Сохраняет логотип"""
        Image.fromarray(logo_array.astype(np.uint8)).save(output_path)
        print(f"Сохранён логотип: {output_path}")


def compare_methods(image_path, logo_path, output_dir, key=42, block_size=16):
    """
    Сравнивает два метода внедрения ЦВЗ на одном изображении
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("СРАВНЕНИЕ МЕТОДОВ ВНЕДРЕНИЯ ЦВЗ")
    print("="*60)
    
    # Создаём объект водяного знака
    watermark = DigitalWatermark(image_path, logo_path)
    
    # Режим 1: LSB с ключом
    output_lsb = os.path.join(output_dir, "watermark_lsb.png")
    metrics_lsb = watermark.embed_lsb(output_lsb, key=key)
    
    # Режим 2: Адаптивный
    output_adaptive = os.path.join(output_dir, "watermark_adaptive.png")
    metrics_adaptive = watermark.embed_adaptive(output_adaptive, block_size=block_size)
    
    # Извлечение и проверка
    print("\n" + "="*40)
    print("ИЗВЛЕЧЕНИЕ И ПРОВЕРКА ЦВЗ")
    print("="*40)
    
    # Извлекаем из LSB
    extracted_lsb = watermark.extract_lsb(output_lsb, (watermark.logo_height, watermark.logo_width), key=key)
    watermark.save_logo(extracted_lsb, os.path.join(output_dir, "extracted_lsb_logo.png"))
    
    # Извлекаем из адаптивного
    extracted_adaptive = watermark.extract_adaptive(output_adaptive, (watermark.logo_height, watermark.logo_width), block_size=block_size)
    watermark.save_logo(extracted_adaptive, os.path.join(output_dir, "extracted_adaptive_logo.png"))
    
    # Вывод результатов сравнения
    print("\n" + "="*40)
    print("СРАВНЕНИЕ МЕТРИК")
    print("="*40)
    print(f"{'Метод':<15} {'PSNR (дБ)':<12} {'MSE':<12} {'SSIM':<8}")
    print("-" * 50)
    print(f"{'LSB с ключом':<15} {metrics_lsb['PSNR']:<12.2f} {metrics_lsb['MSE']:<12.4f} {metrics_lsb['SSIM']:<8.4f}")
    print(f"{'Адаптивный':<15} {metrics_adaptive['PSNR']:<12.2f} {metrics_adaptive['MSE']:<12.4f} {metrics_adaptive['SSIM']:<8.4f}")
    
    return metrics_lsb, metrics_adaptive


def process_dataset_watermarking(dataset_path, logo_path, dataset_name, output_dir, max_images=30):
    """
    Обрабатывает набор изображений для исследовательской части
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    pgm_files = [f for f in os.listdir(dataset_path) if f.lower().endswith('.pgm')]
    try:
        pgm_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        pgm_files.sort()
    pgm_files = pgm_files[:max_images]
    
    print(f"\nОбработка набора {dataset_name}, {len(pgm_files)} изображений")
    
    for idx, img_file in enumerate(pgm_files):
        img_path = os.path.join(dataset_path, img_file)
        print(f"  {idx+1}/{len(pgm_files)}: {img_file}")
        
        watermark = DigitalWatermark(img_path, logo_path)
        
        # Адаптивный метод
        output_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_watermark.png")
        metrics = watermark.embed_adaptive(output_path, block_size=16)
        metrics['image'] = img_file
        metrics['dataset'] = dataset_name
        results.append(metrics)
    
    return results


if __name__ == "__main__":
    print("="*60)
    print("ЦИФРОВЫЕ ВОДЯНЫЕ ЗНАКИ")
    print("="*60)
    
    # Пути
    image_path = r"C:\...\...\...\BOSSbase_1.01\1.pgm"
    logo_path = os.getenv("LOGO_PATH")  # подготовить логотип
    
    output_dir = r"C:\...\...\...\...\...\results\watermarking"
    
    if not os.path.exists(logo_path):
        print("\nЛоготип не найден! Создаю тестовый логотип...")
        # тестовый логотип
        test_logo = np.zeros((64, 64), dtype=np.uint8)
        test_logo[20:44, 20:44] = 255
        Image.fromarray(test_logo).save(logo_path)
        print(f"Создан тестовый логотип: {logo_path}")
    
    if os.path.exists(image_path) and os.path.exists(logo_path):
        compare_methods(image_path, logo_path, output_dir)
    else:
        print(f"\nПроверьте пути:")
        print(f"   Изображение: {image_path} - {os.path.exists(image_path)}")
        print(f"   Логотип: {logo_path} - {os.path.exists(logo_path)}")