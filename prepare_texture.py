"""
Подготовка текстурных изображений для стеганографии
"""

import os
from PIL import Image
from tqdm import tqdm
from config import *
from dotenv import load_dotenv
load_dotenv()

# ========== КОНФИГУРАЦИЯ ==========
# Путь к скачанному текстурному датасету (УКАЖИТЕ ВАШ ПУТЬ!)
TEXTURE_PATH = os.getenv("DATASET_PATH_TEXTURE")

# Папка для сохранения подготовленных изображений
OUTPUT_PATH = os.getenv("TEXTURE_PATH")

def prepare_textures():
    """Конвертирует текстурные изображения в PGM 512x512"""
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Ищем все изображения в папке и подпапках
    image_files = []
    for root, dirs, files in os.walk(TEXTURE_PATH):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, f))
    
    print(f"Найдено изображений: {len(image_files)}")
    
    # Берем первые IMAGES_COUNT
    selected = image_files[:DEFAULT_MAX_IMAGES]
    
    for idx, img_path in enumerate(tqdm(selected, desc="Обработка текстур")):
        try:
            img = Image.open(img_path)
            
            # Grayscale
            if img.mode != 'L':
                img = img.convert('L')
            
            # Изменяем размер
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            
            # Сохраняем в PGM
            output_name = f"texture_{idx+1:04d}.pgm"
            output_path = os.path.join(OUTPUT_PATH, output_name)
            img.save(output_path)
            
        except Exception as e:
            print(f"Ошибка: {img_path} - {e}")
    
    print(f"\n✅ Готово! Сохранено {DEFAULT_MAX_IMAGES} изображений в {OUTPUT_PATH}")


if __name__ == "__main__":
    prepare_textures()