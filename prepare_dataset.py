"""
Скрипт для подготовки медицинских изображений к использованию:
- Конвертация в grayscale (8-битные серые)
- Изменение размера до 512x512
- Сохранение в PGM формате (как BOSSbase)
"""
import os
from PIL import Image
from tqdm import tqdm  # для красивого прогресс-бара

from config import TARGET_SIZE, DEFAULT_MAX_IMAGES
from dotenv import load_dotenv
load_dotenv()

# путь к скачанному датасету (укажите ваш путь)
DATASET_PATH = os.getenv("DATASET_PATH_MEDICAL")

# папка для сохранения подготовленных изображений
OUTPUT_PATH = os.getenv("OUTPUT_PATH_MEDICAL")

# какие папки обрабатывать
FOLDERS = ["brain_glioma", "brain_menin", "brain_tumor"]

def prepare_images():
    """Конвертирует и подготавливает изображения"""
    
    # создаем выходную папку
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    total_processed = 0
    
    for folder in FOLDERS:
        folder_path = os.path.join(DATASET_PATH, folder)
        
        if not os.path.exists(folder_path):
            print(f"Папка не найдена: {folder_path}")
            continue
        
        # находим все JPG файлы
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\nПапка: {folder}")
        print(f"   Найдено изображений: {len(image_files)}")
        
        # берем первые IMAGES_PER_FOLDER изображений
        selected = image_files[:DEFAULT_MAX_IMAGES]
        
        # создаем подпапку в выходной директории
        output_subfolder = os.path.join(OUTPUT_PATH, folder)
        os.makedirs(output_subfolder, exist_ok=True)
        
        # обрабатываем каждое изображение
        for idx, img_file in enumerate(tqdm(selected, desc=f"   Обработка {folder}")):
            try:
                # Загружаем изображение
                img_path = os.path.join(folder_path, img_file)
                img = Image.open(img_path)
                
                # Конвертируем в grayscale
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Изменяем размер до 512x512
                img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                
                # Сохраняем в PGM (как в BOSSbase)
                output_name = f"{folder}_{idx+1:04d}.pgm"
                output_path = os.path.join(output_subfolder, output_name)
                img.save(output_path)
                
                total_processed += 1
                
            except Exception as e:
                print(f"   Ошибка при обработке {img_file}: {e}")
    
    print(f"\n✅ Готово! Обработано {total_processed} изображений")
    print(f"📁 Сохранено в: {OUTPUT_PATH}")
    
    # Показываем структуру выходной папки
    print("\nСтруктура выходной папки:")
    for folder in FOLDERS:
        folder_path = os.path.join(OUTPUT_PATH, folder)
        if os.path.exists(folder_path):
            count = len([f for f in os.listdir(folder_path) if f.endswith('.pgm')])
            print(f"   {folder}/: {count} файлов")


def check_image_info():
    """Проверяет информацию о подготовленных изображениях"""
    print("\n" + "="*50)
    print("ПРОВЕРКА ИЗОБРАЖЕНИЙ")
    print("="*50)
    
    for folder in FOLDERS:
        folder_path = os.path.join(OUTPUT_PATH, folder)
        if not os.path.exists(folder_path):
            continue
        
        pgm_files = [f for f in os.listdir(folder_path) if f.endswith('.pgm')]
        
        if not pgm_files:
            continue
        
        # Проверяем первое изображение
        sample = os.path.join(folder_path, pgm_files[0])
        img = Image.open(sample)
        
        print(f"\n📁 {folder}/")
        print(f"   Количество: {len(pgm_files)}")
        print(f"   Размер: {img.size}")
        print(f"   Режим: {img.mode}")
        print(f"   Пример: {pgm_files[0]}")


def create_combined_set():
    """
    Создает единую папку со 100 изображениями для удобства
    (смешивает все типы)
    """
    combined_path = os.path.join(OUTPUT_PATH, "combined_medical")
    os.makedirs(combined_path, exist_ok=True)
    
    idx = 1
    for folder in FOLDERS:
        folder_path = os.path.join(OUTPUT_PATH, folder)
        if not os.path.exists(folder_path):
            continue
        
        pgm_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pgm')])
        
        for pgm_file in pgm_files:
            src = os.path.join(folder_path, pgm_file)
            dst = os.path.join(combined_path, f"med_{idx:04d}.pgm")
            img = Image.open(src)
            img.save(dst)
            idx += 1
    
    print(f"\n✅ Создан объединенный набор в: {combined_path}")
    print(f"   Всего файлов: {idx-1}")


if __name__ == "__main__":
    print("="*50)
    print("ПОДГОТОВКА МЕДИЦИНСКИХ ИЗОБРАЖЕНИЙ")
    print("="*50)
    
    # Проверяем, существует ли папка с датасетом
    if not os.path.exists(DATASET_PATH):
        print(f"\n❌ ОШИБКА: Папка с датасетом не найдена!")
        print(f"   Укажите правильный путь в переменной DATASET_PATH")
        print(f"   Текущий путь: {DATASET_PATH}")
        
        # Показываем возможные варианты
        desktop = r"C:\folder"
        print(f"\n📁 Проверьте папки на рабочем столе:")
        for item in os.listdir(desktop):
            item_path = os.path.join(desktop, item)
            if os.path.isdir(item_path) and ('brain' in item.lower() or 'mri' in item.lower()):
                print(f"   - {item}")
        exit()
    
    # Запускаем подготовку
    prepare_images()
    check_image_info()
    
    # Спрашиваем, создать ли объединенный набор
    print("\n" + "="*50)
    response = input("Создать объединенный набор (все типы в одной папке)? (y/n): ")
    if response.lower() == 'y':
        create_combined_set()
    
    print("\n✨ Готово! Теперь можно использовать эти изображения для исследования.")
    print("   Укажите путь к папке в переменной MEDICAL_PATH в основном скрипте.")