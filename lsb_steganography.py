"""
Модуль для LSB стеганографии в 8-битных серых изображениях
Содержит класс LSBSteganography для внедрения и извлечения сообщений
"""

import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from config import LENGTH_BITS, BITS_PER_BYTE

class LSBSteganography:
    """Класс для работы с LSB стеганографией в 8-битных серых изображениях"""
    
    def __init__(self, image_path=None):
        """
        Инициализация с загрузкой изображения
        
        Args:
            image_path: путь к 8-битному серому изображению (PGM, BMP, PNG)
        """
        self.image = None      # массив пикселей (H, W)
        self.height = 0 # H
        self.width = 0 # W
        self.image_path = None # путь к изображению
        
        # есть путь - загружаем изображение
        if image_path:
            self.load_image(image_path)
    
    def load_image(self, image_path):
        """Загружает 8-битное серое изображение"""
        img = Image.open(image_path).convert('L')  # 'L' = grayscale
        self.image = np.array(img, dtype=np.uint8)
        self.height, self.width = self.image.shape
        self.image_path = image_path
        print(f"Загружено изображение {self.width}x{self.height} из {image_path}")
    
    def save_image(self, image_array, output_path):
        """Сохраняет изображение"""
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        Image.fromarray(image_array.astype(np.uint8)).save(output_path)
        print(f"Сохранено изображение: {output_path}")
    
    # ========== извлечение битовой плоскости ==========
    
    def extract_bit_plane(self, k):
        """
        Извлекает k-ю битовую плоскость
        
        Args:
            k: номер бита (1..8), где 1 — LSB, 8 — MSB
        
        Returns:
            np.ndarray: бинарное изображение (0 или 255)
        """
        if self.image is None:
            raise ValueError("Изображение не загружено")
        
        bit_pos = k - 1  # 0 для LSB, 7 для MSB
        bit_plane = (self.image >> bit_pos) & 1 # сдвиг битов; обнуление всех битов, кроме самого младшего
        return (bit_plane * 255).astype(np.uint8) # преобразует биты в видимые пиксели(черный или белый)
    
    def visualize_bit_planes(self, output_path=None):
        """
        Визуализирует все 8 битовые плоскости в сетке 2x4
        """
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        
        for k in range(1, 9):
            plane = self.extract_bit_plane(k)
            axes[k-1].imshow(plane, cmap='gray', vmin=0, vmax=255)
            axes[k-1].set_title(f'Bit Plane {k} ({"LSB" if k==1 else "MSB" if k==8 else ""})')
            axes[k-1].axis('off')
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Сохранена визуализация: {output_path}")
        plt.show()
    
    # ========== внедрение сообщения ==========
    
    def embed_message(self, message_path, k, output_path, verbose=True):
        """
        Внедряет сообщение из текстового файла в k-ю битовую плоскость
        
        Args:
            message_path: путь к текстовому файлу
            k: номер бита (1..8)
            output_path: путь для сохранения стего-изображения
            verbose: выводить ли отладочную информацию
        
        Returns:
            int: количество внедренных бит
        """
        # читаем сообщение
        with open(message_path, 'r', encoding='utf-8') as f:
            message = f.read()
        
        if verbose:
            print(f"\n=== ВНЕДРЕНИЕ ===")
        
        # преобразуем текст в байты UTF-8
        message_bytes = message.encode('utf-8')
        msg_byte_count = len(message_bytes)
        
        if verbose:
            print(f"Размер сообщения: {msg_byte_count} байт")
        
        # преобразуем байты в биты (старший бит первым)
        message_bits = []
        for byte in message_bytes:
            for i in range(7, -1, -1):  # от 7 до 0
                message_bits.append((byte >> i) & 1)
        
        max_bits = self.height * self.width
        
        # проверяем, помещается ли сообщение
        total_bits_needed = LENGTH_BITS + len(message_bits)
        if total_bits_needed > max_bits:
            if verbose:
                print(f"Предупреждение: сообщение слишком большое, обрезаем...")
            max_message_bits = max_bits - LENGTH_BITS
            message_bits = message_bits[:max_message_bits]
            msg_byte_count = len(message_bits) // BITS_PER_BYTE
            if verbose:
                print(f"Обрезано до {msg_byte_count} байт")
        
        # кодируем длину как 32-битное число (старший бит первый)
        length_bits = []
        for i in range(LENGTH_BITS-1, -1, -1):  # от 31 до 0
            length_bits.append((msg_byte_count >> i) & 1)
        
        if verbose:
            print(f"Длина сообщения: {msg_byte_count} байт = {msg_byte_count*BITS_PER_BYTE} бит")
            print(f"Всего бит для внедрения: {len(length_bits) + len(message_bits)} из {max_bits} максимальных")
        
        # объединяем
        full_bits = length_bits + message_bits
        
        # внедряем
        # создаём копию изображения и преобразуем в int32
        stego_image = self.image.copy().astype(np.int32)
        # определяем позицию бита, который будем заменять
        bit_pos = k - 1

        # проходим по всем битам сообщения (full_bits = [биты длины] + [биты сообщения])
        for idx, bit in enumerate(full_bits):
            # конец картинки
            if idx >= max_bits:
                break
            # преобразование в координаты (i, j) пикселя
            i = idx // self.width
            j = idx % self.width

            # внедряем один бит в пиксель:
            # 1. (stego_image[i, j] & ~(1 << bit_pos)) — обнуляем целевой бит
            # 2. (bit << bit_pos) — сдвигаем бит сообщения на нужную позицию
            # 3. | — объединяем: старые биты + новый бит
            stego_image[i, j] = (stego_image[i, j] & ~(1 << bit_pos)) | (bit << bit_pos)
        
        # приводим значения к диапазону [0, 255] и типу uint8 (стандарт для изображений)
        stego_image = np.clip(stego_image, 0, 255).astype(np.uint8)
        # сохраняем результат
        self.save_image(stego_image, output_path)
        
        # проверка корректности записи
        if verbose:
            self._verify_embedding(output_path, k, length_bits)
        
        return len(full_bits)
    
    def _verify_embedding(self, output_path, k, expected_length_bits):
        """Проверяет, что биты длины записаны корректно"""
        bit_pos = k - 1
        check_img = np.array(Image.open(output_path).convert('L'), dtype=np.uint8)
        check_bits = []
        # читаем первые 32 пикселя (где хранится длина сообщения)
        for idx in range(LENGTH_BITS):
            i = idx // self.width
            j = idx % self.width
            check_bits.append(int((check_img[i, j] >> bit_pos) & 1))
        
        # сравниваем то, что прочитали, с тем, что должны были записать
        if check_bits == expected_length_bits:
            print("✓ Биты длины записаны корректно")
        else:
            print("✗ Ошибка: биты длины не совпадают!")
    
    # ========== извлечение сообщения ==========
    
    def extract_message(self, k, verbose=True):
        """
        Извлекает сообщение из k-й битовой плоскости
        
        Args:
            k: номер бита (1..8)
            verbose: выводить ли отладочную информацию
        
        Returns:
            str: извлеченное сообщение
        """
        if self.image is None:
            raise ValueError("Изображение не загружено")
        
        bit_pos = k - 1
        max_bits = self.height * self.width
        
        if verbose:
            print(f"\n=== ИЗВЛЕЧЕНИЕ ===")
        
        # извлекаем 32 бита (длину сообщения)
        length_bits = []
        for idx in range(LENGTH_BITS):
            i = idx // self.width
            j = idx % self.width
            bit = int((self.image[i, j] >> bit_pos) & 1)
            length_bits.append(bit)
        
        # восстанавливаем длину
        msg_byte_count = 0
        for bit in length_bits:
            msg_byte_count = (msg_byte_count << 1) | bit
        
        if verbose:
            print(f"Извлеченная длина: {msg_byte_count} байт ({msg_byte_count*BITS_PER_BYTE} бит)")
        
        # проверка на разумность длины
        max_possible_bytes = (max_bits - LENGTH_BITS) // BITS_PER_BYTE
        if msg_byte_count <= 0 or msg_byte_count > max_possible_bytes:
            if verbose:
                print(f"ОШИБКА: Некорректная длина сообщения ({msg_byte_count} байт)")
            return ""
        
        # извлекаем биты сообщения
        total_bits = msg_byte_count * BITS_PER_BYTE
        message_bits = []
        for idx in range(total_bits):
            pos = idx + LENGTH_BITS
            i = pos // self.width
            j = pos % self.width
            if i >= self.height:
                break
            bit = int((self.image[i, j] >> bit_pos) & 1)
            message_bits.append(bit)
        
        # преобразуем биты обратно в байты
        message_bytes = bytearray()
        for i in range(0, len(message_bits) - 7, BITS_PER_BYTE):
            byte_val = 0
            for j in range(BITS_PER_BYTE):
                byte_val |= (message_bits[i + j] << (7 - j))
            message_bytes.append(byte_val)
        
        # декодируем UTF-8
        try:
            message = message_bytes.decode('utf-8')
            return message
        except UnicodeDecodeError as e:
            if verbose:
                print(f"Ошибка декодирования: {e}")
            return message_bytes.decode('utf-8', errors='replace')
    
    def get_info(self):
        """Возвращает информацию об изображении"""
        return {
            'width': self.width,
            'height': self.height,
            'total_pixels': self.width * self.height,
            'max_bytes': (self.width * self.height - LENGTH_BITS) // BITS_PER_BYTE,
            'image_path': self.image_path
        }