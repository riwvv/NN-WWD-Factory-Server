# app/utils/text_utils.py
import random
from typing import List

import numpy as np
import Levenshtein  # pip install python-Levenshtein

RUSSIAN_WORDS = [
    "привет", "пока", "компьютер", "музыка", "время", "работа", "дом", "день",
    "ночь", "окно", "стол", "стул", "вода", "огонь", "земля", "небо",
    "человек", "город", "улица", "машина", "книга", "фильм", "игра"
]


def generate_similar_words(word: str, max_distance: int = 2,
                            alphabet: str = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя',
                            length_variation: int = 2) -> List[str]:
    """Генерирует слова, похожие на заданное, с вариациями длины."""
    similar = set()

    # 1. Замены букв (как было раньше)
    for i in range(len(word)):
        for letter in alphabet:
            if letter != word[i]:
                new_word = word[:i] + letter + word[i+1:]
                if Levenshtein.distance(word, new_word) <= max_distance:
                    similar.add(new_word)

    # 2. Добавление случайных букв (+1..+length_variation символов)
    for extra in range(1, length_variation + 1):
        for _ in range(10):
            insert_pos = random.randint(0, len(word))
            random_letter = random.choice(alphabet)
            new_word = word[:insert_pos] + random_letter + word[insert_pos:]
            if Levenshtein.distance(word, new_word) <= max_distance + extra:
                similar.add(new_word)

    # 3. Удаление букв (-1..-length_variation символов)
    for remove in range(1, length_variation + 1):
        for _ in range(10):
            if len(word) - remove > 0:
                remove_pos = random.randint(0, len(word) - remove)
                new_word = word[:remove_pos] + word[remove_pos + remove:]
                if Levenshtein.distance(word, new_word) <= max_distance + remove:
                    similar.add(new_word)

    return list(similar)


def generate_random_text() -> str:
    """Генерирует случайный текст для отрицательного примера."""
    return random.choice(RUSSIAN_WORDS)


def generate_noise(sample_rate: int = 24000, duration: float = 1.0) -> np.ndarray:
    """Генерирует белый шум."""
    return np.random.normal(0, 0.1, int(sample_rate * duration))
