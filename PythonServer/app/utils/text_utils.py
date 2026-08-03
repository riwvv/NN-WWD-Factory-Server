# app/utils/text_utils.py
import os
import random
from typing import List

import numpy as np
import Levenshtein  # pip install python-Levenshtein

# Ручное перечисление слов не масштабируется — сколько ни добавляй,
# найдётся слово/междометие, которого не было в списке. Поэтому вместо
# продолжения ручного списка используем готовый частотный список русских
# слов (топ-3000 по частоте встречаемости в реальной речи — субтитры
# фильмов/сериалов, источник: hermitdave/FrequencyWords). Он естественным
# образом покрывает и обычную лексику, и междометия (ну, ой, ах, угу и т.п.),
# раз они реально часто встречаются в живой речи.
_WORDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "russian_words.txt")
with open(_WORDS_FILE, encoding="utf-8") as _f:
    _FREQUENCY_WORDS = [line.strip() for line in _f if line.strip()]

# Командные глаголы — то, что пользователь реально будет говорить после
# ключевого слова ("Юки, открой...", "Юки, включи..."), их держим отдельным
# списком явно, чтобы не полагаться на случай, попадут ли они в топ-3000.
_COMMAND_WORDS = [
    "открой", "закрой", "включи", "выключи", "запусти", "останови",
    "покажи", "найди", "сохрани", "удали", "отправь", "позвони",
    "напомни", "запиши", "прочитай", "переведи", "посчитай", "проверь",
    "подожди", "повтори", "стоп", "старт", "далее", "назад", "громче", "тише",
]

RUSSIAN_WORDS = _FREQUENCY_WORDS + _COMMAND_WORDS


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
