import numpy as np
import Levenshtein

from app.utils.text_utils import (
    RUSSIAN_WORDS,
    generate_similar_words,
    generate_random_text,
    generate_noise,
)


def test_generate_similar_words_within_distance():
    word = "привет"
    similar = generate_similar_words(word, max_distance=2, length_variation=2)
    assert len(similar) > 0
    for candidate in similar:
        assert candidate != word


def test_generate_similar_words_single_letter_replacement_distance_one():
    word = "привет"
    similar = generate_similar_words(word, max_distance=2, length_variation=0)
    for candidate in similar:
        assert Levenshtein.distance(word, candidate) <= 2


def test_generate_random_text_returns_known_word():
    text = generate_random_text()
    assert text in RUSSIAN_WORDS


def test_generate_noise_shape_and_type():
    noise = generate_noise(sample_rate=16000, duration=0.5)
    assert isinstance(noise, np.ndarray)
    assert len(noise) == 8000
