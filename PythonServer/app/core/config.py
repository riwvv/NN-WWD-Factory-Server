import os

class Settings:
    # Основные настройки
    AUDIO_OUTPUT_DIR: str = "generated_audio"
    DATASETS_DIR: str = "datasets"          # Папка для датасетов
    PACKAGE_DIR: str = "packages"           # Папка для готовых пакетов (ZIP)
    SAMPLE_RATES: list = [8000, 24000, 48000]
    DEFAULT_SAMPLE_RATE: int = 24000
    DEFAULT_SPEAKER: str = "xenia"
    DEFAULT_LANGUAGE: str = "ru"

settings = Settings()