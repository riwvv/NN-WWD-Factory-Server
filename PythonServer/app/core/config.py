import os

# PythonServer/app/core/config.py -> подняться на 3 уровня до PythonServer/,
# чтобы REAL_SAMPLES_DIR не зависел от того, с каким рабочим каталогом
# запущен процесс (C#-клиент не всегда стартует его из папки PythonServer).
_PYTHONSERVER_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Settings:
    # Основные настройки
    AUDIO_OUTPUT_DIR: str = "generated_audio"
    DATASETS_DIR: str = "datasets"          # Папка для датасетов
    PACKAGE_DIR: str = "packages"           # Папка для готовых пакетов (ZIP)
    SAMPLE_RATES: list = [8000, 24000, 48000]
    DEFAULT_SAMPLE_RATE: int = 24000
    DEFAULT_SPEAKER: str = "xenia"
    DEFAULT_LANGUAGE: str = "ru"

    # Сюда пользователь вручную кладёт реальные записи (не синтетику TTS) —
    # они подмешиваются к сгенерированному датасету перед обучением.
    REAL_SAMPLES_DIR: str = os.path.join(_PYTHONSERVER_DIR, "real_samples")


settings = Settings()

# Заранее создаём папки, чтобы пользователю было очевидно, куда класть файлы,
# даже если он ни разу не запускал обучение.
os.makedirs(os.path.join(settings.REAL_SAMPLES_DIR, "positive"), exist_ok=True)
os.makedirs(os.path.join(settings.REAL_SAMPLES_DIR, "negative"), exist_ok=True)
