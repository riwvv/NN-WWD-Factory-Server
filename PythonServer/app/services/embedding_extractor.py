import os
import pathlib
import numpy as np
import onnxruntime as ort
import openwakeword

# Заморожённый бэкенд openWakeWord: mel-спектрограмма -> скользящие окна из
# 76 фреймов с шагом 8 -> Google speech_embedding (96-мерный вектор на окно).
# Обучаем поверх ЭТИХ признаков только маленькую голову (см. model_trainer.py),
# а не CNN с нуля -- бэкенд предобучен на огромном объёме речи и уже умеет
# то, что CNN с нуля пришлось бы выучивать заново на паре тысяч клипов.
SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE * 1  # 1 секунда клипа, как и раньше
WINDOW_SIZE = 76
STEP_SIZE = 8
EMBEDDING_DIM = 96
# Для 1-секундного клипа (97 mel-фреймов) окно=76/шаг=8 даёт ровно 3 окна.
NUM_WINDOWS = (97 - WINDOW_SIZE) // STEP_SIZE + 1
FEATURE_DIM = NUM_WINDOWS * EMBEDDING_DIM

_MODELS_DIR = os.path.join(pathlib.Path(openwakeword.__file__).parent.resolve(), "resources", "models")
MELSPEC_MODEL_PATH = os.path.join(_MODELS_DIR, "melspectrogram.onnx")
EMBEDDING_MODEL_PATH = os.path.join(_MODELS_DIR, "embedding_model.onnx")

_mel_session = None
_emb_session = None


def _get_sessions():
    global _mel_session, _emb_session
    if _mel_session is None:
        _mel_session = ort.InferenceSession(MELSPEC_MODEL_PATH)
        _emb_session = ort.InferenceSession(EMBEDDING_MODEL_PATH)
    return _mel_session, _emb_session


def waveform_to_embedding(waveform: np.ndarray) -> np.ndarray:
    """Превращает 1-секундный waveform (float32, [-1, 1], 16кГц, уже
    дополненный/обрезанный до NUM_SAMPLES) в фиксированный вектор признаков
    FEATURE_DIM, конкатенируя эмбеддинги всех скользящих окон клипа."""
    mel_sess, emb_sess = _get_sessions()

    # melspectrogram.onnx ожидает 16-бит PCM (как и в референсной реализации
    # openWakeWord), а не float — иначе распределение входа не совпадёт с тем,
    # на чём предобучен бэкенд.
    pcm16 = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16).astype(np.float32)[None, :]
    mel_input_name = mel_sess.get_inputs()[0].name
    mel = mel_sess.run(None, {mel_input_name: pcm16})[0]
    mel = np.squeeze(mel)  # (time, 32)
    mel = mel / 10 + 2  # документированный трансформ openWakeWord под embedding-модель

    windows = []
    for i in range(0, mel.shape[0], STEP_SIZE):
        w = mel[i:i + WINDOW_SIZE]
        if w.shape[0] == WINDOW_SIZE:
            windows.append(w)

    if not windows:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    batch = np.expand_dims(np.array(windows), axis=-1).astype(np.float32)  # (n, 76, 32, 1)
    emb_input_name = emb_sess.get_inputs()[0].name
    emb = emb_sess.run(None, {emb_input_name: batch})[0]
    emb = emb.reshape(emb.shape[0], -1)  # (n, 96)

    # Фиксируем размер на случай, если длина клипа даёт чуть другое число окон.
    if emb.shape[0] < NUM_WINDOWS:
        emb = np.pad(emb, ((0, NUM_WINDOWS - emb.shape[0]), (0, 0)))
    else:
        emb = emb[:NUM_WINDOWS]

    return emb.flatten().astype(np.float32)
