"""
Оценка качества обученной wake-word модели на фиксированном held-out
тестовом наборе (PythonServer/test_samples/positive и /negative).

Важно: test_samples — это НЕ то же самое, что real_samples. real_samples
подмешивается в обучение, test_samples — специально отложенный набор,
который модель никогда не видела при обучении. Только так цифры отсюда
что-то значат.

Пайплайн — цепочка из трёх onnx (см. app/services/embedding_extractor.py и
model_trainer.py): melspectrogram.onnx -> embedding_model.onnx (заморожённый
бэкенд openWakeWord) -> classifier ({wake_word}_model.onnx, наша голова).
Все три файла и config.json должны лежать в одной папке (так их упаковывает
/download-package) — так и оценивается: --model указывает на classifier onnx,
остальные два берутся из той же папки по именам в config.json.

Использование:
    python evaluate_model.py --model путь/к/юки_model.onnx --config путь/к/config.json
    python evaluate_model.py --model путь/к/юки_model.onnx --config путь/к/config.json --threshold 0.7
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import onnxruntime as ort


def load_and_prepare(path: str, num_samples: int, sample_rate: int) -> np.ndarray:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    if sr != sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

    if len(audio) < num_samples:
        audio = np.pad(audio, (0, num_samples - len(audio)))
    else:
        audio = audio[:num_samples]

    return audio


def waveform_to_embedding(waveform: np.ndarray, mel_sess: ort.InferenceSession,
                           emb_sess: ort.InferenceSession, config: dict) -> np.ndarray:
    """Воспроизводит app/services/embedding_extractor.py, но на сессиях,
    построенных из файлов, реально лежащих в пакете (а не из установленного
    пакета openwakeword) — так оценка честно проверяет то, что уедет в пакет."""
    window_size = config["window_size"]
    step_size = config["step_size"]
    embedding_dim = config["embedding_dim"]
    num_windows = config["num_windows"]

    pcm16 = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16).astype(np.float32)[None, :]
    mel = mel_sess.run(None, {mel_sess.get_inputs()[0].name: pcm16})[0]
    mel = np.squeeze(mel) / 10 + 2

    windows = []
    for i in range(0, mel.shape[0], step_size):
        w = mel[i:i + window_size]
        if w.shape[0] == window_size:
            windows.append(w)

    if not windows:
        return np.zeros(num_windows * embedding_dim, dtype=np.float32)

    batch = np.expand_dims(np.array(windows), axis=-1).astype(np.float32)
    emb = emb_sess.run(None, {emb_sess.get_inputs()[0].name: batch})[0]
    emb = emb.reshape(emb.shape[0], -1)

    if emb.shape[0] < num_windows:
        emb = np.pad(emb, ((0, num_windows - emb.shape[0]), (0, 0)))
    else:
        emb = emb[:num_windows]

    return emb.flatten().astype(np.float32)


def evaluate_folder(sessions: dict, folder: str, config: dict,
                     threshold: float, expected_label: str) -> list[dict]:
    results = []
    if not os.path.isdir(folder):
        return results

    files = [f for f in os.listdir(folder) if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a"))]
    for filename in files:
        path = os.path.join(folder, filename)
        try:
            waveform = load_and_prepare(path, config["num_samples"], config["sample_rate"])
        except Exception as e:
            print(f"  ОШИБКА чтения {filename}: {e}")
            continue

        # Нормализация (train-статистика mean/std) запечена прямо в classifier
        # onnx при экспорте (см. model_trainer.py) — тут просто сырые эмбеддинги.
        features = waveform_to_embedding(waveform, sessions["mel"], sessions["emb"], config)

        input_tensor = features.reshape(1, -1).astype(np.float32)
        output = sessions["classifier"].run(None, {config["classifier_input_name"]: input_tensor})
        probabilities = output[0][0]
        positive_prob = float(probabilities[1])
        detected = positive_prob > threshold

        results.append({
            "file": filename,
            "expected": expected_label,
            "detected": detected,
            "probability": positive_prob,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Оценка wake-word модели на held-out тестовом наборе")
    parser.add_argument("--model", required=True, help="Путь к classifier .onnx файлу ({wake_word}_model.onnx)")
    parser.add_argument("--config", required=True, help="Путь к config.json")
    parser.add_argument("--test-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_samples"))
    parser.add_argument("--threshold", type=float, default=None, help="Порог срабатывания (по умолчанию — 0.8, как в C#-детекторе)")
    args = parser.parse_args()

    # C# записывает JSON с BOM — utf-8-sig корректно его съедает
    with open(args.config, encoding="utf-8-sig") as f:
        config = json.load(f)

    threshold = args.threshold if args.threshold is not None else 0.8

    package_dir = os.path.dirname(os.path.abspath(args.model))
    melspec_path = os.path.join(package_dir, config["melspec_model_file"])
    embedding_path = os.path.join(package_dir, config["embedding_model_file"])

    sessions = {
        "mel": ort.InferenceSession(melspec_path, providers=["CPUExecutionProvider"]),
        "emb": ort.InferenceSession(embedding_path, providers=["CPUExecutionProvider"]),
        "classifier": ort.InferenceSession(args.model, providers=["CPUExecutionProvider"]),
    }

    positive_dir = os.path.join(args.test_dir, "positive")
    negative_dir = os.path.join(args.test_dir, "negative")

    positive_results = evaluate_folder(sessions, positive_dir, config, threshold, "positive")
    negative_results = evaluate_folder(sessions, negative_dir, config, threshold, "negative")

    if not positive_results and not negative_results:
        print(f"Тестовый набор пуст. Положи файлы в:\n  {positive_dir}\n  {negative_dir}")
        return

    print(f"\n{'=' * 60}")
    print(f"Модель: {args.model}")
    print(f"Порог срабатывания: {threshold}")
    print(f"{'=' * 60}\n")

    # --- Positive (должны быть обнаружены) ---
    true_positives = sum(1 for r in positive_results if r["detected"])
    false_negatives = len(positive_results) - true_positives
    print(f"POSITIVE ({len(positive_results)} файлов — реально сказанное ключевое слово):")
    print(f"  Поймано:  {true_positives}/{len(positive_results)} ({100 * true_positives / max(len(positive_results), 1):.1f}%)")
    print(f"  Пропущено: {false_negatives}")
    for r in sorted(positive_results, key=lambda r: r["probability"]):
        if not r["detected"]:
            print(f"    ПРОПУЩЕНО: {r['file']} (probability={r['probability']:.4f})")

    print()

    # --- Negative (не должны быть обнаружены) ---
    false_positives = sum(1 for r in negative_results if r["detected"])
    true_negatives = len(negative_results) - false_positives
    print(f"NEGATIVE ({len(negative_results)} файлов — шум/слова/тишина, НЕ ключевое слово):")
    print(f"  Верно отвергнуто: {true_negatives}/{len(negative_results)} ({100 * true_negatives / max(len(negative_results), 1):.1f}%)")
    print(f"  Ложных срабатываний: {false_positives}")
    for r in sorted(negative_results, key=lambda r: -r["probability"]):
        if r["detected"]:
            print(f"    ЛОЖНОЕ СРАБАТЫВАНИЕ: {r['file']} (probability={r['probability']:.4f})")

    print(f"\n{'=' * 60}")
    total = len(positive_results) + len(negative_results)
    correct = true_positives + true_negatives
    print(f"ИТОГО: {correct}/{total} правильно ({100 * correct / max(total, 1):.1f}%)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
