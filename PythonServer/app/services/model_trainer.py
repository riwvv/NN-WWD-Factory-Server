import os
import json
import numpy as np
import torch
import torchaudio
import onnx
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .task_manager import update_task

# Единый источник правды для параметров препроцессинга — те же значения
# используются при подготовке признаков (feature_extractor.py) и зашиваются
# в config.json для C#-рантайма. Никаких рассинхронов между train и inference:
# mel-спектрограмма считается внутри самой модели и экспортируется в ONNX
# вместе с CNN, так что математика гарантированно одна и та же везде.
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 160
N_MELS = 128
INPUT_WIDTH = 128
NUM_SAMPLES = SAMPLE_RATE * 1  # 1 секунда аудио на входе


class WakeWordModel(nn.Module):
    """CNN-детектор wake word со встроенным mel-фронтендом.

    На вход подаётся сырой waveform (batch, num_samples), а не готовая
    Mel-спектрограмма — вся математика STFT/Mel/CNN живёт в одном графе,
    поэтому экспорт в ONNX даёт идентичный препроцессинг и в Python, и в C#.
    """

    def __init__(self, sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
                 n_mels=N_MELS, input_width=INPUT_WIDTH, num_classes=2):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.input_width = input_width

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        h = n_mels // 4
        w = input_width // 4
        self.fc1 = nn.Linear(32 * h * w, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, waveform):
        # waveform: (batch, num_samples)
        mel = self.melspec(waveform)
        mel_db = self.amplitude_to_db(mel)

        t = mel_db.shape[-1]
        if t < self.input_width:
            mel_db = nn.functional.pad(mel_db, (0, self.input_width - t))
        else:
            mel_db = mel_db[..., : self.input_width]

        x = mel_db.unsqueeze(1)  # добавляем канал -> (batch, 1, n_mels, input_width)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # сырые логиты — для CrossEntropyLoss при обучении


class WakeWordInferenceModel(nn.Module):
    """Обёртка поверх WakeWordModel, добавляющая softmax — только для экспорта.
    Обучение всегда идёт на сырых логитах (WakeWordModel.forward), а наружу
    (в ONNX, для C#) отдаём уже готовые вероятности, чтобы не заставлять
    потребителя пакета самому делать softmax."""

    def __init__(self, base_model: WakeWordModel):
        super().__init__()
        self.base_model = base_model

    def forward(self, waveform):
        logits = self.base_model(waveform)
        return nn.functional.softmax(logits, dim=1)


def _pad_or_trim(waveform: np.ndarray, num_samples: int) -> np.ndarray:
    if len(waveform) < num_samples:
        return np.pad(waveform, (0, num_samples - len(waveform)))
    return waveform[:num_samples]


def train_model(positive_features_path: str, negative_features_path: str,
                 task_id: str = None, epochs: int = 20, batch_size: int = 32,
                 wake_word: str = "model", sample_rate: int = SAMPLE_RATE):
    """
    Обучает модель на сырых waveform-массивах и экспортирует готовый граф в ONNX.

    Аргументы:
        positive_features_path: путь к positive_waveforms.npy
        negative_features_path: путь к negative_waveforms.npy
        task_id: идентификатор задачи для обновления статуса
        epochs: количество эпох
        batch_size: размер батча
        wake_word: слово для активации (используется в имени файла)
        sample_rate: частота дискретизации (для справки — модель всегда обучается на SAMPLE_RATE)

    Возвращает:
        tuple: (путь к модели .onnx, путь к конфигу config.json)
    """
    try:
        if task_id:
            update_task(task_id, status="training", message="Загрузка признаков...")

        positive_waveforms = np.load(positive_features_path)
        negative_waveforms = np.load(negative_features_path)

        X = np.concatenate([positive_waveforms, negative_waveforms], axis=0)
        y = np.concatenate([
            np.ones(len(positive_waveforms)),
            np.zeros(len(negative_waveforms))
        ], axis=0)

        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = WakeWordModel()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_state_dict = None

        base_name = wake_word.replace(" ", "_").lower()
        onnx_filename = f"{base_name}_model.onnx"
        config_filename = f"{base_name}_config.json"

        onnx_path = os.path.join(os.getcwd(), onnx_filename)
        config_path = os.path.join(os.getcwd(), config_filename)

        for epoch in range(epochs):
            if task_id:
                update_task(task_id,
                    message=f"Обучение: эпоха {epoch + 1}/{epochs}",
                    progress=int((epoch + 1) / epochs * 100)
                )

            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                output = model(batch_x)  # сырые логиты
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    output = model(batch_x)  # сырые логиты
                    loss = criterion(output, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(output, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

            val_acc = correct / total
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # Экспортируем лучшую по val_acc версию модели (с softmax-обёрткой)
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        model.eval()
        model.to("cpu")

        inference_model = WakeWordInferenceModel(model)
        inference_model.eval()

        dummy_input = torch.zeros(1, NUM_SAMPLES, dtype=torch.float32)
        torch.onnx.export(
            inference_model,
            dummy_input,
            onnx_path,
            input_names=["waveform"],
            output_names=["probabilities"],
            opset_version=18,
            dynamo=True,
        )

        # dynamo-экспортёр по умолчанию выносит веса тензоров во внешний файл
        # {onnx_path}.data — для маленькой модели это не нужно и опасно (файл
        # с весами легко забыть при упаковке пакета, ровно как тут и вышло).
        # Перечитываем модель со внешними данными и сохраняем как один
        # самодостаточный .onnx-файл без внешних зависимостей.
        exported_model = onnx.load(onnx_path, load_external_data=True)
        onnx.save_model(exported_model, onnx_path, save_as_external_data=False)

        external_data_path = onnx_path + ".data"
        if os.path.exists(external_data_path):
            os.remove(external_data_path)

        config = {
            "input_height": N_MELS,
            "input_width": INPUT_WIDTH,
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "sample_rate": SAMPLE_RATE,
            "num_samples": NUM_SAMPLES,
            "onnx_input_name": "waveform",
            "onnx_output_name": "probabilities",
            "wake_word": wake_word,
            "model_name": base_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "best_val_acc": best_val_acc,
            "created_at": datetime.now().isoformat()
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        if task_id:
            update_task(task_id, status="completed", message="Обучение завершено!", file_path=onnx_path)

        return onnx_path, config_path

    except Exception as e:
        if task_id:
            update_task(task_id, status="failed", message=f"Ошибка обучения: {str(e)}")
        raise
