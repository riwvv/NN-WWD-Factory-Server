import os
import json
import shutil
import numpy as np
import torch
import onnx
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .task_manager import update_task
from .embedding_extractor import (
    SAMPLE_RATE, NUM_SAMPLES, FEATURE_DIM, NUM_WINDOWS, EMBEDDING_DIM,
    WINDOW_SIZE, STEP_SIZE, MELSPEC_MODEL_PATH, EMBEDDING_MODEL_PATH,
)


class WakeWordHead(nn.Module):
    """Лёгкая голова-классификатор поверх заморожённого бэкенда openWakeWord
    (mel-спектрограмма -> Google speech_embedding, см. embedding_extractor.py).

    Бэкенд предобучен на огромном объёме речи и уже "знает", что такое речь
    вообще -- модели тут остаётся выучить только различение конкретного слова
    в этом готовом акустическом пространстве, а не акустику с нуля на паре
    тысяч клипов (ровно то, чего не хватало старой CNN-с-нуля)."""

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 64, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)  # сырые логиты — для CrossEntropyLoss при обучении


class WakeWordHeadInference(nn.Module):
    """Обёртка поверх WakeWordHead для экспорта: запекает нормализацию
    (train-статистика mean/std) и softmax прямо в граф. Так экспортированный
    onnx самодостаточен — на вход сырые (ненормализованные) эмбеддинги,
    downstream (C#, evaluate_model.py) не нужно знать про mean/std вообще."""

    def __init__(self, base_model: WakeWordHead, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.base_model = base_model

    def forward(self, x):
        x = (x - self.mean) / self.std
        logits = self.base_model(x)
        return nn.functional.softmax(logits, dim=1)


def _group_train_val_split(groups: np.ndarray, val_fraction: float = 0.2):
    """Делит индексы на train/val по группам целиком (а не по отдельным
    сэмплам) — аугментированные и time-shift копии одной записи (та же
    group_id, см. feature_extractor.py) всегда оказываются на одной стороне.
    Иначе почти неотличимые копии утекают между train и val, и val_acc
    перестаёт что-то значить."""
    unique_groups = np.unique(groups)
    shuffled = np.random.permutation(unique_groups)
    n_val_groups = max(1, int(len(shuffled) * val_fraction))
    val_groups = set(shuffled[:n_val_groups])

    val_mask = np.array([g in val_groups for g in groups])
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    return np.random.permutation(train_idx), np.random.permutation(val_idx)


def train_model(positive_features_path: str, negative_features_path: str,
                 positive_groups_path: str, negative_groups_path: str,
                 task_id: str = None, epochs: int = 20, batch_size: int = 32,
                 wake_word: str = "model", sample_rate: int = SAMPLE_RATE):
    """
    Обучает лёгкую голову поверх заморожённых эмбеддингов openWakeWord и
    экспортирует её в ONNX. Сам граф классификатора не содержит препроцессинг
    (mel/embedding) — это два отдельных заморожённых .onnx-файла
    (melspectrogram.onnx, embedding_model.onnx), которые копируются рядом
    с результатом и должны идти в комплекте при инференсе.

    Аргументы:
        positive_features_path: путь к positive_embeddings.npy
        negative_features_path: путь к negative_embeddings.npy
        positive_groups_path: путь к positive_groups.npy (для группового split)
        negative_groups_path: путь к negative_groups.npy (для группового split)
        task_id: идентификатор задачи для обновления статуса
        epochs: количество эпох
        batch_size: размер батча
        wake_word: слово для активации (используется в имени файла)
        sample_rate: частота дискретизации (для справки — фичи всегда считаются на SAMPLE_RATE)

    Возвращает:
        tuple: (путь к модели .onnx, путь к конфигу config.json)
    """
    try:
        if task_id:
            update_task(task_id, status="training", message="Загрузка признаков...")

        positive_embeddings = np.load(positive_features_path)
        negative_embeddings = np.load(negative_features_path)
        positive_groups = np.load(positive_groups_path)
        negative_groups = np.load(negative_groups_path)

        X = np.concatenate([positive_embeddings, negative_embeddings], axis=0)
        y = np.concatenate([
            np.ones(len(positive_embeddings)),
            np.zeros(len(negative_embeddings))
        ], axis=0)
        # Префикс классом — на случай, если у positive- и negative-файла
        # случайно совпадёт имя (после обрезки _aug{N}), группы не схлопнутся.
        groups = np.concatenate([
            np.char.add("pos:", positive_groups),
            np.char.add("neg:", negative_groups),
        ])

        # Нормализация по train-статистике — эмбеддинги openWakeWord не
        # отнормированы под нашу голову, без этого обучение неустойчиво.
        train_idx, val_idx = _group_train_val_split(groups)
        mu = X[train_idx].mean(axis=0)
        sigma = X[train_idx].std(axis=0) + 1e-6
        X = (X - mu) / sigma

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = WakeWordHead()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
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

        inference_model = WakeWordHeadInference(model, mu, sigma)
        inference_model.eval()

        dummy_input = torch.zeros(1, FEATURE_DIM, dtype=torch.float32)
        torch.onnx.export(
            inference_model,
            dummy_input,
            onnx_path,
            input_names=["features"],
            output_names=["probabilities"],
            opset_version=18,
            dynamo=True,
        )

        # dynamo-экспортёр по умолчанию выносит веса тензоров во внешний файл
        # {onnx_path}.data — для маленькой модели это не нужно и опасно (файл
        # с весами легко забыть при упаковке пакета). Перечитываем модель со
        # внешними данными и сохраняем как один самодостаточный .onnx-файл.
        exported_model = onnx.load(onnx_path, load_external_data=True)
        onnx.save_model(exported_model, onnx_path, save_as_external_data=False)

        external_data_path = onnx_path + ".data"
        if os.path.exists(external_data_path):
            os.remove(external_data_path)

        # Заморожённый бэкенд (mel + embedding) кладём рядом — без них config
        # и голова-классификатор бесполезны, инференс — это цепочка из трёх onnx.
        melspec_dest = os.path.join(os.getcwd(), "melspectrogram.onnx")
        embedding_dest = os.path.join(os.getcwd(), "embedding_model.onnx")
        shutil.copy(MELSPEC_MODEL_PATH, melspec_dest)
        shutil.copy(EMBEDDING_MODEL_PATH, embedding_dest)

        config = {
            "pipeline": "openwakeword_frozen_backbone",
            "sample_rate": SAMPLE_RATE,
            "num_samples": NUM_SAMPLES,
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "embedding_dim": EMBEDDING_DIM,
            "num_windows": NUM_WINDOWS,
            "feature_dim": FEATURE_DIM,
            "feature_mean": mu.tolist(),
            "feature_std": sigma.tolist(),
            "melspec_model_file": "melspectrogram.onnx",
            "embedding_model_file": "embedding_model.onnx",
            "classifier_model_file": onnx_filename,
            "classifier_input_name": "features",
            "classifier_output_name": "probabilities",
            "wake_word": wake_word,
            "model_name": base_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "best_val_acc": best_val_acc,
            "created_at": datetime.now().isoformat()
        }

        # Без explicit encoding="utf-8" Python на Windows пишет файл в
        # локальную кодировку консоли (cp1251 на русской локали) — кириллица
        # в wake_word/model_name ломает файл при чтении как UTF-8 downstream.
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        if task_id:
            update_task(task_id, status="completed", message="Обучение завершено!", file_path=onnx_path)

        return onnx_path, config_path

    except Exception as e:
        if task_id:
            update_task(task_id, status="failed", message=f"Ошибка обучения: {str(e)}")
        raise
