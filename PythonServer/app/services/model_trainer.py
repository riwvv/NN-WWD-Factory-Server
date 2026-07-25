import os
import json
import numpy as np
import torch
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .task_manager import update_task
from app.core.config import settings

class WakeWordModel(nn.Module):
    """Простая свёрточная модель для детекции wake word."""
    def __init__(self, input_shape, num_classes=2):
        super().__init__()
        # input_shape: (height, width) например (128, 128)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Вычисляем размер после свёрток и пулинга
        h, w = input_shape
        h = h // 4  # два пулинга
        w = w // 4
        self.fc1 = nn.Linear(32 * h * w, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def train_model(positive_features_path: str, negative_features_path: str, 
                task_id: str = None, epochs: int = 20, batch_size: int = 32,
                wake_word: str = "model", sample_rate: int = 16000):
    """
    Обучает модель на основе признаков и сохраняет конфиг.
    
    Аргументы:
        positive_features_path: путь к positive_features.npy
        negative_features_path: путь к negative_features.npy
        task_id: идентификатор задачи для обновления статуса
        epochs: количество эпох
        batch_size: размер батча
        wake_word: слово для активации (используется в имени файла)
        sample_rate: частота дискретизации
    
    Возвращает:
        tuple: (путь к модели .pth, путь к конфигу config.json)
    """
    try:
        # 1. Загрузка признаков
        if task_id:
            update_task(task_id, status="training", message="Загрузка признаков...")
        
        positive_features = np.load(positive_features_path)
        negative_features = np.load(negative_features_path)
        
        # 2. Подготовка данных
        X = np.concatenate([positive_features, negative_features], axis=0)
        y = np.concatenate([
            np.ones(len(positive_features)),
            np.zeros(len(negative_features))
        ], axis=0)
        
        # Перемешиваем данные
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # 3. Разделение на train/val (80/20)
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # 4. Преобразование в тензоры
        X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
        y_val = torch.tensor(y_val, dtype=torch.long)
        
        # 5. Создание DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 6. Инициализация модели
        input_shape = X_train.shape[2:]  # (height, width)
        model = WakeWordModel(input_shape)
        
        # 7. Настройка обучения
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 8. Обучение
        best_val_acc = 0.0
        
        # Имена файлов с учётом wake_word
        base_name = wake_word.replace(" ", "_").lower()
        model_filename = f"{base_name}_model.pth"
        config_filename = f"{base_name}_config.json"
        
        os.makedirs(settings.PACKAGE_DIR, exist_ok=True)
        model_path = os.path.join(settings.PACKAGE_DIR, model_filename)
        config_path = os.path.join(settings.PACKAGE_DIR, config_filename)
        
        for epoch in range(epochs):
            if task_id:
                update_task(task_id, 
                    message=f"Обучение: эпоха {epoch + 1}/{epochs}",
                    progress=int((epoch + 1) / epochs * 100)
                )
            
            # Train
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(output, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            val_acc = correct / total
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
            
            # Сохраняем лучшую модель
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_path)
        
        # 9. Сохраняем конфиг модели
        config = {
            "input_height": input_shape[0],
            "input_width": input_shape[1],
            "n_mels": 128,
            "n_fft": 512,
            "hop_length": 160,
            "sample_rate": sample_rate,
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
            update_task(task_id, status="completed", message="Обучение завершено!", file_path=model_path)
        
        # Возвращаем оба пути
        return model_path, config_path
        
    except Exception as e:
        if task_id:
            update_task(task_id, status="failed", message=f"Ошибка обучения: {str(e)}")
        raise