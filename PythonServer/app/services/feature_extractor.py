import os
import numpy as np
import librosa
from .task_manager import update_task

def extract_features_from_folder(folder_path: str, sample_rate: int = 16000, n_mels: int = 128, max_len: int = 128):
    """Извлекает Mel-спектрограммы из всех .wav файлов в папке."""
    features = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            audio, sr = librosa.load(file_path, sr=sample_rate)
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            if mel_spec_db.shape[1] < max_len:
                pad_width = max_len - mel_spec_db.shape[1]
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mel_spec_db = mel_spec_db[:, :max_len]
            
            features.append(mel_spec_db)
    
    return np.array(features)

async def prepare_features(dataset_dir: str, task_id: str = None):
    """Подготавливает признаки из датасета."""
    positive_dir = os.path.join(dataset_dir, "positive")
    negative_dir = os.path.join(dataset_dir, "negative")
    
    if task_id:
        update_task(task_id, message="Извлечение признаков из положительных примеров...")
    
    positive_features = extract_features_from_folder(positive_dir)
    
    if task_id:
        update_task(task_id, message="Извлечение признаков из отрицательных примеров...")
    
    negative_features = extract_features_from_folder(negative_dir)
    
    features_dir = os.path.join(dataset_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    
    positive_path = os.path.join(features_dir, "positive_features.npy")
    negative_path = os.path.join(features_dir, "negative_features.npy")
    
    np.save(positive_path, positive_features)
    np.save(negative_path, negative_features)
    
    return positive_path, negative_path