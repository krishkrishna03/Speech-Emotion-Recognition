import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, Reshape
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ❗ Force GPU only
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    raise RuntimeError("❌ No GPU found. This model must run on GPU only.")

tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
print("✅ GPU detected and set for training.")

# Emotion label map
emotion_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5
}

# Feature extraction
def extract_features(file_path, max_pad_len=100):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc

# Load dataset
def load_dataset(directory):
    X, y = [], []
    print("Loading dataset...")
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            for file in tqdm(os.listdir(folder_path), desc=f"Loading {folder}"):
                if file.endswith(".wav"):
                    file_path = os.path.join(folder_path, file)
                    try:
                        features = extract_features(file_path)
                        X.append(features)
                        y.append(folder.lower())
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
    return np.array(X), np.array(y)

# Load data
DATASET_PATH = "data"
X, y = load_dataset(DATASET_PATH)
print(f"✅ Dataset loaded: {X.shape[0]} samples")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Prepare input shape
X = X.reshape(X.shape[0], 40, 100, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    InputLayer(input_shape=(40, 100, 1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("clstm_speech_emotion_model1.h5")
print("✅ Model trained and saved.")
