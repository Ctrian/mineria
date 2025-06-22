import torch
import pickle
import cv2
import os
import numpy as np
from torchreid import models
from torchvision import transforms
from PIL import Image
from sklearn.cluster import KMeans

def load_pretrained_weights(model, weights_path):
    """
    Carga los pesos preentrenados en el modelo especificado.

    Args:
        model: El modelo de PyTorch al que se le cargarán los pesos.
        weights_path (str): Ruta al archivo .pth de los pesos preentrenados.
    """
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    print(f"Pesos cargados desde {weights_path}")

# Inicializa OSNet
def load_reid_model():
    model = models.build_model(name='osnet_x1_0', num_classes=1000)
    load_pretrained_weights(model, r'D:\Mineria de Datos\mineria_video_recognition-main\mineria_video_recognition-main\osnet_x1_0_imagenet.pth')
    model.eval().cuda()
    return model

def get_dominant_color(image, k=3):
    """
    Extrae el color dominante del crop (imagen BGR).
    """
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)]
    return dominant.astype(int).tolist()  # Devuelve lista [B, G, R]

# Preprocesamiento de imagen
def preprocess_image(crop):
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return transform(img).unsqueeze(0).cuda()

# Obtener embedding
def get_embedding(model, crop):
    with torch.no_grad():
        input_tensor = preprocess_image(crop)
        embedding = model(input_tensor)
    return embedding.squeeze().cpu().numpy()

def is_same_person(embedding1, embedding2, threshold=0.75):
    """
    Compara dos embeddings usando similitud de coseno.
    """
    a = np.asarray(embedding1)
    b = np.asarray(embedding2)
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return similarity > threshold  # cuanto más alto, más parecidos
# Cargar base de datos
def load_database(path='person_db.pkl'):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return []

# Guardar base de datos
def save_database(db, path='person_db.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(db, f)