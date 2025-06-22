import torch
# Add the correct import for models (update the path as needed)
from torchreid import models
from torchvision import transforms
from reid_utils import load_pretrained_weights

# Define the image transformation pipeline
transform = transforms.Compose([
	transforms.Resize((256, 128)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
# 1. Detectar GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Mover modelo a GPU
reid_model = models.build_model(name='osnet_x1_0', num_classes=1000)
load_pretrained_weights(reid_model, 'osnet_x1_0_imagenet.pth')
reid_model.eval().to(device)

# 3. Mover tensores a GPU
img_tensor = transform(img).unsqueeze(0).to(device)