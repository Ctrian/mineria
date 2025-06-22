import torch
print("¿GPU disponible?", torch.cuda.is_available())
print("Versión CUDA usada:", torch.version.cuda)