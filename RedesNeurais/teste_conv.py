import numpy as np
from PIL import Image
from NN_conv import Net
import torch

model = Net()
model.load_state_dict(torch.load('RedesNeurais/Pesos/weights_CNN_mnist_sum.pth')) #carrega apenas os pesos do modelo
print(model.eval())

teste5 = torch.from_numpy(np.array([np.asarray(Image.open('Imagens_Teste/5_teste.png').convert("L"))/255])).float().unsqueeze(1)
teste7 = torch.from_numpy(np.array([np.asarray(Image.open('Imagens_Teste/7_teste3.png').convert("L"))/255])).float().unsqueeze(1)

t = torch.cat([teste5, teste7], dim=1)
tr = model(t)
print(tr.argmax())
