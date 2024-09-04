import numpy as np
from PIL import Image
from NN_sum import Net
import torch

model = Net()
model.load_state_dict(torch.load('RedesNeurais/Pesos/weights_mnist_double_sum.pth')) #carrega apenas os pesos do modelo
model.eval()

teste3 = np.asarray(Image.open('Imagens_Teste/3_teste2.png').convert("L"))
teste7 = np.asarray(Image.open('Imagens_Teste/7_teste3.png').convert("L"))
t = np.concatenate((teste3,teste7.T),axis=0)
t2 = torch.from_numpy(np.array([t/255])).float()
t2.shape
tr = model(t2)
print(tr.argmax())
