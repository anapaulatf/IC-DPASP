import torch
import torch.nn as nn

#arquitetura da rede neural - recebe imagem 28x28
class Net(nn.Module):
  def __init__(self):
    super().__init__()

    self.encoder = nn.Sequential(
      nn.Conv2d(1, 6, 5), 
      nn.MaxPool2d(2, 2),
      nn.ReLU(True),
      nn.Conv2d(6, 16, 5),
      nn.MaxPool2d(2, 2),
      nn.ReLU(True)
    )
    self.classifier = nn.Sequential(
      nn.Linear(16 * 4 * 4, 120),
      nn.ReLU(),
      nn.Linear(120, 84),
      nn.ReLU(),
      nn.Linear(84, 10),
      nn.Softmax(1) 
    )

  def forward(self, x):
    #primeiro canal
    x1 = x[:,0,:,:].unsqueeze(1) #primeiro canal [N,1,28,28]
    x1 = self.encoder(x1)
    x1 = x1.view(-1, 16 * 4 * 4)
    x1 = self.classifier(x1) #[64,10]

    #segundo canal
    x2 = x[:,1,:,:].unsqueeze(1) #segundo canal [N,1,28,28]
    x2 = self.encoder(x2)
    x2 = x2.view(-1, 16 * 4 * 4)
    x2 = self.classifier(x2) #[64,10]

    # Convolução
    Z = []
    for i in range(len(x2)): 
      BATCH_SIZE, IN_CH, OUT_CH = 1, 1, 1
      kernel_size = len(x2[i])

      # Pad ensure all non-zero outputs are computed.
      h = nn.Conv1d(IN_CH, OUT_CH, kernel_size=kernel_size, padding= kernel_size - 1, bias=False)
      with torch.no_grad():
        h.weight.copy_(torch.flip(x2[i], dims=[0]).reshape(OUT_CH, IN_CH, -1))
      z = h(x1[i].reshape(BATCH_SIZE, IN_CH, -1)).reshape(-1)
      Z.append(z)

    Z = torch.stack(Z)
    return Z.log()