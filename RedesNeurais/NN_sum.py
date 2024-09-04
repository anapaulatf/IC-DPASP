'''
Rede Neural que recebe um tensor com dois canais, cada um com um número MNIST.
Cada canal é passado através da rede e devolve um tensor de tamanho 10, com 
respectivas probabilidades. 
É feita uma convolução 1d para computar as probabilidades da soma 
'''
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # Mudança no tamanho do kernel e adição de padding
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 11 * 4, 120),  # Ajuste das dimensões
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 19),  # 19 classes de saída
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 11 * 4)  # cria um tensor 16x11x4. 0 -1 coloca qlqr outra dim q eu tiver
        x = self.classifier(x)
        return x.log()