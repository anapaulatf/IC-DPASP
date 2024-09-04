#carrega bibliotecas
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from doublechannel import MNISTDoubleChannel
from NN_conv import Net
import matplotlib.pyplot as plt
import time

#carrega dados
mnist_train = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
mnist_test = datasets.MNIST('./data', train=False, download=True,
                            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

# cria os dados customizados e dataloader
mnist_double_train = MNISTDoubleChannel(mnist_train)
train_loader = DataLoader(mnist_double_train, batch_size=64, shuffle=False)

# faz o mesmo para os dados de teste
mnist_double_test = MNISTDoubleChannel(mnist_test)
test_loader = DataLoader(mnist_double_test, batch_size=64, shuffle=False)


model = Net()

'''TREINAMENTO'''
# Hiperparâmetros
learning_rate = 0.15e-3
epochs = 13
loss_fn =  nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Listas para armazenar resultados
train_losses = []
test_losses = []
test_accuracies = []
tempo = [0]

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    train_loss = 0
    for epoch, (X, y) in enumerate(dataloader):
        # Calcula prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)  

        # Backpropagation
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        
        train_loss += loss.item()

        if epoch % 100 == 0:
            loss, current = loss.item(), (epoch+1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_losses.append(train_loss / len(dataloader))

# teste
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    #Garante que não serão computados gradientes durante o teste
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    test_losses.append(test_loss)
    test_accuracies.append(correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    t0 = time.time()
    train_loop(train_loader, model, loss_fn, optimizer)
    t1 = time.time()
    test_loop(test_loader, model, loss_fn)
    t3 = tempo[-1]
    tempo.append(t1-t0+t3)
print("Done!")

# Plotando os resultados
plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss for Epoch')

# Acurácia
plt.subplot(1, 2, 2)
plt.plot([100 * acc for acc in test_accuracies], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy for Epoch')
plt.suptitle("Metrics - learning rate 0.15e-3")
plt.savefig('RedesNeurais/Graficos/graph_sum_conv2.png')


# Exportar dados do modelo
import pandas as pd
tempo = tempo.pop(0)
df = pd.DataFrame({
    'train_losses_conv' : train_losses, 
    'test_losses_conv' : test_losses,
    'test_accuracies' : test_accuracies,
    'time': tempo
})

df.to_csv('RedesNeurais/Results/results3_conv.csv', index = False)

