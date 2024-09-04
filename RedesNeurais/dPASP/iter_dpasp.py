import pasp
import numpy as np 
import torchvision
import time
program_str = '''
% Addition of MNIST digits.

#python
import torch
import torchvision

class Net(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = torch.nn.Sequential(
      torch.nn.Conv2d(1, 6, 5),
      torch.nn.MaxPool2d(2, 2),
      torch.nn.ReLU(True),
      torch.nn.Conv2d(6, 16, 5),
      torch.nn.MaxPool2d(2, 2),
      torch.nn.ReLU(True)
    )
    self.classifier = torch.nn.Sequential(
      torch.nn.Linear(16 * 4 * 4, 120),
      torch.nn.ReLU(),
      torch.nn.Linear(120, 84),
      torch.nn.ReLU(),
      torch.nn.Linear(84, 10),
      torch.nn.Softmax(1)
    )

  def forward(self, x):
    x = self.encoder(x)
    x = x.view(-1, 16 * 4 * 4)
    x = self.classifier(x)
    return x

def digit_net(): return Net()

def mnist_data():
  train = torchvision.datasets.MNIST(root = "/tmp", train = True, download = True)
  test  = torchvision.datasets.MNIST(root = "/tmp", train = False, download = True)
  return train.data.float().reshape(len(train), 1, 28, 28)/255., train.targets, \
         test.data.float().reshape(len(test), 1, 28, 28)/255., test.targets

def normalize(X_R, Y_R, X_T, Y_T, mu, sigma):
  return (X_R-mu)/sigma, Y_R, (X_T-mu)/sigma, Y_T

train_X, train_Y, test_X, test_Y = normalize(*mnist_data(), 0.1307, 0.3081)
def pick_slice(data, which):
  h = len(data)//2
  return slice(h, len(data)) if which else slice(0, h)
def mnist_images_train(which): return train_X[pick_slice(train_X, which)]
def mnist_images_test(which): return test_X[pick_slice(test_X, which)]
def mnist_labels_train():
  labels = torch.concatenate((train_Y[:(h := len(train_Y)//2)].reshape(-1, 1),
                              train_Y[h:].reshape(-1, 1)), axis=1)
  return [[f"sum({x.item() + y.item()})"] for x, y in labels]
#end.

input(0) ~ test(@mnist_images_test(0)), train(@mnist_images_train(0)).
input(1) ~ test(@mnist_images_test(1)), train(@mnist_images_train(1)).

?::digit(X, {0..9}) as @digit_net with optim = "Adam", lr = 0.001 :- input(X).
sum(Z) :- digit(0, X), digit(1, Y), Z = X+Y.

#semantics maxent.
#learn @mnist_labels_train, lr = 1., niters = 13, alg = "lagrange", batch = 64.
#query sum(X).
'''

def mnist_labels():
  "Return the first and second digit values of the test set."
  Y = torchvision.datasets.MNIST(root="/tmp", train=False, download=True).targets
  return Y[:(h := len(Y)//2)].data.numpy(), Y[h:].data.numpy()

with open('resultspasp64.txt', 'a') as file: 
  for i in range(1,14):
    num = i
    new_str = program_str.replace('niters = 13', f'niters = {i}')

    t0 = time.time()
    P = pasp.parse(new_str, from_str=True)
    R = P(quiet=True)                        # Run program and capture output.
    Y = np.argmax(R, axis=1).reshape(len(R)) # Retrieve what sum is likeliest.
    D = mnist_labels()                       # D contains the digits of the test set.
    T = D[0] + D[1]                          # The ground-truth in the test set.
    accuracy = np.sum(Y == T)/len(T)         # Digit sum accuracy.
    t1 = time.time()
    t = t1 - t0
    file.write(f"Num de epochs: {i}, Accuracy: {accuracy},  Tempo: {t} \n")

