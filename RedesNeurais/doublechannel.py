import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class MNISTDoubleChannel(Dataset):
    '''Dataset personalizado com MNIST "normal" e MNIST embaralhado.
    O label consiste na soma dos labels'''
    
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        #self.indices = list(range(len(mnist_dataset)))
        self.shuffled_indices = torch.randperm(len(self))  # gera índices aleatórios

    def __len__(self):
        return len(self.mnist_dataset)
    
    def __getitem__(self, idx):
      # imagem e label normais 
      img, label = self.mnist_dataset[idx]
      
      # imagem e label com o índice aleatório
      shuffled_idx = self.shuffled_indices[idx]
      shuffled_img, shuffled_label = self.mnist_dataset[shuffled_idx]
      
      # calcula a soma dos labels 
      sum_label = label + shuffled_label

      # adiciona uma dimensão extra para cada imagem
      #img = img.unsqueeze(0)  # [1, 28, 28]
      #shuffled_img = shuffled_img.unsqueeze(0)  # [1, 28, 28]

      # concatena as imagens na nova dimensão
      #concat = torch.cat([img, shuffled_img], dim=0)
      concat = torch.cat([img, shuffled_img], dim=0)
      #concat = concat.permute(1, 0, 2, 3).squeeze(0)
      
      return concat, sum_label
    
'''

mnist_train = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
mnist_test = datasets.MNIST('./data', train=False, download=True,
                            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

# cria os dados customizados e dataloader
mnist_double_train = MNISTDoubleChannel(mnist_train)

#verificar tamanhos
tensor,label = mnist_double_train[0]
print(tensor.shape)

#teste

idsh = torch.randperm(len(mnist_train))
img, label = mnist_train[0]
idsh0 = idsh[0]
imgx,labelx = mnist_train[idsh0]

sum = label + labelx

Img = torch.cat([img, imgx], dim=0) # [2, 28, 28]
'''