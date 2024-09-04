import torch
from torch.utils.data import Dataset 

class MNISTDoubleDataset(Dataset):
    '''Dataset personalizado com MNIST "normal" e MNIST embaralhado.
    Eles são concatenados formando tensores de 56x28.
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

        #concatena dos dados 
        image = torch.cat((img, shuffled_img),1)
        return image, sum_label
