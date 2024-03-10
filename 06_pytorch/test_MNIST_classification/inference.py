#import torch 
from models import MLP, CNN
import torch
from torch.utils.data import Dataset, DataLoader 
from torchvision import datasets
from torchvision.transforms import ToTensor 
import matplotlib.pyplot as plt 
import numpy as np 
import sys 

model_dict = sys.argv[1]

training_data = datasets.MNIST('./dataset', 
                        train = True, 
                        transform = ToTensor(), 
                        target_transform = None, 
                        download = True)
training_loader = DataLoader(training_data, batch_size=1, shuffle=True)

model = CNN()
state_dict = torch.load(model_dict)
model.load_state_dict(state_dict['model_state_dict'])
model.eval()
with torch.no_grad():
    fig, axes = plt.subplots(1, 5)
    for i, (image, label) in enumerate(training_loader):
        if i == 5: break
    #    print(model)
        pred = model(image)# .detach().numpy()
        print(pred, label)
        pred = pred.argmax(dim=1)
        print(pred)
        axes[i].set_title(f"Ans: {label[0]}\nPred: {pred[0]}")
        axes[i].imshow(image.squeeze())
    plt.tight_layout()
    plt.show()
