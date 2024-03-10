import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
from torchvision import datasets
from torchvision.transforms import ToTensor 
#import matplotlib.pyplot as plt
from models import SimpleLinearModel, LinearModel, MLP, MLP_softmax, CNN
from tqdm import tqdm 
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/mnist_experiment_4')
import sys
import argparse

# def accuracy(model, test_loader):
#     correct = 0 
#     for test_imgs, test_labels in test_loader:
#         test_imgs = Variable(test_imgs).float()
#         output = model(test_imgs)
#         predicted = torch.max(output,1)[1]
#         correct += (predicted == test_labels).sum()
#     print("Test accuracy:{:.3f}% ".format( float(correct) / (len(test_loader)*BATCH_SIZE)))

def load_MNIST_dataset():
    training_data = datasets.MNIST('./dataset', 
                            train = True, 
                            transform = ToTensor(), 
                            target_transform = None, 
                            download = True)

    test_data = datasets.MNIST('./dataset', 
                            train = False, 
                            transform = ToTensor(), 
                            target_transform = None, 
                            download = True)

    train_dataloader = DataLoader(training_data, batch_size = 1024, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size = 1024, shuffle=True)
    return train_dataloader, test_dataloader

def train(model, optimizer, criterion, begin_epoch, num_epochs, train_dataloader, test_dataloader, device):
    #epoch = 0 # initial epoch number 
    pre_val_loss = 10000000.0 # This is compared with a validation loss at each epoch. If the current one is larger than the previous one, then the model params are saved.

    # ===Training===
    for ep in range(begin_epoch+1, begin_epoch+num_epochs):
        model.train()
        print(f"Epoch: {ep}")

        cum_loss = 0.0
        for batch_index, (X, labels) in enumerate(tqdm(train_dataloader)):
            X, labels = X.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            #print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            cum_loss += loss.item()
        mean_loss = cum_loss / len(train_dataloader)
        print(f"    Mean loss at {ep}: {mean_loss}")

        #======Validation=========
        model.eval()
        cum_val_loss = 0.0 # To which validation loss of each batch was accumulated and is used to compute the mean validation loss over batches.
        with torch.no_grad():
            for X_val, labels_val in test_dataloader:
                X_val, labels_val = X_val.to(device), labels_val.to(device)
                outputs = model(X_val)
                val_loss = criterion(outputs, labels_val)
                cum_val_loss += val_loss.item()
        mean_val_loss = cum_val_loss / len(test_dataloader)
        print(f"    Mean validation loss for each batch (Epoch {ep}): {mean_val_loss}")
        
        if val_loss.item() < pre_val_loss:
            torch.save({
                    'epoch': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss},
                    f"model_dict.pth")
            
            print(f"    INFO: Epoch {ep}: Save the model parameters because of the decrease of validation loss.")
            pre_val_loss = val_loss
        
        # Save the model parameters at each epoch for restart training. 
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss},
            f"model_dict.cpt")

    print("FINISHED: Training's done!")

    return None


def main():
    # ====
    p = argparse.ArgumentParser()
    p.add_argument('-p', '--model_state_dict', help='Pretrained model parameter file (e.g.,`.pth`)', required=False, default=None) 
    p.add_argument('-e', '--num_epoch', required=False, default=10, type=int) 
    args = p.parse_args()
    model_state_dict = args.model_state_dict
    num_epochs = args.num_epoch
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    train_dataloader, test_dataloader = load_MNIST_dataset()

    # ==== Models ====
#    model = SimpleLinearModel().to(device)
#    model = LinearModel().to(device)
#    model = MLP().to(device)
    model = CNN().to(device)

    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.model_state_dict: 
        cpt_file = model_state_dict
        cpt = torch.load(cpt_file)
        model.load_state_dict(cpt['model_state_dict'])
        optimizer.load_state_dict(cpt['optimizer_state_dict'])
        begin_epoch = cpt['epoch']
        loss = cpt['loss']
        print("INFO: Restart training.")

    else:
        begin_epoch = 0
        print("INFO: Initial training's been started.")

    train(model, optimizer, criterion, begin_epoch, num_epochs, train_dataloader, test_dataloader, device)
 
main()

