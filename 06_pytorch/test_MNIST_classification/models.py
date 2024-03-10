import torch.nn as nn 
import torch.nn.functional as F

class SimpleLinearModel(nn.Module):

    def __init__(self, input_features=28*28, output_features=10):
        super(SimpleLinearModel, self).__init__()
        self.l1 = nn.Linear(input_features, 28)
        self.l2 = nn.Linear(28, output_features)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y = F.relu(self.l1(x))
        y = self.l2(y)
        #y = F.relu(y)
        return y
    

class LinearModel(nn.Module):

    def __init__(self, input_features=28*28, output_features=10):
        super(LinearModel, self).__init__()
        self.l1 = nn.Linear(input_features, output_features)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y = self.l1(x)
        return y
    
class MLP(nn.Module):
    # Three layers
    def __init__(self, input_features=28*28, output_features=10):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(input_features, 500)
        self.l2 = nn.Linear(500, 100)
        self.l3 = nn.Linear(100, output_features)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        y = self.l3(y)
        return y
    
class MLP_softmax(nn.Module):
    # Three layers
    def __init__(self, input_features=28*28, output_features=10):
        super(MLP_softmax, self).__init__()
        self.l1 = nn.Linear(input_features, 500)
        self.l2 = nn.Linear(500, 100)
        self.l3 = nn.Linear(100, output_features)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        y = self.l3(y)
        y = F.softmax(y)
        return y
    

class CNN(nn.Module):
    def __init__(self, in_channels=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10) 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 3*3*64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x) 
        return F.log_softmax(x, dim=1)