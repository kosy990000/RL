import torch
import torch.nn as nn
import torch.nn.functional as F


class normalCNNActionValue(nn.Module):
    def __init__(self, state_dim, action_dim, activation=F.relu):
        super(normalCNNActionValue, self).__init__()
        
        #  고정
        seed = 42
        torch.manual_seed(seed)

        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)  # [N, 4, 84, 84] -> [N, 16, 20, 20]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # [N, 16, 20, 20] -> [N, 32, 9, 9]
        self.in_features = 32 * 9 * 9
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)
        self.activation = activation

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    

class advCNNActionValue(nn.Module):
    def __init__(self, state_dim, action_dim, activation=F.relu):
        super(advCNNActionValue, self).__init__()
        
        seed = 42
        torch.manual_seed(seed)

        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=3, stride=2, padding=1)  # [N, 4, 84, 84] -> [N, 16, 42, 42]
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # [N, 16, 42, 42] -> [N, 16, 21, 21]


        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # [N, 16, 21, 21] -> [N, 32, 21, 21]
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # [N, 32, 21, 21] -> [N, 32, 10, 10] 여기서 feature 버려질 수 있음음


        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # [N, 32, 10, 10] -> [N, 64, 10, 10]
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2) # [N, 64, 10, 10] -> [N, 64, 5, 5]


        # FC
        self.in_features = 64 * 5 * 5  # [N, 64, 5, 5]
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.activation(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.activation(self.bn3(self.conv3(x)))
        x = self.pool3(x)
 
        x = x.view((-1, self.in_features))

        x = self.fc1(x)
        x = self.fc2(x)
        return x
