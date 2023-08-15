import torch
import torch.nn as nn

class MLP(nn.Module): 
    def __init__(self, image_size, hidden_size, num_classes) : 
        # 상속 해주는 클래스를 부팅 
        super().__init__()
        
        self.image_size = image_size
        self.mlp1 = nn.Linear(in_features=image_size*image_size, out_features=hidden_size)
        self.mlp2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.mlp3 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.mlp4 = nn.Linear(in_features=hidden_size, out_features=num_classes)
    
    def forward(self, x) : 
        # x : [batch_size, 28, 28, 1] 
        batch_size = x.shape[0]
        # reshape 
        x = torch.reshape(x, (-1, self.image_size * self.image_size))
        # mlp1 ~ mlp4 진행 
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        # 출력 
        return x