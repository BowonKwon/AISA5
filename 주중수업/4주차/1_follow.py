# 패키지 임포트
import torch # 파이토치 패키지 임포트
import torch.nn as nn # nn 패키지 임포트
from torchvision.datasets import MNIST # MNIST 데이터셋 불러오기
from torchvision.transforms import ToTensor # ToTensor 클래스 임포트
from torch.utils.data import DataLoader # DataLoader 클래스 임포트
from torch.optim import Adam # Adam 클래스 임포트

# hyperparameter 선언1: 이미지와 레이어 사이즈, 클래스 개수 설정
image_size = 28
hidden_size = 500
num_classes = 10
# hyperparameter 선언2: 학습 설정(배치 사이즈, 학습률, 에포크, 디바이스)
batch_size = 100
lr = 0.001
epochs = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 

# 모델 설계도 그리기
class MLP(nn.Module): # nn.Module을 상속받는 MLP 클래스 선언
    def __init__(self, image_size, hidden_size, num_classes): # 클래스 초기화: MLP 레이어 정의
        super().__init__() # 상속받은 상위 클래스의 초기화 메서드 호출
        self.image_size = image_size # 이미지 사이즈 저장
        self.mlp1 = nn.Linear(image_size * image_size, hidden_size) # 첫 번째 MLP 레이어 선언(입력층 -> 은닉층1)
        self.mlp2 = nn.Linear(hidden_size, hidden_size) # 두 번째 MLP 레이어 선언(은닉층1 -> 은닉층2)
        self.mlp3 = nn.Linear(hidden_size, hidden_size) # 세 번째 MLP 레이어 선언(은닉층2 -> 은닉층3)
        self.mlp4 = nn.Linear(hidden_size, num_classes) # 네 번째 MLP 레이어 선언(은닉층3 -> 출력층)
    def forward(self, x): # 순전파: 데이터가 레이어 통과하는 방식 지정
        batch_size = x.shape[0] # 입력 텐서의 배치 크기 저장(x: [batch_size, 28, 28, 1])
        x = torch.reshape(x, (-1, self.image_size * self.image_size)) # 28*28 픽셀 이미지를 1차원 벡터로 변환(펼치기)
        # 순전파 수행: 입력 이미지를 순차적으로 MLP 레이어에 통과시킴
        x = self.mlp1(x) # [batch_size, 500]
        x = self.mlp2(x) # [batch_size, 500]
        x = self.mlp3(x) # [batch_size, 500]
        x = self.mlp4(x) # [batch_size, 10]
        # 최종 출력 반환
        return x
 
# 설계도를 바탕으로 모델 만들기 <- hyperparmeter 사용 
myMLP = MLP(image_size, hidden_size, num_classes).to(device)

# 데이터 불러오기 
# dataset 설정(학습, 테스트 데이터)
train_mnist = MNIST(root='../../data/mnist', train=True, transform=ToTensor(), download=True)
test_mnist = MNIST(root='../../data/mnist', train=False, transform=ToTensor(), download=True)
# dataloader 설정(학습, 테스트 데이터)
train_loader = DataLoader(dataset=train_mnist, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_mnist, batch_size=batch_size, shuffle=False)

# Loss 선언 
loss_fn = nn.CrossEntropyLoss()
# Optimizer 선언
optim = Adam(params=myMLP.parameters(), lr=lr)

# 학습을 위한 반복 (Loop) for / while
for epoch in range(epochs):
# 입력할 데이터를 위해 데이터 준비 (dataloader)
    for idx, (images, targets) in enumerate(train_loader):
        # 데이터와 타깃을 디바이스에 올리기
        images = images.to(device)
        targets = targets.to(device)
        # 모델에 데이터를 넣기 
        output = myMLP(images)
        # 모델의 출력과 정답을 비교하기 (Loss 사용)
        loss = loss_fn(output, targets)
        # 역전파 수행
        loss.backward()
        # 가중치 업데이트
        optim.step()
        # 그래디언트 초기화
        optim.zero_grad()
        
        # 100번 반복마다 loss 출력
        if idx % 100 == 0:
            print(loss)