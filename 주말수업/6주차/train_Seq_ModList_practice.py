# 패키지 임포트
import os                                   # os 패키지 임포트
import torch                                # 파이토치 패키지 임포트
import torch.nn as nn                       # nn 패키지 임포트
from torchvision.datasets import MNIST      # MNIST 데이터셋 불러오기
from torchvision.transforms import ToTensor # ToTensor 클래스 임포트
from torch.utils.data import DataLoader     # DataLoader 클래스 임포트
from torch.optim import Adam                # Adam 클래스 임포트

from collections import OrderedDict         # OrderedDict 클래스 임포트

# hyperparameter 선언(학습률, 이미지 사이즈, 클래스 개수, 배치 사이즈, 은닉층 사이즈, 에포크 수, 결과 저장 폴더)
lr = 0.001
image_size = 28
num_classes = 10
batch_size = 100
hidden_size = 500
epochs = 3
results_folder = 'results'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 디바이스 설정

# 저장
# 상위 저장 폴더가 없으면 상위 저장 폴더 생성
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
# 결과 저장할 하위 타깃 폴더 생성
target_folder_name = max([0] + [int(e) for e in os.listdir(results_folder)])+1  # 하위 타깃 폴더 이름
target_folder = os.path.join(results_folder, str(target_folder_name))           # 하위 타깃 폴더 경로
os.makedirs(target_folder)                                                      # 하위 타깃 폴더 생성
# 타깃 폴더에 하이퍼파라미터 저장
with open(os.path.join(target_folder, 'hparam.txt'), 'w') as f:                 # 타깃 폴더에 hparam.txt 파일 생성
    f.write(f'{lr}\n')
    f.write(f'{image_size}\n')
    f.write(f'{num_classes}\n') 
    f.write(f'{batch_size}\n') 
    f.write(f'{hidden_size}\n') 
    f.write(f'{epochs}\n') 
    f.write(f'{results_folder}\n') 

# 모델 설계도 그리기
class MLP(nn.Module): # nn.Module을 상속받는 MLP 클래스 선언
    def __init__(self, image_size, hidden_size, num_classes): # 클래스 초기화: MLP 레이어 정의
        super().__init__() # 상속받은 상위 클래스의 초기화 메서드 호출
        self.image_size = image_size # 이미지 사이즈 저장
        self.mlp1 = nn.Linear(image_size * image_size, hidden_size) # 첫 번째 MLP 레이어 선언(입력층 -> 은닉층1)
        self.mlp2 = nn.Linear(hidden_size, hidden_size) # 두 번째 MLP 레이어 선언(은닉층1 -> 은닉층2)
        self.mlp3 = nn.Linear(hidden_size, hidden_size) # 세 번째 MLP 레이어 선언(은닉층2 -> 은닉층3)
        self.mlp4 = nn.Linear(hidden_size, num_classes) # 네 번째 MLP 레이어 선언(은닉층3 -> 출력층)
        self.test = nn.Sequential(OrderedDict(          # Sequential 클래스를 사용하여 레이어를 순차적으로 쌓음
            [
                ('test1', nn.Conv2d(1, 1, 1, 1)),       # 레이어 이름: test1, 레이어: 1x1 컨볼루션 레이어
                ('test2', nn.Conv2d(1, 1, 1, 1)),       # 레이어 이름: test2, 레이어: 1x1 컨볼루션 레이어
                ('test3', nn.Conv2d(1, 1, 1, 1)),       # 레이어 이름: test3, 레이어: 1x1 컨볼루션 레이어
                ('test4', nn.Conv2d(1, 1, 1, 1))        # 레이어 이름: test4, 레이어: 1x1 컨볼루션 레이어
            ]
            )
        )
        self.fc_s1 = [nn.Linear(100, 100) for _ in range(10)]                   # 100x100 선형 레이어를 10개 만들어 리스트에 저장
        self.fc_s2 = nn.ModuleList([nn.Linear(100, 100) for _ in range(10)])    # 100x100 선형 레이어를 10개 만들어 모듈 리스트에 저장
        
    def forward(self, x): # 순전파: 데이터가 레이어 통과하는 방식 지정
        batch_size = x.shape[0] # 입력 텐서의 배치 크기 저장(x: [batch_size, 28, 28, 1])
        x = torch.reshape(x, (-1, self.image_size * self.image_size)) # 28*28 픽셀 이미지를 1차원 벡터로 변환(펼치기)
        # 순전파 수행: 입력 이미지를 순차적으로 MLP 레이어에 통과시킴
        x = self.mlp1(x) # [batch_size, 500]
        x = self.mlp2(x) # [batch_size, 500]
        x = self.mlp3(x) # [batch_size, 500]
        x = self.mlp4(x) # [batch_size, 10]
        
        # for idx, fc in enumerate(self.fc_s2):
        #     if idx == 0: continue
        #     if idx % 2 == 0:
        #         x = reshape...
                
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

# 평가함수 구현
# 전체 데이터에 대한 정확도를 계산하는 함수
def evaluate(model, loader, device):    # 모델, 데이터 로더, 디바이스를 인자로 받음
    with torch.no_grad():               # 그래디언트 계산 비활성화
        model.eval()                    # 모델을 평가 모드로 설정
        total = 0                       # 전체 데이터 개수 저장 변수
        correct = 0                     # 정답 개수 저장 변수
        for images, targets in loader:  # 데이터 로더로부터 미니배치를 하나씩 꺼내옴
            images, targets = images.to(device), targets.to(device) # 디바이스에 데이터를 보냄
            output = model(images)      # 모델에 미니배치 데이터 입력하여 결괏값 계산
            output_index = torch.argmax(output, dim = 1) # 결괏값 중 가장 큰 값의 인덱스를 뽑아냄
            total += targets.shape[0]   # 전체 데이터 개수 누적
            correct += (output_index == targets).sum().item() # 정답 개수 누적
        
    acc = correct / total * 100 # 정확도(%) 계산
    model.train()               # 모델을 학습 모드로 설정
    return acc                  # 정확도(%) 반환
# 클래스별 정확도를 계산하는 함수
def evaluate_by_class(model, loader, device, num_classes):  # 모델, 데이터 로더, 디바이스, 클래스 개수를 인자로 받음
    with torch.no_grad():                                   # 그래디언트 계산 비활성화
        model.eval()                                        # 모델을 평가 모드로 설정
        total = torch.zeros(num_classes)                    # 클래스별 전체 데이터 개수 저장 변수
        correct = torch.zeros(num_classes)                  # 클래스별 정답 개수 저장 변수
        for images, targets in loader:                      # 데이터 로더로부터 미니배치를 하나씩 꺼내옴
            images, targets = images.to(device), targets.to(device) # 디바이스에 데이터를 보냄
            output = model(images)                          # 모델에 미니배치 데이터 입력하여 결괏값 계산
            output_index = torch.argmax(output, dim = 1)    # 결괏값 중 가장 큰 값의 인덱스를 뽑아냄
            for _class in range(num_classes):               # 클래스 개수만큼 반복
                total[_class] += (targets == _class).sum().item() # 클래스별 전체 데이터 개수 누적
                correct[_class] += ((targets == _class) * (output_index == _class)).sum().item() # 클래스별 정답 개수 누적
        
    acc = correct / total * 100 # 클래스별 정확도(%) 계산
    model.train()               # 모델을 학습 모드로 설정
    return acc                  # 클래스별 정확도(%) 반환

_max = -1 # 최대 정확도 저장 변수
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
            
            acc = evaluate(myMLP, test_loader, device)                          # 전체 데이터에 대한 정확도 계산
            # acc = evaluate_by_class(myMLP, test_loader, device, num_classes)  # 클래스별 정확도 계산
            
            # 정확도가 높아지면 모델 저장
            if _max < acc : # acc가 높아지면
                print('새로운 acc 등장, 모델 weight 업데이트', acc) # acc 출력
                _max = acc  # _max에 acc 저장
                # 모델 저장
                torch.save(
                    myMLP.state_dict(),
                    os.path.join(target_folder, 'myMLP_best.ckpt')
                )