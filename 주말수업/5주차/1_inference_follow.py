# 패키지 임포트
import os                                       # 경로 설정을 위한 os 패키지 임포트
import torch                                    # 파이토치 패키지 임포트
import torch.nn as nn                           # 파이토치의 nn 패키지 임포트
from PIL import Image                           # 이미지를 다루기 위한 PIL 패키지 임포트
from torchvision.transforms import Resize       # 이미지 크기를 조절하는 함수 임포트
from torchvision.transforms import ToTensor     # 이미지를 텐서로 변환하는 함수 임포트

# 타겟하는 학습 세팅을 설정
target_folder = '../../주중수업/5주차/results/1' # 타겟 폴더 설정
assert os.path.exists(target_folder), 'target folder does not exists' # 타겟 폴더가 존재하는지 확인
 
# 하이퍼파라미터 로드
with open(os.path.join(target_folder, 'hparam.txt'), 'r') as f: # hparam.txt 파일을 읽기 모드로 열기
    data = f.readlines()                                        # 파일의 모든 줄을 읽어서 리스트로 저장
print(data)                                                     # 읽어온 데이터 출력

lr = float(data[0].strip())         # 학습률 저장
image_size = int(data[1].strip())   # 이미지 사이즈 저장
num_classes = int(data[2].strip())  # 클래스 개수 저장
batch_size = int(data[3].strip())   # 배치 크기 저장
hidden_size = int(data[4].strip())  # 은닉층 크기 저장
epochs = int(data[5].strip())       # 에포크 수 저장
results_folder = data[6].strip()    # 결과 폴더 저장

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU 사용 여부에 따라 device 설정

# 모델 class 만들기
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
        x = torch.reshape(x, (-1, image_size * image_size)) # 28*28 픽셀 이미지를 1차원 벡터로 변환(펼치기)
        # 순전파 수행: 입력 이미지를 순차적으로 MLP 레이어에 통과시킴
        x = self.mlp1(x) # [batch_size, 500]
        x = self.mlp2(x) # [batch_size, 500]
        x = self.mlp3(x) # [batch_size, 500]
        x = self.mlp4(x) # [batch_size, 10]
        # 최종 출력 반환
        return x
    
# 모델 선언
myMLP = MLP(image_size, hidden_size, num_classes).to(device)

# 저장된 모델 가중치 불러오기
ckpt = torch.load(                          # 저장된 모델 가중치 불러오기
    os.path.join(                           # 경로 설정
        target_folder, 'myMLP_best.ckpt'    # 타겟 폴더 내의 myMLP_best.ckpt 파일 경로
        )                                   
    )                                       
myMLP.load_state_dict(ckpt)                 # 모델에 가중치 저장

# 추론 데이터를 가지고 오기
image_path = './test_image.jpg'             # 추론할 이미지 경로
assert os.path.exists(image_path), 'target image does not exists' # 이미지가 존재하는지 확인
input_image = Image.open().convert('L')     # 이미지를 흑백으로 변환

# 학습 과정에서 사용했던 전처리 과정을 그대로 실행 
resizer = Resize(image_size)                # 크기 맞추기: 이미지 크기를 조절하는 함수 선언
totensor = ToTensor()                       # 크기 맞추기: 이미지를 텐서로 변환하는 함수 선언
image = totensor(resizer(input_image)).to(device) # 이미지를 텐서로 변환 후 device로 이동

# 모델 추론 진행
output = myMLP(image)                       # 모델에 이미지 입력 후 출력값 저장
# 추론 결과를 우리가 이해할 수 있는 형태로 변환 
output = torch.argmax(output).item()        # 출력값 중 가장 큰 값의 인덱스를 추론 결과로 저장

print(f'Model says, the image is {output}') # 모델이 추론한 결과 출력