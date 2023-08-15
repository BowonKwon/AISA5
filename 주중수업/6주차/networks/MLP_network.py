import torch            # 파이토치 라이브러리 임포트
import torch.nn as nn   # 파이토치 뉴럴넷 라이브러리 임포트

# 모델 설계도 그리기
# nn.Module을 상속받는 MLP 클래스 선언
class MLP(nn.Module):
    # 클래스 초기화 함수 정의
    def __init__(self, image_size, hidden_size, num_classes):
        # 상속받은 클래스의 초기화 메서드 호출
        super().__init__()
        # 하이퍼파라미터 저장
        self.image_size = image_size                                    # 이미지 크기
        self.mlp1 = nn.Linear(image_size * image_size, hidden_size)     # 첫 번째 MLP 레이어 선언(입력층 -> 은닉층1)
        self.mlp2 = nn.Linear(hidden_size, hidden_size)                 # 두 번째 MLP 레이어 선언(은닉층1 -> 은닉층2)
        self.mlp3 = nn.Linear(hidden_size, hidden_size)                 # 세 번째 MLP 레이어 선언(은닉층2 -> 은닉층3)
        self.mlp4 = nn.Linear(hidden_size, num_classes)                 # 네 번째 MLP 레이어 선언(은닉층3 -> 출력층)
    # 순전파 함수 정의
    def forward(self, x):
        # 입력 이미지의 배치 크기 저장
        batch_size = x.shape[0]
        # 입력 이미지를 1차원 벡터로 변환
        x = torch.reshape(x, (-1, self.image_size * self.image_size))
        # MLP 레이어를 통과한 후, ReLU 함수를 적용
        x = self.mlp1(x) # [batch_size, 500]
        x = self.mlp2(x) # [batch_size, 500]
        x = self.mlp3(x) # [batch_size, 500]
        x = self.mlp4(x) # [batch_size, 10]
        # 출력값 반환
        return x