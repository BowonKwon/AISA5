import torch            # 파이토치 라이브러리 임포트
import torch.nn as nn   # 파이토치 뉴럴넷 라이브러리 임포트

# 모델 설계도 그리기

# LeNet-5 모델 설계
class LeNet(nn.Module):
    # 클래스 초기화 함수 정의
    def __init__(self, image_size, num_classes):
        # 상속받은 클래스의 초기화 메서드 호출
        super().__init__()
        
        # 모델의 입력 이미지 크기, 클래스 개수 저장
        self.image_size = image_size            # 이미지 크기
        self.conv1 = nn.Sequential(             # 첫번째 합성곱 레이어
            nn.Conv2d(3, 6, 5, 1, 0),           # 3채널 입력, 6채널 출력, 5x5 커널, 스트라이드 1, 패딩 0
            nn.BatchNorm2d(6),                  # 배치 정규화
            nn.ReLU()                           # ReLU 활성화 함수
            )   
        self.pool1 = nn.MaxPool2d(2, 2)         # 2x2 커널, 스트라이드 2
        self.conv2 = nn.Sequential(             # 두번째 합성곱 레이어
            nn.Conv2d(6, 16, 5, 1, 0),          # 6채널 입력, 16채널 출력, 5x5 커널, 스트라이드 1, 패딩 0
            nn.BatchNorm2d(16),                 # 배치 정규화
            nn.ReLU()                           # ReLU 활성화 함수
            )
        self.pool2 = nn.MaxPool2d(2, 2)         # 2x2 커널, 스트라이드 2
        
        self.fc1 = nn.Linear(400, 120)          # 400개 노드 입력, 120개 노드 출력
        self.fc2 = nn.Linear(120, 84)           # 120개 노드 입력, 84개 노드 출력
        self.fc3 = nn.Linear(84, num_classes)   # 84개 노드 입력, 클래스 개수 노드 출력
        
    # 순전파 함수 정의
    def forward(self, x):
        # 입력 이미지의 배치 크기 저장(x: [batch_size, 3, 32, 32])
        batch_size = x.shape[0]
        
        # 입력 이미지를 모델의 입력 이미지 크기로 리사이즈
        x = self.conv1(x)   # 첫번째 합성곱 레이어
        x = self.pool1(x)   # 첫번째 풀링 레이어
        x = self.conv2(x)   # 두번째 합성곱 레이어
        x = self.pool2(x)   # 두번째 풀링 레이어
        
        # 배치 크기 x 채널 x 높이 x 너비 크기의 1차원 벡터로 변환
        x = torch.reshape(x, (-1, 400))
          
        # 완전 연결 레이어 통과
        x = self.fc1(x)     # 첫번째 완전 연결 레이어
        x = self.fc2(x)     # 두번째 완전 연결 레이어
        x = self.fc3(x)     # 세번째 완전 연결 레이어
        
        # 출력값 반환
        return x