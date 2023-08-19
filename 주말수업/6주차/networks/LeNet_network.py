import torch            # 파이토치 라이브러리 임포트
import torch.nn as nn   # 파이토치 뉴럴넷 라이브러리 임포트

# 모델 설계도 그리기

# LeNet-5 모델 설계: 기본
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
    
# LeNet-5 모델 설계: Linear 주입
class LeNet_Linear(nn.Module):
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

        ### linear를 주입하기 위한 코드 작업 ###

        self.inj_linear1 = nn.Sequential(nn.Linear(1176, 2048), nn.ReLU())  # 1176개 노드 입력, 2048개 노드 출력
        self.inj_linear2 = nn.Sequential(nn.Linear(2048, 1176), nn.ReLU())  # 2048개 노드 입력, 1176개 노드 출력
        
        ######################################
        
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

        ### linear를 주입하기 위한 코드 작업 ###
        
        _, c, w, h = x.shape                    # x: [batch_size, c, w, h]
        x = torch.reshape(x, (-1, c * w * h))   # x: [batch_size, c * w * h]
        x = self.inj_linear1(x)                 # 첫번째 주입 레이어
        x = self.inj_linear2(x)                 # 두번째 주입 레이어
        x = torch.reshape(x, (-1, c, w, h))     # x: [batch_size, c, w, h]

        ######################################

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
    
# LeNet-5 모델 설계: MultiConv
class LeNet_MultiConv(nn.Module):
    # 클래스 초기화 함수 정의
    def __init__(self, image_size, num_classes):
        # 상속받은 클래스의 초기화 메서드 호출
        super().__init__()
        
        # 모델의 입력 이미지 크기, 클래스 개수 저장
        self.image_size = image_size            # 이미지 크기
        self.conv_block1_1 = nn.Sequential(     # 1-1번째 합성곱 블록
            nn.Conv2d(3, 6, 5, 1, 2),           # 3채널 입력, 6채널 출력, 5x5 커널, 스트라이드 1, 패딩 2
            nn.BatchNorm2d(6),                  # 배치 정규화
            nn.ReLU()                           # ReLU 활성화 함수
            )
        self.conv_block1_2 = nn.Sequential(     # 1-2번째 합성곱 블록
            nn.Conv2d(6, 6, 5, 1, 2),           # 6채널 입력, 6채널 출력, 5x5 커널, 스트라이드 1, 패딩 2
            nn.BatchNorm2d(6),                  # 배치 정규화
            nn.ReLU()                           # ReLU 활성화 함수
            )   
        self.conv_block1_3 = nn.Sequential(     # 1-3번째 합성곱 블록
            nn.Conv2d(6, 6, 5, 1, 2),           # 6채널 입력, 6채널 출력, 5x5 커널, 스트라이드 1, 패딩 2
            nn.BatchNorm2d(6),                  # 배치 정규화
            nn.ReLU()                           # ReLU 활성화 함수
            )   
        self.conv_block1_4 = nn.Sequential(     # 1-4번째 합성곱 블록
            nn.Conv2d(6, 6, 5, 1, 0),           # 6채널 입력, 6채널 출력, 5x5 커널, 스트라이드 1, 패딩 0
            nn.BatchNorm2d(6),                  # 배치 정규화
            nn.ReLU()                           # ReLU 활성화 함수
            )   
        self.conv_blocks1 = nn.Sequential(      # 1번째 합성곱 블록
            self.conv_block1_1,                 # 1-1
            self.conv_block1_2,                 # 1-2
            self.conv_block1_3,                 # 1-3
            self.conv_block1_4,                 # 1-4
        )
        self.pool1 = nn.MaxPool2d(2, 2)         # 2x2 커널, 스트라이드 2

        self.conv_block2_1 = nn.Sequential(     # 2-1번째 합성곱 블록
            nn.Conv2d(6, 16, 5, 1, 2),          # 6채널 입력, 16채널 출력, 5x5 커널, 스트라이드 1, 패딩 2
            nn.BatchNorm2d(16),                 # 배치 정규화
            nn.ReLU()                           # ReLU 활성화 함수
            )
        self.conv_block2_2 = nn.Sequential(     # 2-2번째 합성곱 블록
            nn.Conv2d(16, 16, 5, 1, 2),         # 16채널 입력, 16채널 출력, 5x5 커널, 스트라이드 1, 패딩 2
            nn.BatchNorm2d(16),                 # 배치 정규화
            nn.ReLU()                           # ReLU 활성화 함수
            )
        self.conv_block2_3 = nn.Sequential(     # 2-3번째 합성곱 블록
            nn.Conv2d(16, 16, 5, 1, 0),         # 16채널 입력, 16채널 출력, 5x5 커널, 스트라이드 1, 패딩 0
            nn.BatchNorm2d(16),                 # 배치 정규화
            nn.ReLU()                           # ReLU 활성화 함수
            )
        self.conv_blocks2 = nn.Sequential(      # 2번째 합성곱 블록
            self.conv_block2_1,                 # 2-1
            self.conv_block2_2,                 # 2-2
            self.conv_block2_3                  # 2-3
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
        x = self.conv_blocks1(x)    # 첫번째 합성곱 블록
        x = self.pool1(x)           # 첫번째 풀링 레이어
        x = self.conv_blocks2(x)    # 두번째 합성곱 블록
        x = self.pool2(x)           # 두번째 풀링 레이어
        
        # 배치 크기 x 채널 x 높이 x 너비 크기의 1차원 벡터로 변환
        x = torch.reshape(x, (-1, 400))
          
        # 완전 연결 레이어 통과
        x = self.fc1(x)             # 첫번째 완전 연결 레이어
        x = self.fc2(x)             # 두번째 완전 연결 레이어
        x = self.fc3(x)             # 세번째 완전 연결 레이어
        
        # 출력값 반환
        return x
    
# LeNet-5 모델 설계: MergeConv
class LeNet_MergeConv(nn.Module):
    # 클래스 초기화 함수 정의
    def __init__(self, image_size, num_classes):
        # 상속받은 클래스의 초기화 메서드 호출
        super().__init__()
        
        # 모델의 입력 이미지 크기, 클래스 개수 저장
        self.image_size = image_size            # 이미지 크기
        self.conv1_1 = nn.Sequential(           # 1-1번째 합성곱 레이어
            nn.Conv2d(3, 6, 1, 1, 0),           # 3채널 입력, 6채널 출력, 1x1 커널, 스트라이드 1, 패딩 0
            nn.BatchNorm2d(6),                  # 배치 정규화     
            nn.ReLU()                           # ReLU 활성화 함수
            )   
        self.conv1_2 = nn.Sequential(           # 1-2번째 합성곱 레이어
            nn.Conv2d(3, 6, 3, 1, 1),           # 3채널 입력, 6채널 출력, 3x3 커널, 스트라이드 1, 패딩 1 
            nn.BatchNorm2d(6),                  # 배치 정규화
            nn.ReLU()                           # ReLU 활성화 함수
            )   
        self.conv1_3 = nn.Sequential(           # 1-3번째 합성곱 레이어
            nn.Conv2d(3, 6, 5, 1, 2),           # 3채널 입력, 6채널 출력, 5x5 커널, 스트라이드 1, 패딩 2
            nn.BatchNorm2d(6),                  # 배치 정규화
            nn.ReLU()                           # ReLU 활성화 함수
            )   
        self.conv1 = nn.Sequential(             # 첫번째 합성곱 레이어
            nn.Conv2d(18, 6, 5, 1, 0),          # 18채널 입력, 6채널 출력, 5x5 커널, 스트라이드 1, 패딩 0
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
        x1 = self.conv1_1(x)    # 1-1번째 합성곱 레이어
        x2 = self.conv1_2(x)    # 1-2번째 합성곱 레이어
        x3 = self.conv1_3(x)    # 1-3번째 합성곱 레이어
        
        mer_x = torch.cat((x1, x2, x3), dim=1)  # 1-1, 1-2, 1-3번째 합성곱 레이어의 출력값을 채널 방향으로 합침

        x = self.conv1(mer_x)   # 첫번째 합성곱 레이어
        x = self.pool1(x)       # 첫번째 풀링 레이어
        x = self.conv2(x)       # 두번째 합성곱 레이어
        x = self.pool2(x)       # 두번째 풀링 레이어
        
        # 배치 크기 x 채널 x 높이 x 너비 크기의 1차원 벡터로 변환
        x = torch.reshape(x, (-1, 400))
          
        # 완전 연결 레이어 통과
        x = self.fc1(x)         # 첫번째 완전 연결 레이어
        x = self.fc2(x)         # 두번째 완전 연결 레이어
        x = self.fc3(x)         # 세번째 완전 연결 레이어
        
        # 출력값 반환
        return x