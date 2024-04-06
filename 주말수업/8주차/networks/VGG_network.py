import torch                # 파이토치 라이브러리 임포트
import torch.nn as nn       # 파이토치 뉴럴넷 라이브러리 임포트

# VGG 네트워크의 기본 convolutional 레이어를 정의하는 클래스
class VGG_conv(nn.Module):
    def __init__(self, in_channel, out_channel, one_filter=False):
        super().__init__()
        # 필터 크기와 패딩을 결정하는 조건
        kernel_size = 1 if one_filter else 3
        padding = 0 if one_filter else 1
        
        # convolutional 레이어 정의
        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=kernel_size,  # 1/3
                              stride=1,
                              padding=padding)          # 0/1
        # Batch normalization 레이어 정의
        self.bn = nn.BatchNorm2d(num_features=out_channel)
        # ReLU 활성화 함수 정의
        self.relu = nn.ReLU()
    # 순전파 함수 정의    
    def forward(self, x):
        x = self.conv(x)    # Convolutional 레이어 통과
        x = self.bn(x)      # Batch normalization 레이어 통과
        x = self.relu(x)    # ReLU 활성화 함수 통과
        return x

# VGG 네트워크의 블록을 정의하는 클래스
class VGG_Block(nn.Module):
    def __init__(self, num_convs, in_channel, out_channel, one_filter=False):
        super().__init__()
    
        # 첫 번째 convolutional 레이어를 리스트에 추가
        self.convs_list = [
            VGG_conv(
                in_channel=in_channel,
                out_channel=out_channel)
        ]
        # 나머지 convolutional 레이어들을 리스트에 추가
        for idx in range(num_convs-1):
            self.convs_list.append(
                VGG_conv(
                    in_channel=out_channel,
                    out_channel=out_channel)
            )
        # one_filter 조건이 참일 경우 마지막 레이어를 제거하고 새로운 레이어 추가
        if one_filter:
            self.convs_list.pop()
            VGG_conv(
                in_channel=out_channel,
                out_channel=out_channel,
                one_filter=True)
        # PyTorch의 ModuleList로 레이어들을 관리
        self.convs_module = nn.ModuleList(self.convs_list)
        # Max pooling 레이어 정의
        self.max_pool = nn.MaxPool2d(2, 2)
    
    # 순전파 함수 정의
    def forward(self, x):
        # 모든 convolutional 레이어를 순차적으로 거친 후 max pooling 수행
        for module in self.convs_module:
            x = module(x)
        x = self.max_pool(x)
        return x

# VGG 네트워크의 분류기를 정의하는 클래스
class VGG_Classifier(nn.Module):
    def __init__(self, num_classes, image_size):
        super().__init__()
        in_feature = 512 if image_size == 32 else 25088
        # Fully connected 레이어와 ReLU 활성화 함수 정의
        self.fc1 = nn.Linear(in_feature, 4096)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(4096, num_classes)
    
    # 순전파 함수 정의
    def forward(self, x):
        # Fully connected 레이어와 ReLU 활성화 함수를 순차적으로 적용
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# VGG 네트워크의 여러 버전을 정의하는 클래스들 (VGG_A, VGG_B, VGG_C, VGG_D, VGG_E)
class VGG_A(nn.Module):
    def __init__(self): 
        super().__init__()
        self.VGG_Block1 = VGG_Block(1, 3, 64) # num_convs, in_channel, out_channel, one_filter=False
        self.VGG_Block2 = VGG_Block(1, 64, 128)
        self.VGG_Block3 = VGG_Block(2, 128, 256)
        self.VGG_Block4 = VGG_Block(2, 256, 512)
        self.VGG_Block5 = VGG_Block(2, 512, 512)
    
    def forward(self, x):
        x = self.VGG_Block1(x)
        x = self.VGG_Block2(x)
        x = self.VGG_Block3(x)
        x = self.VGG_Block4(x)
        x = self.VGG_Block5(x)
        return x
    
class VGG_B(VGG_A):
    def __init__(self): 
        super().__init__()
        self.VGG_Block1 = VGG_Block(2, 3, 64)
        self.VGG_Block2 = VGG_Block(2, 64, 128)
 
class VGG_C(VGG_B):
    def __init__(self): 
        super().__init__()
        self.VGG_Block3 = VGG_Block(3, 128, 256, one_filter=True)
        self.VGG_Block4 = VGG_Block(3, 256, 512, one_filter=True)
        self.VGG_Block5 = VGG_Block(3, 512, 512, one_filter=True)

class VGG_D(VGG_B):
    def __init__(self): 
        super().__init__()
        self.VGG_Block3 = VGG_Block(3, 128, 256)
        self.VGG_Block4 = VGG_Block(3, 256, 512)
        self.VGG_Block5 = VGG_Block(3, 512, 512)

class VGG_E(VGG_D):
    def __init__(self): 
        super().__init__()
        self.VGG_Block3 = VGG_Block(4, 128, 256)
        self.VGG_Block4 = VGG_Block(4, 256, 512)
        self.VGG_Block5 = VGG_Block(4, 512, 512)

# VGG 네트워크 전체를 정의하는 클래스    
class VGG(nn.Module):
    def __init__(self, num_classes, image_size, config='a'): 
        super().__init__()
        
        # config 인자에 따라 VGG 네트워크의 버전을 선택
        if config == 'a':
            self.net = VGG_A()
        elif config == 'b':
            self.net = VGG_B()
        elif config == 'c':
            self.net = VGG_C()
        elif config == 'd':
            self.net = VGG_D()
        elif config == 'e':
            self.net = VGG_E()
        
        # 분류기 정의
        self.classifier = VGG_Classifier(num_classes, image_size)
    
    # 순전파 함수 정의
    def forward(self, x):
        # 입력 데이터의 형상을 저장
        b, c, w, h = x.shape
        # VGG 네트워크를 거친 후 결과를 평탄화
        x = self.net(x)
        x = torch.reshape(x, (b, -1)) # 평탄화
        # 분류기를 거쳐 최종 결과 반환
        x = self.classifier(x)
        return x