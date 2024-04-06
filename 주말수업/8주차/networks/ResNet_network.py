import torch                        # 파이토치 패키지 임포트
import torch.nn as nn               # nn 패키지 임포트

# ResNet의 여러 버전에 따른 레이어 수 정의
_NUMS_18 = [2, 2, 2, 2]
_NUMS_34 = [3, 4, 6, 3]
_NUMS_50 = [3, 4, 6, 3]
_NUMS_101 = [3, 4, 23, 3]
_NUMS_152 = [3, 8, 36, 3]

# ResNet의 채널 수 정의
_CHANNELS_33 = [64, 128, 256, 512]
_CHANNELS_131 = [256, 512, 1024, 2048] 

class InputPart(nn.Module):
    # ResNet의 입력 부분을 정의하는 클래스
    def __init__(self, in_channel=3, out_channel=64, image_size=224):
        super().__init__()
        # 초기 convolutional 레이어 정의
        self.conv = nn. Sequential(
            nn.Conv2d(in_channel, out_channel, 7, 2, 3),        # 7x7 convolutional, stride=2, padding=3
            nn.BatchNorm2d(out_channel),                        # batch normalization
            nn.ReLU(),                                          # ReLU            
        )
        # Max Pooling 레이어 정의
        self.pool = nn.MaxPool2d(3, 2, 1)                       # 3x3 max pooling, stride=2, padding=1
        poolsize = 56 if image_size == 224 else 8               # 입력 이미지 크기에 따라 max pooling 크기 조정
        self.pool2 = nn.AdaptiveMaxPool2d((poolsize, poolsize))
        
    # 입력 이미지를 convolutional 및 pooling 레이어를 통과시키는 함수
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x
    
class OutputPart(nn.Module):
    # ResNet의 출력 부분을 정의하는 클래스
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config                                    # ResNet 버전(18, 34, 50, 101, 152)
        self.in_channel = 512 if config in [18, 34] else 2048   # ResNet 버전에 따른 입력 채널 수
        
        # Average pooling 및 fully connected 레이어 정의
        self.pool = nn.AdaptiveAvgPool2d((1, 1))                # 1x1 average pooling
        self.fc = nn.Linear(self.in_channel, num_classes)       # fully connected 레이어
    
    # 입력 텐서를 pooling 및 fully connected 레이어를 통과시키는 함수
    def forward(self, x):
        # x: (batch_size, out_channel= 512 / 2048, h= 7, w= 7) -> 18, 34 / 50, 101, 152 layer
        batch_size, c, h, w = x.shape           # 입력 텐서의 크기 저장
        x = self.pool(x)                        # average pooling 레이어 통과
        x = torch.reshape(x, (batch_size, c))   # fully connected 레이어에 입력할 수 있도록 텐서 크기 조정
        x = self.fc(x)                          # fully connected 레이어 통과
        return x

class conv(nn.Module):
    # 기본 convolutional 레이어를 정의하는 클래스
    def __init__(self, filter_size, in_channel, out_channel, stride=1, use_relu=True):
        super().__init__()
        padding = 1 if filter_size == 3 else 0      # filter_size가 3x3이면 padding=1, 1x1이면 padding=0
        self.conv = nn.Conv2d(in_channel,
                              out_channel,
                              filter_size,
                              stride, padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.use_relu = use_relu                    # ReLU 사용 여부
        if use_relu:                                # ReLU 사용 여부에 따라 ReLU 레이어 정의
            self.rl = nn.ReLU()
        
    # 입력 텐서를 convolutional 및 batch normalization 레이어를 통과시키는 함수
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_relu:
            x = self.rl(x)
        return x

class Block(nn.Module):
    # ResNet의 기본 블록을 정의하는 클래스
    def __init__(self, in_channel, out_channel, down_sample=False):
        super().__init__()
        self.down_sample = down_sample          # down sampling 여부
        stride = 1                              # convolutional 레이어의 stride
        if self.down_sample:                    # down sampling이면
            stride = 2                          # convolutional 레이어의 stride를 2로 설정
            # down sampling을 위한 convolutional 레이어 정의
            self.down_sample_net = conv(filter_size=3, in_channel=in_channel, out_channel=out_channel, stride=stride)
        
        # 두 개의 convolutional 레이어 정의
        self.conv1 = conv(filter_size=3, in_channel=in_channel, out_channel=out_channel, stride=stride)
        self.conv2 = conv(filter_size=3, in_channel=out_channel, out_channel=out_channel, use_relu=False)
        self.relu = nn.ReLU()
        
    # 입력 텐서를 두 개의 convolutional 레이어 및 skip connection을 통과시키는 함수
    def forward(self, x):
        x_skip = x.clone()                          # skip connection을 위해 입력 텐서 복사
        
        x = self.conv1(x)                           # 첫 번째 convolutional 레이어 통과
        x = self.conv2(x)                           # 두 번째 convolutional 레이어 통과
        
        if self.down_sample:                        # down sampling이면
            x_skip = self.down_sample_net(x_skip)   # 입력 텐서를 down sampling 레이어 통과
        
        x = x + x_skip                              # skip connection
        
        x = self.relu(x)                            # ReLU 통과
        return x

class BottleNeck(nn.Module):
    # ResNet의 BottleNeck 블록을 정의하는 클래스
    def __init__(self, in_channel, out_channel, down_sample=False):
        super().__init__()
        
        middle_channel = out_channel // 4           # BottleNeck 블록의 중간 채널 수
        stride = 2 if down_sample else 1            # convolutional 레이어의 stride
        
        # down sampling을 위한 convolutional 레이어 정의
        self.down_sample_net = conv(filter_size=3, in_channel=in_channel, out_channel=out_channel, stride=stride)
        
        self.conv1 = conv(filter_size=1, in_channel=in_channel, out_channel=middle_channel, stride=stride)
        self.conv2 = conv(filter_size=3, in_channel=middle_channel, out_channel=middle_channel)
        self.conv3 = conv(filter_size=1, in_channel=middle_channel, out_channel=out_channel, use_relu=False)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x_skip = x.clone()                          # skip connection을 위해 입력 텐서 복사
        
        x = self.conv1(x)                           # 첫 번째 convolutional 레이어 통과
        x = self.conv2(x)                           # 두 번째 convolutional 레이어 통과
        x = self.conv3(x)                           # 세 번째 convolutional 레이어 통과
        
        x_skip = self.down_sample_net(x_skip)       # 입력 텐서를 down sampling 레이어 통과
        
        x = x + x_skip                              # skip connection
        
        x = self.relu(x)                            # ReLU 통과
        return x

class MiddlePart(nn.Module):
    # ResNet의 중간 부분을 정의하는 클래스
    def __init__(self, config):
        super().__init__()
        if config == 18:
            _nums = _NUMS_18
            _channels = _CHANNELS_33
            self.TARGET = Block
        elif config == 34:
            _nums = _NUMS_34
            _channels = _CHANNELS_33
            self.TARGET = Block
        elif config == 50:
            _nums = _NUMS_50
            _channels = _CHANNELS_131
            self.TARGET = BottleNeck
        elif config == 101:
            _nums = _NUMS_101
            _channels = _CHANNELS_131
            self.TARGET = BottleNeck
        elif config == 152:
            _nums = _NUMS_152
            _channels = _CHANNELS_131
            self.TARGET = BottleNeck
        
        self.layer1 = self.make_layer(_nums[0], 64, _channels[0])
        self.layer2 = self.make_layer(_nums[1], _channels[0], _channels[1], down_sample=True)
        self.layer3 = self.make_layer(_nums[2], _channels[1], _channels[2], down_sample=True)
        self.layer4 = self.make_layer(_nums[3], _channels[2], _channels[3], down_sample=True)
    
    def make_layer(self, _num, in_channel, out_channel, down_sample=False):
        layer = [                                               # 레이어 정의
            self.TARGET(in_channel, out_channel, down_sample)   # ResNet의 기본 블록 또는 BottleNeck 블록
        ]
        for idx in range(_num-1):                               # 레이어 반복
            layer.append(                                       # 레이어 추가
                self.TARGET(out_channel, out_channel)           # ResNet의 기본 블록 또는 BottleNeck 블록
            )
        layer = nn.Sequential(*layer)                           # 레이어를 Sequential로 묶기
        return layer
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class ResNet(nn.Module):
    # 전체 ResNet 아키텍처를 정의하는 클래스
    def __init__(self, num_classes, config=18):
        super().__init__()
        # ResNet의 입력, 중간, 출력 부분 정의
        self.input_part = InputPart()
        self.output_part = OutputPart(config, num_classes)
        self.middel_part = MiddlePart(config)
    
    # 입력 이미지를 ResNet 아키텍처를 통과시키는 함수
    def forward(self, x):
        x = self.input_part(x)
        x = self.middel_part(x)
        x = self.output_part(x)
        return x