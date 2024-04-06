from torch.utils.data import DataLoader         # 데이터 로더 임포트
from torchvision.transforms import Compose      # 이미지 변환 함수들을 묶어주는 함수 임포트
from torchvision.transforms import Resize       # 이미지 크기 조절 함수 임포트
from torchvision.transforms import ToTensor     # 이미지를 텐서로 변환하는 함수 임포트
from torchvision.transforms import Normalize    # 이미지를 정규화하는 함수 임포트

from torchvision.datasets import MNIST          # MNIST 데이터셋 임포트
from torchvision.datasets import CIFAR10        # CIFAR10 데이터셋 임포트

CIFAR10_MEAN = [0.491, 0.482, 0.447]            # CIFAR10 데이터셋의 평균
CIFAR10_STD = [0.247, 0.244, 0.262]             # CIFAR10 데이터셋의 표준편차
IMAGE_MEAN = [0.485, 0.456, 0.406]              # ImageNet 데이터셋의 평균
IMAGE_STD = [0.229, 0.224, 0.225]               # ImageNet 데이터셋의 표준편차

# 데이터셋에 맞는 전처리 과정을 정의
def get_transform(args):
    # 데이터셋이 MNIST인 경우
    if args.data == 'mnist':
        trans = Compose([                                   # Compose 함수를 이용하여 여러 전처리 과정을 묶어줌
            Resize((args.image_size, args.image_size)),     # 이미지 크기를 32x32로 조절
            ToTensor()                                      # 이미지를 텐서로 변환
        ])
    # 데이터셋이 CIFAR인 경우
    elif args.data == 'cifar':
        trans = Compose([                                   # Compose 함수를 이용하여 여러 전처리 과정을 묶어줌
            Resize((args.image_size, args.image_size)),     # 이미지 크기를 32x32로 조절
            ToTensor(),                                     # 이미지를 텐서로 변환
            Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)   # 이미지를 정규화
        ])
    # 데이터셋이 DOG인 경우
    elif args.data == 'dog':
        trans = Compose([                                   # Compose 함수를 이용하여 여러 전처리 과정을 묶어줌
            Resize((args.image_size, args.image_size)),     # 이미지 크기를 32x32로 조절
            ToTensor(),                                     # 이미지를 텐서로 변환
            Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)       # 이미지를 정규화
        ])
    return trans

# 데이터 로더를 불러오는 함수 정의
def get_loaders(args):
    # dataset 설정(학습, 테스트 데이터)
    if args.data == 'mnist':
        train_dataset = MNIST(root='../../data/mnist', train=True, transform=get_transform(args), download=True)
        test_dataset = MNIST(root='../../data/mnist', train=False, transform=get_transform(args), download=True)
    elif args.data == 'cifar':
        train_dataset = CIFAR10(root='../../data/cifar', train=True, transform=get_transform(args), download=True)
        test_dataset = CIFAR10(root='../../data/cifar', train=False, transform=get_transform(args), download=True)
    elif args.data == 'dog':
        if args.dataset_type == 'imagefolder':
            from torchvision.datasets import ImageFolder
            train_dataset = ImageFolder(root='../../data/dog/train', transform=get_transform(args))
            print('train dataset loaded')
            test_dataset = ImageFolder(root='../../data/dog/val', transform=get_transform(args))
            print('validation dataset loaded')
        elif args.dataset_type == 'custom1':
            pass
        elif args.dataset_type == 'custom2':
            pass

    # dataloader 설정(학습, 테스트 데이터)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader