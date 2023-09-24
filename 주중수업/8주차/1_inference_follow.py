# 패키지 임포트
import os                                       # 경로 설정을 위한 os 패키지 임포트
import torch                                    # 파이토치 패키지 임포트
import torch.nn as nn                           # nn 패키지 임포트
from PIL import Image                           # 이미지를 다루기 위한 PIL 패키지 임포트
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from utils.parser import infer_parse_args       # 하이퍼파라미터를 받기 위한 함수 임포트
from utils.load_hparam import load_hparams      # 하이퍼파라미터를 불러오기 위한 함수 임포트
from networks.MLP_network import MLP            # MLP 클래스 임포트
from networks.LeNet_network import LeNet        # LeNet 클래스 임포트
from networks.ResNet_network import ResNet      # ResNet 클래스 임포트
from utils.get_loader import get_transform      # 데이터를 불러오기 위한 함수 임포트

# 메인 함수 정의
def main():
    # 하이퍼파라미터 받기
    args = infer_parse_args()
    
    # 타겟 폴더와 타겟 이미지가 존재하는지 확인
    assert os.path.exists(args.trained_folder), 'target folder does not exists'
    assert os.path.exists(args.target_image), 'target image does not exists'
    
    # 하이퍼파라미터 불러오기
    args = load_hparams(args)
          
    # 모델 객체 만들기
    # model = MLP(args.image_size, args.hidden_size, args.num_classes).to(args.device)
    # model = LeNet(args.image_size, args.num_classes).to(args.device)
    model = ResNet(args.num_classes, args.resnet_config).to(args.device)
    
    # 저장된 모델 가중치 불러오기
    ckpt = torch.load(                                  # 모델 가중치 불러오기
        os.path.join(                                   # 경로 설정
            args.trained_folder, 'model_best.ckpt'      # 가중치가 저장된 경로
            ),
        map_location=torch.device('cpu')                                  
        )                                       
    model.load_state_dict(ckpt)                         # 모델에 가중치 저장

    # 추론할 이미지 불러오기
    input_image = Image.open(args.target_image) #.convert('L')

    # 이미지를 모델에 입력할 수 있는 형태로 변환
    trans = get_transform(args)                 # 이미지를 변환하기 위한 함수 불러오기
    image = trans(input_image).to(args.device)  # 이미지를 디바이스에 올리기
    image = image.unsqueeze(0)                  # 이미지 차원 늘리기

    # 모델에 이미지 입력 후 출력값 저장
    model.eval()
    output = model(image)
    
    # 출력값 중 가장 큰 값의 인덱스를 추론 결과로 저장
    output = torch.argmax(output).item()

    # 추론 결과 출력
    print(f'Model says, the image is {output}')

# 이 파일이 메인 파일이면 main 함수 실행
if __name__ == '__main__':
    main()