import argparse                             # 하이퍼파라미터 파싱을 위한 argparse 라이브러리
import torch                                # 파이토치 라이브러리

 # 하이퍼파라미터 파싱 함수
def parse_args():
    # 하이퍼파라미터를 받기 위한 parser 객체 생성
    parser = argparse.ArgumentParser()
    # 하이퍼파라미터를 받기 위한 인자 추가
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=1000)
    parser.add_argument('--total_epochs', type=int, default=3)
    parser.add_argument('--results_folder', type=str, default='results')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--do_save', action='store_true', help='if given, save results')
    parser.add_argument('--data', type=str, default='mnist', choices=['mnist', 'cifar'])
    parser.add_argument('--vgg_config', type=str, default='a', choices=['a', 'b', 'c', 'd', 'e'])
    # 하이퍼파라미터를 받아서 args에 저장
    args = parser.parse_args()
    # 하이퍼파라미터 반환
    return args

# 추론 시 하이퍼파라미터 파싱 함수
def infer_parse_args():
    # 하이퍼파라미터를 받기 위한 parser 객체 생성
    parser = argparse.ArgumentParser()
    # 하이퍼파라미터를 받기 위한 인자 추가
    parser.add_argument('--trained_folder', type=str)
    parser.add_argument('--target_image', type=str)
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # 하이퍼파라미터를 받아서 args에 저장
    args = parser.parse_args()
    # 하이퍼파라미터 반환
    return args