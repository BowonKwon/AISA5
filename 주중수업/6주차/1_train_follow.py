# 패키지 임포트
import os                                               # os 패키지 임포트
import torch                                            # 파이토치 패키지 임포트
import torch.nn as nn                                   # nn 패키지 임포트
from torch.optim import Adam                            # Adam 클래스 임포트
from networks.MLP_network import MLP                    # MLP 클래스 임포트
from utils.parser import parse_args                     # 하이퍼파라미터를 받기 위한 함수 임포트
from utils.target_folder import get_target_folder       # 결과를 저장할 폴더를 만들기 위한 함수 임포트
from utils.get_loader import get_loaders                # 데이터를 불러오기 위한 함수 임포트
from utils.eval import evaluate #, evaluate_by_class    # 정확도를 계산하기 위한 함수 임포트

# 메인 함수 정의
def main():
    # 하이퍼파라미터 받기
    args = parse_args()
    # 결과를 저장할 폴더 만들기
    target_folder = get_target_folder(args)
    # 모델 객체 만들기
    myMLP = MLP(args.image_size, args.hidden_size, args.num_classes).to(args.device)
    # 데이터 불러오기
    train_loader, test_loader = get_loaders(args)
    # Loss 선언
    loss_fn = nn.CrossEntropyLoss()
    # Optimizer 선언
    optim = Adam(params=myMLP.parameters(), lr=args.lr)

    _max = -1                                                   # 최대 정확도 저장 변수
    # 학습 시작
    for epoch in range(args.total_epochs):                      # 에포크 수만큼 반복
        for idx, (images, targets) in enumerate(train_loader):  # 데이터를 불러오기
            images = images.to(args.device)                     # 데이터를 디바이스에 올리기
            targets = targets.to(args.device)                   # 타깃을 디바이스에 올리기
            output = myMLP(images)                              # 모델에 이미지 입력 후 출력값 저장
            loss = loss_fn(output, targets)                     # loss 계산
            loss.backward()                                     # 역전파 수행
            optim.step()                                        # 그래디언트 업데이트
            optim.zero_grad()                                   # 그래디언트 초기화
            
            # 100번 반복마다 loss 출력
            if idx % 100 == 0:
                print(loss)
                
                # 전체 데이터(/클래스별) 정확도 계산
                acc = evaluate(myMLP, test_loader, args.device)
                # acc = evaluate_by_class(myMLP, test_loader, args.device, args.num_classes)
                
                # 정확도가 높아지면 모델 저장
                if _max < acc :                                         # 정확도가 높아지면
                    print('새로운 acc 등장, 모델 weight 업데이트', acc)   # 새로운 최대 정확도 출력
                    _max = acc                                          # 최대 정확도 업데이트
                    torch.save(                                         # 모델 저장
                        myMLP.state_dict(),
                        os.path.join(target_folder, 'myMLP_best.ckpt')
                    )
                    
# 이 파일이 메인 파일이면 main 함수 실행
if __name__ == '__main__' :
    main()                 