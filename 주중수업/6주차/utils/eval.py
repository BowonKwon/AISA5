import torch                            # 파이토치 라이브러리 임포트

# 평가함수 구현
# 전체 데이터에 대한 정확도를 계산하는 함수
def evaluate(model, loader, device):    # 모델, 데이터 로더, 디바이스를 인자로 받음
    with torch.no_grad():               # 그래디언트 계산 비활성화
        model.eval()                    # 모델을 평가 모드로 설정
        total = 0                       # 전체 데이터 개수 저장 변수
        correct = 0                     # 정답 개수 저장 변수
        for images, targets in loader:  # 데이터 로더로부터 미니배치를 하나씩 꺼내옴
            images, targets = images.to(device), targets.to(device) # 디바이스에 데이터를 보냄
            output = model(images)      # 모델에 미니배치 데이터 입력하여 결괏값 계산
            output_index = torch.argmax(output, dim = 1) # 결괏값 중 가장 큰 값의 인덱스를 뽑아냄
            total += targets.shape[0]   # 전체 데이터 개수 누적
            correct += (output_index == targets).sum().item() # 정답 개수 누적

    acc = correct / total * 100 # 정확도(%) 계산
    model.train()               # 모델을 학습 모드로 설정
    return acc                  # 정확도(%) 반환

# 클래스별 정확도를 계산하는 함수
def evaluate_by_class(model, loader, device, num_classes):  # 모델, 데이터 로더, 디바이스, 클래스 개수를 인자로 받음
    with torch.no_grad():                                   # 그래디언트 계산 비활성화
        model.eval()                                        # 모델을 평가 모드로 설정
        total = torch.zeros(num_classes)                    # 클래스별 전체 데이터 개수 저장 변수
        correct = torch.zeros(num_classes)                  # 클래스별 정답 개수 저장 변수
        for images, targets in loader:                      # 데이터 로더로부터 미니배치를 하나씩 꺼내옴
            images, targets = images.to(device), targets.to(device) # 디바이스에 데이터를 보냄
            output = model(images)                          # 모델에 미니배치 데이터 입력하여 결괏값 계산
            output_index = torch.argmax(output, dim = 1)    # 결괏값 중 가장 큰 값의 인덱스를 뽑아냄
            for _class in range(num_classes):               # 클래스 개수만큼 반복
                total[_class] += (targets == _class).sum().item() # 클래스별 전체 데이터 개수 누적
                correct[_class] += ((targets == _class) * (output_index == _class)).sum().item() # 클래스별 정답 개수 누적

    acc = correct / total * 100 # 클래스별 정확도(%) 계산
    model.train()               # 모델을 학습 모드로 설정
    return acc                  # 클래스별 정확도(%) 반환