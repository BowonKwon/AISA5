# 패키지 임포트

# hyperparameter 선언

# 모델 설계도 그리기

# 설계도 바탕으로 모델 만들기 <- hyperparameter 사용

# 데이터 불러오기
    # dataset 설정(학습, 테스트 데이터)
    # dataloader 설정(학습, 테스터 데이터)

# loss 선언
# optimizer 선언

# 학습을 위한 반복 (loop) for / while
# 입력할 데이터를 위해 데이터 준비 (dataloader)
    # 데이터와 타깃을 디바이스에 올리기
    # 모델에 데이터 넣기
    # 모델의 출력과 정답을 비교하기 (loss 사용)
    # 역전파 수행
    # Loss 바탕으로 가중치 업데이트 진행 (optimizer)
    # 그래디언트 초기화

    # 평가(로깅, 출력), 저장