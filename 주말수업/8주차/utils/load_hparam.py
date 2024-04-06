import os                           # 경로 설정을 위한 os 패키지 임포트
import json                         # json 파일을 다루기 위한 json 패키지 임포트

# 하이퍼파라미터를 불러오기 위한 함수 정의
def load_hparams(args):
    # hparam.json 파일 불러오기
    with open(os.path.join(args.trained_folder, 'hparam.json'), 'r') as f:
        # json 파일을 파이썬 딕셔너리로 변환
        data = json.load(f)
    # 딕셔너리의 key와 value를 하이퍼파라미터로 저장
    for key, value in data.items():     # 딕셔너리의 key와 value를 하나씩 불러오기
        setattr(args, key, value)       # args에 key와 value 저장
    # 하이퍼파라미터를 저장한 args 반환
    return args