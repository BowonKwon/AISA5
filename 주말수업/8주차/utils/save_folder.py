import os       # 경로 설정을 위한 os 패키지 임포트 
import json     # 하이퍼파라미터를 json 파일로 저장하기 위한 json 패키지 임포트

# 타깃 폴더 생성 함수 정의
def get_save_path(args):
    
    # 상위 저장 폴더가 없으면 상위 저장 폴더 생성
    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)
    # 결과 저장할 하위 타깃 폴더 생성
    folder_name = max([0] + [int(e) for e in os.listdir(args.results_folder)])+1 # 하위 타깃 폴더 이름
    save_path = os.path.join(args.results_folder, str(folder_name))          # 하위 타깃 폴더 경로
    os.makedirs(save_path)                                                          # 하위 타깃 폴더 생성

    # 하이퍼파라미터를 json 파일로 저장
    with open(os.path.join(save_path, 'hparam.json'), 'w') as f:    # json 파일 생성
        write_args = args.__dict__.copy()                               # args를 딕셔너리로 변환
        del write_args['device']                                        # 디바이스는 저장하지 않음
        json.dump(write_args, f, indent=4)                              # json 파일에 저장

    # 하이퍼파라미터에 타깃 폴더 경로 저장
    args.save_path = save_path
    
    # 타깃 폴더 경로 반환
    return save_path