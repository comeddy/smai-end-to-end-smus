"""
Bank Marketing 데이터 전처리 스크립트

노트북 1(1-preprocessing.ipynb)과 동일한 로직을 SageMaker Processing 잡으로 실행합니다.
SageMaker Processing 컨테이너의 표준 경로(/opt/ml/processing)를 사용합니다.

입력:  /opt/ml/processing/input/bank-additional-full.csv
출력:  /opt/ml/processing/output/
         train/train.csv           -- 훈련 데이터 (70%, 레이블 첫 컬럼, 헤더 없음)
         validation/validation.csv -- 검증 데이터 (20%, 레이블 첫 컬럼, 헤더 없음)
         test/test_x.csv           -- 테스트 피처 (10%, 헤더 없음)
         test/test_y.csv           -- 테스트 레이블 (10%)
         baseline/baseline.csv    -- Model Monitor 베이스라인용 (훈련과 동일)
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # SageMaker Processing 컨테이너의 기본 데이터 경로
    base_dir = "/opt/ml/processing"

    # 쉼표(,) 구분자 CSV 로드 (이 S3 파일은 쉼표 구분자로 저장되어 있습니다)
    df = pd.read_csv(f"{base_dir}/input/bank-additional-full.csv", sep=",")

    # 컬럼명 정리: XGBoost는 '.' 포함 피처명을 처리하지 못하므로 '_'로 변환
    df.columns = [c.replace(".", "_") for c in df.columns]

    # 파생 변수 생성
    # pdays==999: 이전 캠페인 연락이 없었음을 의미하는 특수값 → 이진 플래그로 변환
    df["no_previous_contact"] = (df["pdays"] == 999).astype(int)
    # 비경제활동 인구 여부: 학생/은퇴자/실업자 → 이진 플래그
    df["not_working"] = df["job"].isin(["student", "retired", "unemployed"]).astype(int)

    # 불필요 컬럼 제거
    # duration: 실제 예측 시점에 알 수 없는 정보 누수(leakage) 컬럼
    # 경제 지표 컬럼: 노이즈 제거 목적
    drop_cols = ["duration", "emp_var_rate", "cons_price_idx",
                 "cons_conf_idx", "euribor3m", "nr_employed"]
    df.drop(columns=drop_cols, inplace=True)

    # 범주형 변수 원-핫 인코딩 후 정수형 변환 (XGBoost 입력 요건)
    df = pd.get_dummies(df).astype(int)

    # 레이블 컬럼(y_yes)을 첫 번째 컬럼으로 이동
    # XGBoost 내장 알고리즘은 첫 번째 컬럼을 레이블로 인식합니다
    label_col = "y_yes"
    cols = [label_col] + [c for c in df.columns if c != label_col]
    df = df[cols]

    # 클래스 불균형을 고려한 stratify 분할: 70% 훈련 / 20% 검증 / 10% 테스트
    # pd.get_dummies(df)는 y 컬럼(yes/no)에서 y_yes와 y_no 두 컬럼을 생성합니다.
    # label_col(y_yes)만 제거하면 y_no(= 1 - y_yes)가 피처로 남아 레이블 누수 발생.
    # y_ 로 시작하는 모든 컬럼을 피처에서 제거하여 누수를 방지합니다.
    X = df.drop(columns=[c for c in df.columns if c.startswith("y_")])
    y = df[label_col]

    # 1단계: 70/30 분할
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    # 2단계: 나머지 30%를 2:1로 분할 → 검증 20% / 테스트 10%
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=1/3, random_state=42, stratify=y_tmp)

    # 훈련/검증 데이터: 레이블 첫 컬럼, 헤더 없는 CSV (XGBoost 내장 알고리즘 요건)
    train_df = pd.concat([y_train, X_train], axis=1)
    val_df   = pd.concat([y_val,   X_val],   axis=1)

    # 출력 디렉터리 생성
    os.makedirs(f"{base_dir}/output/train",      exist_ok=True)
    os.makedirs(f"{base_dir}/output/validation", exist_ok=True)
    os.makedirs(f"{base_dir}/output/test",       exist_ok=True)
    os.makedirs(f"{base_dir}/output/baseline",   exist_ok=True)

    # CSV 저장: header=False, index=False (XGBoost 내장 알고리즘 입력 형식)
    train_df.to_csv(f"{base_dir}/output/train/train.csv",           index=False, header=False)
    val_df.to_csv(  f"{base_dir}/output/validation/validation.csv", index=False, header=False)
    # 테스트 데이터: 피처(X)와 레이블(y)을 분리 저장 → 평가 스크립트에서 독립 로드
    X_test.to_csv(  f"{base_dir}/output/test/test_x.csv",           index=False, header=False)
    y_test.to_csv(  f"{base_dir}/output/test/test_y.csv",           index=False, header=False)
    # 베이스라인: Model Monitor 드리프트 감지의 기준선으로 사용
    train_df.to_csv(f"{base_dir}/output/baseline/baseline.csv",     index=False, header=False)

    print(f"train      : {len(train_df):,}행")
    print(f"validation : {len(val_df):,}행")
    print(f"test       : {len(X_test):,}행")
