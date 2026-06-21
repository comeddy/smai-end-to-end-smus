"""
두 XGBoost 모델 비교 평가 스크립트

BankTrainConservative(model1)와 BankTrainAggressive(model2)를 테스트셋으로 평가하고
더 나은 AUC를 기록한 모델을 선정합니다. 결과는 evaluation.json에 저장되며,
BankAUCCondition 단계가 이 파일을 읽어 모델 등록 여부를 결정합니다.

입력:  /opt/ml/processing/model1/model.tar.gz  -- Conservative 모델 아티팩트
       /opt/ml/processing/model2/model.tar.gz  -- Aggressive 모델 아티팩트
       /opt/ml/processing/test/test_x.csv      -- 테스트 피처
       /opt/ml/processing/test/test_y.csv      -- 테스트 레이블

출력:  /opt/ml/processing/evaluation/evaluation.json
"""
import os, json, tarfile
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score

# SageMaker Processing 컨테이너의 기본 데이터 경로
BASE = "/opt/ml/processing"


def load_booster(model_dir):
    """
    model.tar.gz 아카이브에서 XGBoost 부스터를 로드합니다.

    SageMaker 훈련 잡은 모델 아티팩트를 model.tar.gz로 압축해 S3에 저장합니다.
    파일명은 훈련 스크립트 규약에 따라 'xgboost-model'이어야 합니다.
    """
    tar_path = os.path.join(model_dir, "model.tar.gz")
    if os.path.exists(tar_path):
        # tar.gz 아카이브 압축 해제 (같은 디렉터리에)
        with tarfile.open(tar_path) as t:
            t.extractall(model_dir)
    booster = xgb.Booster()
    # XGBoost 내장 알고리즘이 저장하는 표준 모델 파일명
    booster.load_model(os.path.join(model_dir, "xgboost-model"))
    return booster


def main():
    # 헤더 없는 CSV 로드 (전처리 스크립트의 header=False 저장 방식과 일치)
    test_x = pd.read_csv(f"{BASE}/test/test_x.csv", header=None)
    test_y = pd.read_csv(f"{BASE}/test/test_y.csv", header=None).squeeze()
    # XGBoost 예측을 위한 DMatrix 변환 (메모리 효율적 내부 포맷)
    dtest  = xgb.DMatrix(test_x)

    # 두 모델 로드
    model1 = load_booster(f"{BASE}/model1")  # Conservative (eta=0.5, num_round=50)
    model2 = load_booster(f"{BASE}/model2")  # Aggressive  (eta=0.1, num_round=100)

    # 예측 확률값 계산 (binary:logistic → 0~1 사이 양성 클래스 확률)
    pred1 = model1.predict(dtest)
    pred2 = model2.predict(dtest)

    # AUC: ROC 곡선 아래 넓이 (클수록 분류 성능이 좋음, 1.0이 완벽)
    auc1 = float(roc_auc_score(test_y, pred1))
    auc2 = float(roc_auc_score(test_y, pred2))
    # Accuracy: 0.5 임계값으로 이진 분류 후 정확도 계산
    acc1 = float(accuracy_score(test_y, (pred1 > 0.5).astype(int)))
    acc2 = float(accuracy_score(test_y, (pred2 > 0.5).astype(int)))

    # AUC 기준으로 더 나은 모델 선택
    best_model_num = 1 if auc1 >= auc2 else 2
    best_auc       = max(auc1, auc2)

    print(f"[Model 1 - Conservative] AUC: {auc1:.4f}  Accuracy: {acc1:.4f}")
    print(f"[Model 2 - Aggressive  ] AUC: {auc2:.4f}  Accuracy: {acc2:.4f}")
    print(f"=> Best: Model {best_model_num}  (AUC: {best_auc:.4f})")

    # evaluation.json 작성
    # PropertyFile이 이 파일을 읽어 BankAUCCondition 단계에서 best_auc 값을 추출합니다
    # json_path: "evaluation_metrics.best_auc.value"
    report = {
        "evaluation_metrics": {
            "model1_auc":      {"value": auc1,                  "standard_deviation": "NaN"},
            "model2_auc":      {"value": auc2,                  "standard_deviation": "NaN"},
            "model1_accuracy": {"value": acc1,                  "standard_deviation": "NaN"},
            "model2_accuracy": {"value": acc2,                  "standard_deviation": "NaN"},
            "best_auc":        {"value": best_auc,              "standard_deviation": "NaN"},
            "best_model":      {"value": float(best_model_num), "standard_deviation": "NaN"}
        }
    }
    os.makedirs(f"{BASE}/evaluation", exist_ok=True)
    with open(f"{BASE}/evaluation/evaluation.json", "w") as f:
        json.dump(report, f, indent=2)
    print("evaluation.json 저장 완료")

    # MLflow 로깅 (선택적)
    # pip install은 try 블록 내부에서 실행합니다.
    # 네트워크 장애 등 설치 실패가 evaluation.json 작성을 막으면 안 됩니다.
    # CLAUDE.md: "tracking failure never kills the job"
    try:
        mlflow_tracking_arn = os.environ.get("MLFLOW_TRACKING_ARN")
        exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
        if mlflow_tracking_arn:
            # XGBoost 컨테이너의 구버전 mlflow 충돌 방지:
            #   --target: /tmp/ml_deps에 별도 설치
            #   --no-deps: numpy 등 기존 패키지 재설치 방지 (ABI 충돌 방지)
            #   sys.path.append: 기존 numpy/xgboost 바인딩 우선 유지
            import subprocess, sys as _sys, shutil, importlib as _il
            _ml_dir = "/tmp/ml_deps"
            if os.path.exists(_ml_dir):
                shutil.rmtree(_ml_dir)  # 재시도 시 파일 누적 방지
            subprocess.check_call([_sys.executable, "-m", "pip", "install", "-q",
                                   "--target", _ml_dir, "--no-deps",
                                   "mlflow", "sagemaker-mlflow"])
            if _ml_dir not in _sys.path:
                _sys.path.append(_ml_dir)  # append: 기존 패키지 우선순위 보존
            _il.invalidate_caches()
            import mlflow
            mlflow.set_tracking_uri(mlflow_tracking_arn)
            mlflow.set_experiment(exp_name or "bank-pipeline")
            with mlflow.start_run(run_name="pipeline-evaluation"):
                # 파이프라인 평가 결과를 MLflow에 기록
                mlflow.log_metrics({
                    "pipeline_model1_auc":      auc1,
                    "pipeline_model2_auc":      auc2,
                    "pipeline_model1_accuracy": acc1,
                    "pipeline_model2_accuracy": acc2,
                    "pipeline_best_auc":        best_auc,
                    "pipeline_best_model":      float(best_model_num)
                })
                # 어느 모델이 최선인지 태그로 기록
                mlflow.set_tag("best_model", f"model{best_model_num}")
            print("MLflow 로깅 완료")
    except Exception as e:
        print(f"MLflow 로깅 스킵 (선택적): {e}")


if __name__ == "__main__":
    main()