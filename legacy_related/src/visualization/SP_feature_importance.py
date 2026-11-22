# -*- coding: utf-8 -*-
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 경로
# =========================
RESULT_DIR = Path("data") / "results_threshold_final"
MODEL_PATH = RESULT_DIR / "pitcher_sp_expanded_stack.pkl"

# =========================
# 모델 불러오기
# =========================
pipe = joblib.load(MODEL_PATH)

print(f"모델 로드 완료: {MODEL_PATH}")

# =========================
# Stacking 모델 내부 보기
# =========================
model = pipe.named_steps["model"]
preprocessor = pipe.named_steps["preprocessor"]

print("\n[MODEL TYPE]")
print(type(model))

# Stacking은 일반적인 feature_importances_가 없음
# 대신 최종 메타모델 계수를 확인
final_est = model.final_estimator_
print("\n[FINAL ESTIMATOR]")
print(type(final_est))

# 전처리 후 feature name 추출
feature_names = []

# numeric
num_features = preprocessor.transformers_[0][2]
feature_names.extend(num_features)

# categorical
if len(preprocessor.transformers_) > 1:
    cat_transformer = preprocessor.named_transformers_.get("cat", None)
    cat_features = preprocessor.transformers_[1][2]
    if cat_transformer is not None:
        ohe = cat_transformer.named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(cat_features).tolist()
        feature_names.extend(cat_names)

print("\n[ORIGINAL INPUT FEATURE NAMES]")
print(feature_names)

# =========================
# Stacking 메타 모델 계수 출력
# =========================
if hasattr(final_est, "coef_"):
    coef = final_est.coef_.ravel()

    # 주의:
    # StackingClassifier의 final_estimator는
    # 원래 입력 feature가 아니라 base model들의 출력(probability) 등을 입력으로 받는다.
    # 따라서 여기 coef는 원래 feature importance가 아니라
    # "어떤 base model 예측을 더 신뢰했는지"에 가까움.
    print("\n[STACK META COEFFICIENTS]")
    print(coef)

    meta_names = [f"base_model_output_{i}" for i in range(len(coef))]
    meta_imp = pd.Series(coef, index=meta_names).sort_values(ascending=False)

    print("\n[STACK META IMPORTANCE]")
    print(meta_imp)

    meta_imp.to_csv(RESULT_DIR / "pitcher_sp_expanded_stack_meta_importance.csv", encoding="utf-8-sig")

    plt.figure(figsize=(8, 4))
    meta_imp.sort_values().plot(kind="barh")
    plt.title("Stack Meta Model Coefficients")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "pitcher_sp_expanded_stack_meta_importance.png", dpi=200)
    plt.close()

    print("\n저장 완료:")
    print(RESULT_DIR / "pitcher_sp_expanded_stack_meta_importance.csv")
    print(RESULT_DIR / "pitcher_sp_expanded_stack_meta_importance.png")

else:
    print("\n최종 추정기에 coef_가 없습니다.")