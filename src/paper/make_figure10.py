
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ========================================
# 1. 경로 설정
# ========================================
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "outputs" / "paper" / "weight_grid"
CSV_PATH = DATA_DIR / "weight_grid_diagnostics.csv"

OUTPUT_DIR = DATA_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ========================================
# 2. 데이터 불러오기
# ========================================
df = pd.read_csv(CSV_PATH)

required_cols = {
    "role", "war_weight", "accuracy", "high_f1"
}

missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# WAR 가중치 0.1~0.9만 사용
weights = np.round(np.arange(0.1, 1.0, 0.1), 1)

df = df[
    df["war_weight"].round(1).isin(weights)
].copy()

df["war_weight"] = df["war_weight"].round(1)

role_names = {
    "hitter": "Hitter",
    "sp": "Starting Pitcher",
    "rp": "Relief Pitcher",
}

# 27개 조건이 모두 존재하는지 검증
if len(df) != 27:
    raise ValueError(
        f"Expected 27 conditions, found {len(df)}"
    )

if df[["role", "war_weight"]].duplicated().any():
    raise ValueError("Duplicate role-weight conditions")

for role in role_names:
    role_df = df[df["role"] == role]
    if not np.array_equal(
        role_df["war_weight"].sort_values().to_numpy(),
        weights
    ):
        raise ValueError(f"Incomplete weight grid: {role}")

# ========================================
# 3. 논문용 기본 스타일
# ========================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ========================================
# 4. 단일 패널 생성 함수
# ========================================
def plot_metric(metric, title, ylabel, filename):
    fig = plt.figure(figsize=(6.6, 4.6))
    ax = fig.add_axes([0.12, 0.16, 0.83, 0.71])

    markers = {
        "hitter": "o",
        "sp": "s",
        "rp": "^",
    }

    for role, label in role_names.items():
        subset = (
            df[df["role"] == role]
            .sort_values("war_weight")
        )

        ax.plot(
            subset["war_weight"],
            subset[metric],
            marker=markers[role],
            linewidth=2,
            markersize=5.5,
            label=label,
        )

    # 기존 동일 가중치(0.5:0.5) 기준선
    ax.axvline(
        0.5,
        linestyle="--",
        linewidth=1.3,
        alpha=0.6,
    )

    ax.set_xticks(weights)
    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(0, 1.0)

    ax.set_xlabel("WAR Weight")
    ax.set_ylabel(ylabel)

    ax.set_title(title, pad=35)
    ax.grid(True, linestyle=":", alpha=0.3)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=3,
        frameon=False,
    )

    output_path = OUTPUT_DIR / filename

    fig.savefig(
        output_path,
        dpi=300,
    )
    plt.close(fig)

    print(f"Saved: {output_path}")
    return output_path


# ========================================
# 5. Accuracy / High-F1 그래프
# ========================================
accuracy_path = plot_metric(
    metric="accuracy",
    title="(a) Accuracy",
    ylabel="Accuracy",
    filename="Figure10a_Accuracy.png",
)

f1_path = plot_metric(
    metric="high_f1",
    title="(b) High-F1",
    ylabel="High-F1",
    filename="Figure10b_HighF1.png",
)

# ========================================
# 6. 두 그래프를 Figure 10으로 결합
# ========================================
left = Image.open(accuracy_path).convert("RGB")
right = Image.open(f1_path).convert("RGB")

combined = Image.new(
    "RGB",
    (left.width + right.width, max(left.height, right.height)),
)

combined.paste(left, (0, 0))
combined.paste(right, (left.width, 0))

figure10_path = OUTPUT_DIR / "Figure10_Weight_Sensitivity.png"

combined.save(
    figure10_path,
    dpi=(300, 300),
)

print("\nFigure 10 generation completed.")
print("Final figure:", figure10_path)

# ========================================
# 7. 주요 결과 출력
# ========================================
print("\n=== Best observed results (Test) ===")

for role, label in role_names.items():
    subset = df[df["role"] == role]

    acc_row = subset.loc[subset["accuracy"].idxmax()]
    f1_row = subset.loc[subset["high_f1"].idxmax()]

    print(f"\n{label}")
    print(
        f"Highest Accuracy: {acc_row['accuracy']:.3f}"
        f" (WAR weight={acc_row['war_weight']:.1f})"
    )
    print(
        f"Highest High-F1: {f1_row['high_f1']:.3f}"
        f" (WAR weight={f1_row['war_weight']:.1f})"
    )

print("\nNote: Test maxima are descriptive, not")
print("validated optimal weight selections.")
