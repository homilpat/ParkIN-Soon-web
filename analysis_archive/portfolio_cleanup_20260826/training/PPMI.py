# ============================================================
# MODEL COMPARISON with BorderlineSMOTE
# Logistic Regression, Random Forest, XGBoost, SVM
# 5-Fold Stratified Cross-Validation
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, recall_score, precision_score,
                              f1_score, make_scorer, confusion_matrix)
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import warnings
from pathlib import Path
from validation_utils import (
    nested_threshold_evaluation,
    threshold_at_minimum_sensitivity,
    update_fusion_config,
)
warnings.filterwarnings('ignore')

COLOR_PD = '#C0392B'
COLOR_HC = '#2980B9'
COLOR_BG = '#F8F9FA'
GRAY     = '#7F8C8D'

# ── 데이터 로드 ────────────────────────────────────────────
# 경로는 본인 환경에 맞게 수정
BASE_DIR = Path(__file__).resolve().parent
TRAINING_DATA = BASE_DIR / "data" / "PPMI_BSIT12_SCOPA.csv"
if not TRAINING_DATA.exists():
    raise FileNotFoundError(
        f"학습 데이터가 없습니다: {TRAINING_DATA}\n"
        "먼저 `python prepare_ppmi_bsit.py`를 실행하세요."
    )
df = pd.read_csv(TRAINING_DATA)

# 피처 정의
BSIT_ITEMS  = ['BSIT_CHERRY','BSIT_DILL_PICKLE','BSIT_BANANA','BSIT_CHOCOLATE',
                'BSIT_CINNAMON','BSIT_GASOLINE','BSIT_LEMON','BSIT_ONION',
                'BSIT_PINEAPPLE','BSIT_ROSE','BSIT_SOAP','BSIT_SMOKE']
SCOPA_ITEMS = ['SCOPA_AUT5', 'SCOPA_AUT6', 'SCOPA_AUT7']
FEATURE_COLS = BSIT_ITEMS + SCOPA_ITEMS

X = df[FEATURE_COLS].values
y = df['LABEL_PD'].values   # PD=1, HC=0 (전처리에서 이미 생성됨)

print(f"샘플 수: {len(X)}  |  PD: {y.sum()}  |  HC: {(y==0).sum()}")
print(f"피처 수: {X.shape[1]}")

# ── 모델 정의 ──────────────────────────────────────────────
smote = BorderlineSMOTE(random_state=42, k_neighbors=5)

models = {
    'Logistic\nRegression': ImbPipeline([
        ('smote', smote),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42,
                                   class_weight='balanced'))
    ]),
    'Random\nForest': ImbPipeline([
        ('smote', smote),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42,
                                        class_weight='balanced'))
    ]),
    'XGBoost': ImbPipeline([
        ('smote', smote),
        ('clf', xgb.XGBClassifier(n_estimators=200, random_state=42,
                                   eval_metric='logloss'
                                   ))
    ]),
    'SVM': ImbPipeline([
        ('smote', smote),
        ('scaler', StandardScaler()),
        ('clf', SVC(probability=True, random_state=42,
                    class_weight='balanced'))
    ]),
}

# ── 스코어러 정의 ──────────────────────────────────────────
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

scorers = {
    'AUC':         make_scorer(roc_auc_score, response_method='predict_proba'),
    'Sensitivity': make_scorer(recall_score, pos_label=1),
    'Specificity': make_scorer(specificity_score),
    'F1':          make_scorer(f1_score, pos_label=1),
    'Precision':   make_scorer(precision_score, pos_label=1, zero_division=0),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── 교차검증 실행 ──────────────────────────────────────────
results = {}
print("\n[모델 비교 - 5-Fold Stratified CV + BorderlineSMOTE]\n")
print(f"{'Model':<22} {'AUC':>7} {'Sens':>7} {'Spec':>7} {'F1':>7}")
print("─" * 55)

for name, pipeline in models.items():
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scorers,
                                 return_train_score=False, n_jobs=-1)
    results[name] = {metric: cv_results[f'test_{metric}'] for metric in scorers}
    label = name.replace('\n', ' ')
    print(f"{label:<22} "
          f"{results[name]['AUC'].mean():.3f}  "
          f"{results[name]['Sensitivity'].mean():.3f}  "
          f"{results[name]['Specificity'].mean():.3f}  "
          f"{results[name]['F1'].mean():.3f}")
    
# ── 시각화 ────────────────────────────────────────────────
model_names = list(models.keys())
metrics     = ['AUC', 'Sensitivity', 'Specificity', 'F1']
metric_kr   = {'AUC':'AUC', 'Sensitivity':'민감도(Sensitivity)',
               'Specificity':'특이도(Specificity)', 'F1':'F1 Score'}

fig = plt.figure(figsize=(16, 10), facecolor=COLOR_BG)
fig.suptitle('Model Comparison with BorderlineSMOTE\n(5-Fold Stratified Cross-Validation)',
             fontsize=15, fontweight='bold', y=0.98)

gs      = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
palette = ['#2980B9', '#27AE60', '#E67E22', '#8E44AD']

for idx, metric in enumerate(metrics):
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    ax.set_facecolor('white')

    means = [results[m][metric].mean() for m in model_names]
    stds  = [results[m][metric].std()  for m in model_names]
    x_pos = np.arange(len(model_names))

    bars = ax.bar(x_pos, means, color=palette, alpha=0.85,
                  width=0.55, zorder=3, edgecolor='white', linewidth=1.2)
    ax.errorbar(x_pos, means, yerr=stds, fmt='none', color='#2C3E50',
                capsize=5, capthick=1.5, elinewidth=1.5, zorder=4)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + std + 0.012,
                f'{mean:.3f}', ha='center', va='bottom',
                fontsize=9.5, fontweight='bold', color='#2C3E50')

    # 최고 모델 금색 테두리
    best_idx = int(np.argmax(means))
    bars[best_idx].set_edgecolor('#F39C12')
    bars[best_idx].set_linewidth(2.5)

    ax.set_title(metric_kr[metric], fontsize=12, fontweight='bold',
                 pad=8, color='#2C3E50')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score', fontsize=9, color=GRAY)
    ax.axhline(0.5, color='lightgray', linestyle='--', linewidth=1, zorder=1)
    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

import os
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/model_comparison_borderline_smote.png',
            dpi=150, bbox_inches='tight', facecolor=COLOR_BG)
plt.show()
print("저장 완료: figures/model_comparison_borderline_smote.png")

# ── 요약 테이블 ────────────────────────────────────────────
print("\n[최종 성능 요약 - Mean ± Std]\n")
print(f"{'Model':<22}", end='')
for m in metrics:
    print(f"  {m:>18}", end='')
print()
print("─" * 100)

for name in model_names:
    label = name.replace('\n', ' ')
    print(f"{label:<22}", end='')
    for m in metrics:
        mean = results[name][m].mean()
        std  = results[name][m].std()
        print(f"  {mean:.3f} ± {std:.3f}   ", end='')
    print()

# Select the deployment model by discrimination (mean validation AUC). The
# operating sensitivity/specificity is then set separately using nested-CV thresholds.
best = max(model_names, key=lambda n: results[n]['AUC'].mean())
print("\n평균 교차검증 AUC 기준 최종 모델:")
print(f"  {best.replace(chr(10), ' ')}  (AUC={results[best]['AUC'].mean():.3f})")

from sklearn.model_selection import cross_val_predict, StratifiedKFold
import numpy as np

# Cell 1에서 이미 정의된 X, y, smote 사용
# XGBoost가 최종 선택됐으니 동일 파이프라인으로
oof_pipeline = ImbPipeline([
    ('smote', BorderlineSMOTE(random_state=42, k_neighbors=5)),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced'
    ))
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# OOF 예측값 추출 (학습에 쓰이지 않은 fold에서만 예측)
y_pred_ppmi = cross_val_predict(
    oof_pipeline, X, y,
    cv=skf,
    method='predict_proba'
)[:, 1]  # PD 클래스 확률만

y_true_ppmi = y.copy()

np.save('y_pred_ppmi.npy', y_pred_ppmi)
np.save('y_true_ppmi.npy', y_true_ppmi)

print(f"저장 완료")
print(f"y_pred_ppmi: shape={y_pred_ppmi.shape}, range=[{y_pred_ppmi.min():.4f}, {y_pred_ppmi.max():.4f}]")
print(f"y_true_ppmi: PD={y_true_ppmi.sum()}명, HC={(y_true_ppmi==0).sum()}명")

import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. 최적의 설정으로 최종 모델 파이프라인 생성 
# 성능이 가장 좋았던 XGBoost 기준
final_model = clone(oof_pipeline)

# 2. 전체 데이터(X, y)로 최종 학습 
# (이 셀을 실행하기 전에 데이터 로드 셀이 실행되어 있어야 합니다)
try:
    final_model.fit(X, y)
    # 3. 저장
    os.makedirs('modeling', exist_ok=True)
    joblib.dump(final_model, 'modeling/olf_con_model.pkl')
    print("후각/변비 모델(olf_con_model.pkl) 저장 완료")
except NameError:
    print("❌ 에러: X 또는 y 데이터가 정의되지 않았습니다. 데이터 로드 셀을 먼저 실행하세요.")
    
import joblib
import numpy as np

model = joblib.load('modeling/olf_con_model.pkl')

# 완전 정상 케이스: 냄새 12개 다 맡음, 변비 없음
normal = [1,1,1,1,1,1,1,1,1,1,1,1, 0,0,0]
# 완전 이상 케이스: 냄새 12개 다 못 맡음, 변비 최고
abnormal = [0,0,0,0,0,0,0,0,0,0,0,0, 3,3,3]

print("정상 입력 예측 확률:", model.predict_proba([normal]))
print("이상 입력 예측 확률:", model.predict_proba([abnormal]))
print("클래스 레이블:", model.classes_)

# inner fold에서 임계값을 정하고 outer fold에서 평가합니다.
nested_result = nested_threshold_evaluation(
    oof_pipeline, X, y, minimum_sensitivity=0.80
)
olf_threshold = threshold_at_minimum_sensitivity(y_true_ppmi, y_pred_ppmi, 0.80)
deployment_predictions = (y_pred_ppmi >= olf_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true_ppmi, deployment_predictions, labels=[0, 1]).ravel()
deployment_sensitivity = tp / (tp + fn)
deployment_specificity = tn / (tn + fp)
print(f"후각/배변 OOF AUC: {roc_auc_score(y_true_ppmi, y_pred_ppmi):.6f}")
print(f"후각/배변 nested OOF AUC: {nested_result['auc']:.6f}")
print(f"후각/배변 nested sensitivity: {nested_result['sensitivity']:.6f}")
print(f"후각/배변 nested specificity: {nested_result['specificity']:.6f}")
print(f"후각/배변 nested median threshold: {nested_result['deployment_threshold']:.6f}")
print(f"후각/배변 deployment threshold: {olf_threshold:.6f}")
print(f"후각/배변 deployment OOF sensitivity: {deployment_sensitivity:.6f}")
print(f"후각/배변 deployment OOF specificity: {deployment_specificity:.6f}")

import json
with open('modeling/olf_con_model_metadata.json', 'w', encoding='utf-8') as f:
    json.dump({
        'model': 'LogisticRegression',
        'evaluation': 'nested threshold CV (outer 5-fold, inner 4-fold)',
        'threshold_rule': nested_result['threshold_rule'],
        'auc': nested_result['auc'],
        'sensitivity': nested_result['sensitivity'],
        'specificity': nested_result['specificity'],
        'threshold': olf_threshold,
        'deployment_oof_sensitivity': deployment_sensitivity,
        'deployment_oof_specificity': deployment_specificity,
        'nested_median_threshold': nested_result['deployment_threshold'],
        'fold_thresholds': nested_result['fold_thresholds'],
        'features': FEATURE_COLS,
        'training_pipeline_matches_deployment': True,
    }, f, ensure_ascii=False, indent=2)
update_fusion_config('modeling/fusion_config.json', 'olf', nested_result['auc'], olf_threshold)
update_fusion_config('fusion_config.json', 'olf', nested_result['auc'], olf_threshold)
