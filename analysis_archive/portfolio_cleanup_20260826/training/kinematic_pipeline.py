# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix, roc_curve
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from validation_utils import nested_threshold_evaluation, update_fusion_config

matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
np.random.seed(SEED)
print('설정 완료')

def extract_kinematic_features(x, y, t):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    t = np.array(t, dtype=float)

    dt = np.diff(t) / 1000.0
    dt = np.where(dt == 0, 1e-6, dt)

    dx = np.diff(x); dy = np.diff(y)
    dist         = np.sqrt(dx**2 + dy**2)
    velocity     = dist / dt
    acceleration = np.abs(np.diff(velocity)) / dt[1:]
    jerk         = np.abs(np.diff(acceleration)) / dt[2:]

    def stats(arr):
        if len(arr) == 0: return [0.0, 0.0, 0.0]
        return [float(np.mean(arr)), float(np.std(arr)), float(np.max(arr))]

    feats = {}
    for name, arr in [('velocity', velocity), ('acceleration', acceleration), ('jerk', jerk)]:
        for i, s in enumerate(['mean', 'std', 'max']):
            feats[f'{name}_{s}'] = stats(arr)[i]

    feats['total_distance'] = float(np.sum(dist))
    feats['duration']       = float((t[-1] - t[0]) / 1000.0)
    feats['velocity_cv']    = feats['velocity_std'] / (feats['velocity_mean'] + 1e-6)
    return feats

FEATURE_COLS = [
    'velocity_mean', 'velocity_std', 'velocity_max', 'velocity_cv',
    'acceleration_mean', 'acceleration_std', 'acceleration_max',
    'jerk_mean', 'jerk_std', 'jerk_max',
    'total_distance', 'duration'
]
print(f'피처 수: {len(FEATURE_COLS)}개')

UCI_DIR = os.path.join(BASE_DIR, "data", "UCI", "hw_dataset")

def load_uci(data_dir):
    records = []
    for label, cls in [(1, 'parkinson'), (0, 'control')]:
        cls_dir = os.path.join(data_dir, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.endswith('.txt'): continue
            df = pd.read_csv(
                os.path.join(cls_dir, fname), sep=';', header=None,
                names=['X','Y','Z','Pressure','GripAngle','Timestamp','TestID']
            )
            df = df[df['TestID'] == 0].reset_index(drop=True)
            if len(df) < 10: continue
            feats = extract_kinematic_features(df['X'].values, df['Y'].values, df['Timestamp'].values)
            feats['label'] = label
            feats['source'] = 'UCI'
            records.append(feats)
    return pd.DataFrame(records)

uci_df = load_uci(UCI_DIR)
print(f'UCI: {len(uci_df)}명  PD={uci_df.label.sum()}  HC={len(uci_df)-uci_df.label.sum()}')

PAHAW_DIR = os.path.join(BASE_DIR, "data", "PaHaW", "PaHaW_public")
CORPUS_PATH = os.path.join(BASE_DIR, "data", "PaHaW", "PaHaW_files", "corpus_PaHaW.xlsx")

def load_svc(path):
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    rows = [list(map(float, l.strip().split())) for l in lines[1:n+1]]
    return pd.DataFrame(rows, columns=['Y','X','Timestamp','ButtonStatus','Azimuth','Altitude','Pressure'])

def load_pahaw(pahaw_dir, corpus_path):
    corpus = pd.read_excel(corpus_path)
    corpus['ID'] = corpus['ID'].astype(str).str.zfill(5)
    label_map = {r['ID']: (1 if r['Disease'] == 'PD' else 0) for _, r in corpus.iterrows()}
    records = []
    for subj_id in sorted(os.listdir(pahaw_dir)):
        subj_dir = os.path.join(pahaw_dir, subj_id)
        if not os.path.isdir(subj_dir): continue
        task8 = [f for f in os.listdir(subj_dir) if f.endswith('__8_1.svc')]
        if not task8: continue
        df = load_svc(os.path.join(subj_dir, task8[0]))
        df = df[df['ButtonStatus'] == 1].reset_index(drop=True)
        if len(df) < 10: continue
        label = label_map.get(subj_id)
        if label is None: continue
        feats = extract_kinematic_features(df['X'].values, df['Y'].values, df['Timestamp'].values)
        feats['label'] = label
        feats['source'] = 'PaHaW'
        records.append(feats)
    return pd.DataFrame(records)

pahaw_df = load_pahaw(PAHAW_DIR, CORPUS_PATH)
print(f'PaHaW: {len(pahaw_df)}명  PD={pahaw_df.label.sum()}  HC={len(pahaw_df)-pahaw_df.label.sum()}')

combined_df = pd.concat([uci_df, pahaw_df], ignore_index=True)

before = len(combined_df)
combined_df = combined_df[combined_df['duration'] < 1000].reset_index(drop=True)
print(f'이상값 제거: {before} → {len(combined_df)}명')
print(combined_df.groupby(['source','label']).size().unstack().rename(columns={0:'HC',1:'PD'}))

nan_count = combined_df[FEATURE_COLS].isnull().sum().sum()
inf_count = np.isinf(combined_df[FEATURE_COLS].values).sum()
print(f'NaN: {nan_count}개  Inf: {inf_count}개')

# 원본 raw 피처 (정규화 전)
X_raw   = combined_df[FEATURE_COLS].values.copy()
y       = combined_df['label'].values
sources = combined_df['source'].values
print(f'X_raw shape: {X_raw.shape}')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Pipeline: 소스 정규화 이후 추가 StandardScaler + RF
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1
    ))
])

fold_aucs, fold_f1s = [], []
all_probas, all_preds, all_true = [], [], []

for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_raw, y)):
    X_tr = X_raw[tr_idx].copy()
    X_vl = X_raw[vl_idx].copy()

    # 웹 서비스와 동일하게 raw 운동학 피처를 pipeline에 전달합니다.
    # StandardScaler는 pipeline 안에서 train fold에만 fit됩니다.
    pipeline.fit(X_tr, y[tr_idx])
    proba = pipeline.predict_proba(X_vl)[:, 1]
    pred  = pipeline.predict(X_vl)

    auc = roc_auc_score(y[vl_idx], proba)
    f1  = f1_score(y[vl_idx], pred)
    fold_aucs.append(auc)
    fold_f1s.append(f1)
    all_probas.extend(proba)
    all_preds.extend(pred)
    all_true.extend(y[vl_idx])
    print(f'Fold {fold+1}: AUC={auc:.4f}  F1={f1:.4f}')

print(f'\n평균 AUC: {np.mean(fold_aucs):.4f} (+/-{np.std(fold_aucs):.4f})')
print(f'평균 F1:  {np.mean(fold_f1s):.4f} (+/-{np.std(fold_f1s):.4f})')

fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#EEF4ED')
fpr, tpr, _ = roc_curve(all_true, all_probas)
overall_auc = roc_auc_score(all_true, all_probas)
axes[0].plot(fpr, tpr, color='#13315C', lw=2, label=f'AUC = {overall_auc:.3f}')
axes[0].plot([0,1],[0,1],'--',color='gray')
axes[0].set_title('ROC Curve (5-Fold OOF)', color='#13315C', fontweight='bold')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend()
axes[0].set_facecolor('#F8FAF8')
cm = confusion_matrix(all_true, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['HC','PD'], yticklabels=['HC','PD'])
axes[1].set_title('Confusion Matrix (5-Fold OOF)', color='#13315C', fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
plt.tight_layout()
plt.savefig('kinematic_results.png', dpi=100, bbox_inches='tight')
plt.close()
print('저장: kinematic_results.png')
print(classification_report(all_true, all_preds, target_names=['HC','PD']))

# inner fold에서 threshold를 정하고 outer fold에서 평가합니다.
nested_result = nested_threshold_evaluation(pipeline, X_raw, y)
kin_threshold = nested_result['deployment_threshold']
print(f"운동학 nested OOF AUC: {nested_result['auc']:.6f}")
print(f"운동학 nested sensitivity: {nested_result['sensitivity']:.6f}")
print(f"운동학 nested specificity: {nested_result['specificity']:.6f}")
print(f'운동학 nested median threshold: {kin_threshold:.6f}')

# 전체 raw 피처로 최종 pipeline을 학습합니다. 서비스도 동일한 raw 피처를 입력합니다.
pipeline.fit(X_raw, y)
importances = pd.Series(
    pipeline.named_steps['rf'].feature_importances_, index=FEATURE_COLS
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 5), facecolor='#EEF4ED')
importances.plot(kind='barh', ax=ax, color='#8DA9C4')
ax.set_title('피처 중요도 (전체 학습)', color='#13315C', fontweight='bold')
ax.set_facecolor('#F8FAF8')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('kinematic_importance.png', dpi=100, bbox_inches='tight')
plt.close()
print('저장: kinematic_importance.png')
print(importances)

MODEL_DIR = 'parkinson_web/models'
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, 'kinematic_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump({
        'pipeline':         pipeline,          # StandardScaler + RF
        'feature_cols':     FEATURE_COLS,
        'input_preprocessing': 'raw_features_then_pipeline_standard_scaler',
        'threshold':        kin_threshold,
        'oof_auc':          nested_result['auc'],
        'nested_sensitivity': nested_result['sensitivity'],
        'nested_specificity': nested_result['specificity'],
        'fold_thresholds': nested_result['fold_thresholds'],
    }, f)

print(f'저장 완료: {model_path}')
print(f'피처 ({len(FEATURE_COLS)}개): {FEATURE_COLS}')

import joblib

# 변수명이 'pipeline'으로 되어 있을 것입니다.
try:
    joblib.dump({
        'pipeline': pipeline,
        'feature_cols': FEATURE_COLS,
        'input_preprocessing': 'raw_features_then_pipeline_standard_scaler',
        'threshold': kin_threshold,
        'oof_auc': nested_result['auc'],
        'nested_sensitivity': nested_result['sensitivity'],
        'nested_specificity': nested_result['specificity'],
        'fold_thresholds': nested_result['fold_thresholds'],
    }, 'drawing_kinematic_model.pkl')
    update_fusion_config('modeling/fusion_config.json', 'kin', nested_result['auc'], kin_threshold)
    print("✅ 손그림 운동학 모델(drawing_kinematic_model.pkl) 저장 완료!")
except NameError:
    print("❌ 에러: 'pipeline' 변수가 없습니다. 파이프라인 생성 셀을 실행해주세요.")
    
import numpy as np

# Cell 12-14에서 이미 정의된 all_probas, all_true 사용
np.save('y_pred_kin.npy', np.array(all_probas))
np.save('y_true_kin.npy', np.array(all_true))

print("저장 완료")
print(f"y_pred_kin: shape={np.array(all_probas).shape}")
print(f"y_true_kin: PD={np.array(all_true).sum()}명, "
      f"HC={(np.array(all_true)==0).sum()}명")

