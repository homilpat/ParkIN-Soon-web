import tensorflow as tf
import google.protobuf
print(tf.__version__)
print(google.protobuf.__version__)

import os, warnings, random
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import (roc_auc_score, roc_curve, recall_score,
                              f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from validation_utils import update_fusion_config

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE   = (224, 224)
BATCH_SIZE = 8
SEED       = 42
N_FOLDS    = 5
COLOR_BG   = '#F8F9FA'

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

os.makedirs('./models',  exist_ok=True)
os.makedirs('./figures', exist_ok=True)

print('TensorFlow:', tf.__version__)
print(f'BATCH_SIZE={BATCH_SIZE} | N_FOLDS={N_FOLDS}')

def collect_paths_labels(root_dir):
    classes   = sorted(os.listdir(root_dir))  # ['healthy', 'parkinson']
    label_map = {c: i for i, c in enumerate(classes)}
    print(f'  class_map: {label_map}')
    paths, labels = [], []
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(label_map[cls])
    return np.array(paths), np.array(labels)

SPIRAL_DIR = os.path.join(
    BASE_DIR, 'data', 'parkinsons-drawings', 'parkinsons-drawings', 'spiral'
)
paths_train, labels_train = collect_paths_labels(os.path.join(SPIRAL_DIR, 'training'))
paths_test, labels_test = collect_paths_labels(os.path.join(SPIRAL_DIR, 'testing'))
all_paths = np.concatenate([paths_train, paths_test])
all_labels = np.concatenate([labels_train, labels_test])

def subject_id_from_path(path):
    """V01HE02/V01PE02처럼 반복 측정 번호를 제외한 대상자 ID를 반환합니다."""
    match = re.match(r'^(V\d+[HP])', os.path.basename(path), re.IGNORECASE)
    if not match:
        raise ValueError(f'대상자 ID를 파일명에서 읽을 수 없습니다: {path}')
    return match.group(1).upper()

all_groups = np.array([subject_id_from_path(path) for path in all_paths])

# 기존 폴더 분할은 동일 대상자가 양쪽에 있어 사용하지 않습니다.
# 전체 이미지를 합친 뒤 대상자 단위 outer split의 첫 fold를 최종 holdout으로 고정합니다.
outer_group_split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
train_indices, test_indices = next(outer_group_split.split(all_paths, all_labels, all_groups))
sp_train_paths, sp_train_labels = all_paths[train_indices], all_labels[train_indices]
sp_train_groups = all_groups[train_indices]
sp_test_paths, sp_test_labels = all_paths[test_indices], all_labels[test_indices]
sp_test_groups = all_groups[test_indices]

if set(sp_train_groups) & set(sp_test_groups):
    raise RuntimeError('대상자 단위 train/holdout 분리에 실패했습니다.')

print(f'\nK-Fold 학습용: {len(sp_train_paths)}장')
print(f'  healthy:   {(sp_train_labels==0).sum()}장')
print(f'  parkinson: {(sp_train_labels==1).sum()}장')
print(f'\nHoldout 평가용: {len(sp_test_paths)}장')
print(f'  healthy:   {(sp_test_labels==0).sum()}장')
print(f'  parkinson: {(sp_test_labels==1).sum()}장')

def to_grayscale_3ch(img):
    """RGB 이미지를 grayscale로 변환 후 3채널 복제.
    rescale=1./255 이후 호출되므로 img 범위는 [0,1]
    """
    # 가중 평균으로 grayscale 변환 (인간 시각 기준)
    gray = (0.299 * img[..., 0]
          + 0.587 * img[..., 1]
          + 0.114 * img[..., 2])
    gray = gray[..., np.newaxis]           # (H, W, 1)
    return np.concatenate([gray, gray, gray], axis=-1)  # (H, W, 3)

print('Grayscale 전처리 함수 정의 완료')

# 동작 확인
dummy = np.random.rand(224, 224, 3).astype(np.float32)
out   = to_grayscale_3ch(dummy)
print(f'입력 shape: {dummy.shape}  →  출력 shape: {out.shape}')
print(f'3채널 동일 여부: {np.allclose(out[...,0], out[...,1])}')

train_aug = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=to_grayscale_3ch,  # grayscale 변환
    rotation_range=10,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.08,
    shear_range=0.08,
    brightness_range=[0.85, 1.15],
    # horizontal_flip 제거 (나선 방향 보존)
)

test_aug = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=to_grayscale_3ch,  # 평가/추론도 동일 적용
)

print('Augmentation 설정 완료 (grayscale 포함)')

fig, axes = plt.subplots(2, 4, figsize=(14, 7), facecolor=COLOR_BG)
fig.suptitle('Spiral Sample Images (Grayscale 변환 후)', fontsize=13, fontweight='bold')

for cls_idx, (cls, lbl) in enumerate([('healthy', 0), ('parkinson', 1)]):
    cls_paths = sp_train_paths[sp_train_labels == lbl][:4]
    for row, p in enumerate(cls_paths):
        ax_col = cls_idx * 2 + (row % 2)
        img = img_to_array(load_img(p, target_size=IMG_SIZE)) / 255.0
        img_gray = to_grayscale_3ch(img)
        axes[row // 2][ax_col].imshow(img_gray)
        axes[row // 2][ax_col].set_title(
            cls, fontsize=10,
            color='#2E7D52' if lbl == 0 else '#C0392B')
        axes[row // 2][ax_col].axis('off')

plt.tight_layout()
plt.savefig('./figures/samples_gray.png', dpi=120, bbox_inches='tight')
plt.show()

def build_model(input_shape=(224, 224, 3)):
    base = MobileNetV2(weights='imagenet', include_top=False,
                       input_shape=input_shape)
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-3))(x)
    x = Dropout(0.3)(x)
    x = Dense(64,  activation='relu', kernel_regularizer=l2(1e-3))(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base.input, outputs=output)
    return model, base

def make_generators(tr_paths, tr_labels, vl_paths, vl_labels):
    train_df = pd.DataFrame({'filename': tr_paths, 'class': tr_labels.astype(str)})
    val_df   = pd.DataFrame({'filename': vl_paths, 'class': vl_labels.astype(str)})

    train_gen = train_aug.flow_from_dataframe(
        train_df, x_col='filename', y_col='class',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=True, seed=SEED)

    val_gen = test_aug.flow_from_dataframe(
        val_df, x_col='filename', y_col='class',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=False)

    return train_gen, val_gen

print('모델 빌드 함수 정의 완료')

skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

fold_results = []
best_auc     = 0.0
best_model   = None
oof_preds    = np.full(len(sp_train_labels), np.nan, dtype=float)
fold_models  = []

# K-Fold: sp_train_paths 만 사용
for fold, (train_idx, val_idx) in enumerate(
        skf.split(sp_train_paths, sp_train_labels, sp_train_groups), 1):
    print(f"\n{'='*55}")
    print(f'  Fold {fold}/{N_FOLDS}  |  Train {len(train_idx)}장  Val {len(val_idx)}장')
    print(f"{'='*55}")

    tr_paths, tr_labels = sp_train_paths[train_idx], sp_train_labels[train_idx]
    vl_paths, vl_labels = sp_train_paths[val_idx],   sp_train_labels[val_idx]

    # 클래스 불균형 대응
    cw = compute_class_weight('balanced', classes=np.unique(tr_labels), y=tr_labels)
    class_weight_dict = dict(enumerate(cw))

    train_gen, val_gen = make_generators(tr_paths, tr_labels, vl_paths, vl_labels)
    model, base = build_model()

    # Stage 1 — classifier only
    print('\n[Stage 1] Classifier 학습 (base freeze)')
    model.compile(optimizer=Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    cb1 = [
        EarlyStopping(monitor='val_auc', patience=10,
                      restore_best_weights=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-6)
    ]
    model.fit(train_gen, validation_data=val_gen,
              epochs=40, callbacks=cb1,
              class_weight=class_weight_dict, verbose=1)

    # Stage 2 — fine-tuning
    print('\n[Stage 2] Fine-tuning (상위 30레이어 unfreeze, lr=1e-5)')
    for layer in base.layers[-30:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(optimizer=Adam(1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    cb2 = [
        EarlyStopping(monitor='val_auc', patience=8,
                      restore_best_weights=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=4, min_lr=1e-7)
    ]
    model.fit(train_gen, validation_data=val_gen,
              epochs=30, callbacks=cb2,
              class_weight=class_weight_dict, verbose=1)

    # Fold 평가
    eval_df  = pd.DataFrame({'filename': vl_paths, 'class': vl_labels.astype(str)})
    eval_gen = test_aug.flow_from_dataframe(
        eval_df, x_col='filename', y_col='class',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=False)

    preds  = model.predict(eval_gen, verbose=0).flatten()
    labels = eval_gen.labels
    oof_preds[val_idx] = preds
    fold_models.append(model)

    auc  = roc_auc_score(labels, preds)
    sens = recall_score(labels, (preds > 0.5).astype(int))
    spec = recall_score(labels, (preds > 0.5).astype(int), pos_label=0)
    f1   = f1_score(labels, (preds > 0.5).astype(int))

    fold_results.append({'Fold': fold, 'AUC': auc,
                         'Sensitivity': sens, 'Specificity': spec, 'F1': f1})
    print(f'\n  Fold {fold} | AUC={auc:.4f} | Sens={sens:.4f} | Spec={spec:.4f} | F1={f1:.4f}')

    if auc > best_auc:
        best_auc   = auc
        best_model = model
        print('  >> Best AUC 갱신')

print(f"\n{'='*55}")
print(f'K-Fold 완료  |  Best CV AUC: {best_auc:.4f}')
print(f"{'='*55}")

if np.isnan(oof_preds).any():
    raise RuntimeError("일부 training sample의 OOF 예측이 생성되지 않았습니다.")
oof_auc = roc_auc_score(sp_train_labels, oof_preds)
oof_fpr, oof_tpr, oof_thresholds = roc_curve(sp_train_labels, oof_preds)
finite_mask = np.isfinite(oof_thresholds)
img_threshold = float(
    oof_thresholds[finite_mask][np.argmax(oof_tpr[finite_mask] - oof_fpr[finite_mask])]
)
np.save('y_pred_img_oof.npy', oof_preds)
np.save('y_true_img_oof.npy', sp_train_labels)
print(f'진짜 fold별 OOF AUC: {oof_auc:.4f}')
print(f'이미지 OOF Youden threshold: {img_threshold:.6f}')

df = pd.DataFrame(fold_results)
print('[Fold별 성능]')
print(df.to_string(index=False))
print('\n[평균 +/- 표준편차]')
for col in ['AUC', 'Sensitivity', 'Specificity', 'F1']:
    print(f'  {col:<15}: {df[col].mean():.4f} +/- {df[col].std():.4f}')

fig, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor=COLOR_BG)
fig.suptitle('K-Fold Cross Validation Results', fontsize=13, fontweight='bold')
colors = ['#13315C', '#2E7D52', '#B5620A', '#8DA9C4']
for ax, metric, color in zip(axes, ['AUC','Sensitivity','Specificity','F1'], colors):
    vals = df[metric].values
    ax.bar(range(1, N_FOLDS+1), vals, color=color, alpha=0.85, edgecolor='white')
    ax.axhline(vals.mean(), color='red', linestyle='--', lw=1.5,
               label=f'Mean={vals.mean():.3f}')
    ax.set_xlabel('Fold')
    ax.set_title(metric, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('./figures/kfold_results.png', dpi=150, bbox_inches='tight')
plt.show()

best_model.save('./models/spiral_model.keras')
print('spiral_model.keras 저장 완료')

# Holdout 평가 (testing 폴더 — 완전한 미지 데이터)
test_df  = pd.DataFrame({'filename': sp_test_paths,
                          'class':    sp_test_labels.astype(str)})
test_gen = test_aug.flow_from_dataframe(
    test_df, x_col='filename', y_col='class',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', shuffle=False)

loaded = tf.keras.models.load_model('./models/spiral_model.keras')
preds  = loaded.predict(test_gen, verbose=0).flatten()
labels = test_gen.labels

print(f'class_indices: {test_gen.class_indices}')
print(f'healthy   예측 평균: {preds[labels==0].mean():.4f}  (낮을수록 정상)')
print(f'parkinson 예측 평균: {preds[labels==1].mean():.4f}  (높을수록 PD)')
print(f'\n=== Holdout 최종 성능 ===')
print(f'AUC:         {roc_auc_score(labels, preds):.4f}')
holdout_classes = (preds >= img_threshold).astype(int)
print(f'Threshold:   {img_threshold:.6f} (training OOF에서 선택)')
print(f'Sensitivity: {recall_score(labels, holdout_classes):.4f}')
print(f'Specificity: {recall_score(labels, holdout_classes, pos_label=0):.4f}')
print(f'F1:          {f1_score(labels, holdout_classes):.4f}')
print(classification_report(labels, holdout_classes,
                             target_names=['healthy','parkinson']))

cm = confusion_matrix(labels, holdout_classes)
fig, ax = plt.subplots(1, 1, figsize=(5, 4), facecolor=COLOR_BG)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['healthy', 'parkinson'],
            yticklabels=['healthy', 'parkinson'],
            annot_kws={'size': 16})
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('True',      fontsize=12)
ax.set_title('Confusion Matrix — Holdout Test', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('./figures/confusion_matrix_holdout.png', dpi=150, bbox_inches='tight')
plt.show()

import json
with open('./models/spiral_model_metadata.json', 'w', encoding='utf-8') as f:
    json.dump({
        'model': 'MobileNetV2',
        'threshold_source': 'training fold-specific OOF Youden',
        'threshold': img_threshold,
        'oof_auc': float(oof_auc),
        'holdout_auc': float(roc_auc_score(labels, preds)),
        'holdout_sensitivity': float(recall_score(labels, holdout_classes)),
        'holdout_specificity': float(recall_score(labels, holdout_classes, pos_label=0)),
    }, f, ensure_ascii=False, indent=2)
update_fusion_config(
    'modeling/fusion_config.json',
    'img',
    float(roc_auc_score(labels, preds)),
    img_threshold,
)

# Everything below this point is retained only as an old notebook appendix.
# It repeats training, overwrites predictions, and contains a developer-specific
# save path, so a normal script run must stop after the validated export above.
if __name__ == '__main__':
    raise SystemExit(0)

# [NEW CELL] - 학습부터 그래프 저장까지 한 번에 수행
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, recall_score, f1_score

skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

fold_results = []
all_histories = [] # <--- [추가] 모든 학습 기록을 저장할 바구니
best_auc     = 0.0
best_model   = None

for fold, (train_idx, val_idx) in enumerate(
        skf.split(sp_train_paths, sp_train_labels, sp_train_groups), 1):
    print(f"\n{'='*55}")
    print(f'  Fold {fold}/{N_FOLDS}  |  Train {len(train_idx)}장  Val {len(val_idx)}장')
    print(f"{'='*55}")

    tr_paths, tr_labels = sp_train_paths[train_idx], sp_train_labels[train_idx]
    vl_paths, vl_labels = sp_train_paths[val_idx],   sp_train_labels[val_idx]

    cw = compute_class_weight('balanced', classes=np.unique(tr_labels), y=tr_labels)
    class_weight_dict = dict(enumerate(cw))

    train_gen, val_gen = make_generators(tr_paths, tr_labels, vl_paths, vl_labels)
    model, base = build_model()

    # Stage 1 — Classifier 학습
    print('\n[Stage 1] Classifier 학습 (base freeze)')
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    cb1 = [EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max'),
           ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)]
    h1 = model.fit(train_gen, validation_data=val_gen, epochs=40, callbacks=cb1, class_weight=class_weight_dict, verbose=1)

    # Stage 2 — Fine-tuning
    print('\n[Stage 2] Fine-tuning (상위 30레이어 unfreeze)')
    for layer in base.layers[-30:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization): layer.trainable = True

    model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    cb2 = [EarlyStopping(monitor='val_auc', patience=8, restore_best_weights=True, mode='max'),
           ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7)]
    h2 = model.fit(train_gen, validation_data=val_gen, epochs=30, callbacks=cb2, class_weight=class_weight_dict, verbose=1)

    # [추가] 이번 폴드의 학습 기록 저장
    fold_h = {
        'fold': fold,
        'loss': h1.history['loss'] + h2.history['loss'],
        'val_loss': h1.history['val_loss'] + h2.history['val_loss'],
        'auc': h1.history['auc'] + h2.history['auc'],
        'val_auc': h1.history['val_auc'] + h2.history['val_auc']
    }
    all_histories.append(fold_h)

    # Fold 평가
    eval_df = pd.DataFrame({'filename': vl_paths, 'class': vl_labels.astype(str)})
    eval_gen = test_aug.flow_from_dataframe(eval_df, x_col='filename', y_col='class', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)
    preds = model.predict(eval_gen, verbose=0).flatten()
    labels = eval_gen.labels

    auc = roc_auc_score(labels, preds)
    fold_results.append({'Fold': fold, 'AUC': auc})
    if auc > best_auc:
        best_auc = auc
        best_model = model

# --- [추가] 학습 기록 CSV 저장 ---
history_data = []
for h in all_histories:
    for i in range(len(h['loss'])):
        history_data.append([h['fold'], i+1, h['loss'][i], h['val_loss'][i], h['auc'][i], h['val_auc'][i]])
history_df = pd.DataFrame(history_data, columns=['fold', 'epoch', 'loss', 'val_loss', 'auc', 'val_auc'])
history_df.to_csv('drawing_training_history.csv', index=False)

# --- [추가] 결과 시각화 그래프 ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_df[history_df['fold']==1]['loss'], label='Train Loss')
plt.plot(history_df[history_df['fold']==1]['val_loss'], label='Val Loss')
plt.title('Loss Curve (Fold 1)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_df[history_df['fold']==1]['auc'], label='Train AUC')
plt.plot(history_df[history_df['fold']==1]['val_auc'], label='Val AUC')
plt.title('AUC Curve (Fold 1)')
plt.legend()
plt.show()

print(f"✅ 모든 학습 완료! 'drawing_training_history.csv'가 저장되었습니다.")

import cv2
import numpy as np

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None: pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# 1. 테스트용 이미지 한 장 준비 (PD_AH 폴더의 이미지 중 하나)
# 이미지 경로를 실제 있는 파일로 수정해 주세요!
test_img_path = sp_train_paths[0] 

img = tf.keras.preprocessing.image.load_img(test_img_path, target_size=IMG_SIZE)
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 2. 히트맵 생성 (MobileNetV2의 마지막 conv 레이어 'out_relu')
heatmap = make_gradcam_heatmap(img_array, best_model, "out_relu")

# 3. 원본 이미지와 겹치기
img = cv2.imread(test_img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img

plt.imshow(cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB))
plt.title("Grad-CAM: Parkinson Evidence Area")
plt.axis('off')
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def visualize_diagnostic_panel(img_path, model, last_conv_layer_name):
    # 1. 원본 이미지 로드 (RGB)
    original_img = cv2.imread(img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # 2. 그레이스케일 변환
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    
    # 3. 전처리 (이진화 및 리사이즈 - 모델이 실제로 보는 형태)
    preprocessed = cv2.resize(gray_img, (224, 224))
    
    # 4. 모델 입력용 데이터 생성
    img_input = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB) # 채널 3개로 복사
    img_input = np.expand_dims(img_input / 255.0, axis=0)
    
    # 5. Grad-CAM 히트맵 생성
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_input)
        loss = predictions[:, 0]
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    gate_f = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(output, gate_f)
    
    # 히트맵 후처리
    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)
    
    # 6. 시각화 (5개 패널)
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    titles = ['Original', 'Grayscale', 'Preprocessed', 'Activation', 'Grad-CAM']
    images = [original_img, gray_img, preprocessed, heatmap, None]
    
    for i in range(4):
        cmap = 'gray' if i in [1, 2] else ('jet' if i == 3 else None)
        axes[i].imshow(images[i], cmap=cmap)
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    
    # 최종 Grad-CAM 합성
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    # 원본 이미지와 겹치기 (크기 맞춤)
    resized_orig = cv2.resize(original_img, (224, 224))
    superimposed = cv2.addWeighted(resized_orig, 0.6, heatmap_color, 0.4, 0)
    
    axes[4].imshow(superimposed)
    axes[4].set_title(titles[4])
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.show()

# 실행 (학습 완료 후 실행하세요)
visualize_diagnostic_panel(sp_train_paths[0], best_model, "out_relu")

# [NEW CELL] - HC vs PD 5단계 진단 패널 시각화
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def get_5step_panel(img_path, model, label_name):
    # 1. 원본 로드
    orig = cv2.imread(str(img_path))
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    
    # 2. 그레이스케일 (단순 변환)
    gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    
    # 3. 전처리 (노트북 로직 반영: 224x224 & to_grayscale_3ch)
    img_224 = cv2.resize(orig, (224, 224)) / 255.0
    # 노트북에 정의된 to_grayscale_3ch 함수 사용
    preprocessed = to_grayscale_3ch(img_224) 
    
    # 4. Grad-CAM 히트맵 생성
    img_input = np.expand_dims(preprocessed, axis=0)
    last_conv_layer_name = "out_relu" # MobileNetV2의 마지막 층
    
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_input)
        loss = predictions[:, 0]
    
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    gate_f = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(output, gate_f)
    
    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / (np.max(cam) + 1e-10)
    
    # 5. 최종 합성 이미지
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(np.uint8(img_224 * 255), 0.6, heatmap_color, 0.4, 0)
    
    return [orig, gray, preprocessed, heatmap, superimposed]

# --- 시각화 실행 ---
# HC 샘플 하나, PD 샘플 하나 선정
hc_idx = np.where(sp_train_labels == 0)[0][0]
pd_idx = np.where(sp_train_labels == 1)[0][0]

samples = [
    (sp_train_paths[hc_idx], "Healthy Control (HC)"),
    (sp_train_paths[pd_idx], "Parkinson's Disease (PD)")
]

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
titles = ['Original', 'Grayscale', 'Preprocessed', 'Activation', 'Grad-CAM']

for row, (path, label) in enumerate(samples):
    panel_imgs = get_5step_panel(path, best_model, label)
    
    for col, img in enumerate(panel_imgs):
        if col == 2: # Preprocessed (3ch gray)
            axes[row, col].imshow(img)
        elif col == 3: # Heatmap
            axes[row, col].imshow(img, cmap='jet')
        else:
            axes[row, col].imshow(img, cmap='gray' if col == 1 else None)
            
        if row == 0: axes[row, col].set_title(titles[col], fontweight='bold')
        axes[row, col].axis('off')
    
    # 행 제목 추가
    axes[row, 0].text(-50, 112, label, fontsize=12, fontweight='bold', va='center', ha='right')

plt.tight_layout()
plt.show()

try:
    best_model.save('drawing_cnn_model.keras')
    print("✅ 손그림 이미지 모델(drawing_cnn_model.keras) 저장 완료!")
except NameError:
    print("❌ 에러: 'best_model' 변수가 없습니다. 모델 학습 셀을 먼저 실행해주세요.")
    
import numpy as np

preds  = loaded.predict(test_gen, verbose=0).flatten()
labels = np.array(test_gen.labels)  # 여기만 np.array로 감싸면 됨

np.save('y_pred_img.npy', preds)
np.save('y_true_img.npy', labels)

print("저장 완료")
print(f"y_pred_img: shape={preds.shape}, range=[{preds.min():.4f}, {preds.max():.4f}]")
print(f"y_true_img: shape={labels.shape}, PD={labels.sum()}명, HC={(labels==0).sum()}명")

import numpy as np
from sklearn.model_selection import StratifiedKFold

# OOF와 holdout은 평가 목적이 다르므로 하나의 배열로 합치지 않습니다.
np.save('y_pred_img_holdout.npy', preds)
np.save('y_true_img_holdout.npy', labels)
print(f"진짜 OOF {len(oof_preds)}장과 holdout {len(preds)}장을 분리 저장했습니다.")

best_model.save(
    os.path.join(BASE_DIR, "modeling", "drawing_cnn_savedmodel")
)
print("SavedModel 저장 완료")

