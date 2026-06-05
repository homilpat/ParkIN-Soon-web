"""
Parkin Soon (박인순) - 파킨슨병 조기 선별 키오스크
필요 파일 (modeling 폴더):
  olf_con_model.pkl / drawing_cnn_model.h5 / drawing_kinematic_model.pkl
  fusion_config.json / hospital_data.xlsx
"""

from __future__ import annotations

import os
import io
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_USE_LEGACY_KERAS"] = "0"

import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import cv2
import json
import re
import sqlite3
from streamlit_drawable_canvas import st_canvas
from pathlib import Path
import joblib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import keras
from PIL import Image, ImageDraw
import folium
from streamlit_folium import st_folium
import streamlit.elements.image as st_image

APP_DIR = Path(__file__).resolve().parent

def load_local_env(env_path=APP_DIR / ".env"):
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

load_local_env()

if not hasattr(st_image, "image_to_url"):
    def _canvas_image_to_url(image, width, clamp, channels, output_format, image_id):
        buffer = io.BytesIO()
        image.save(buffer, format=output_format or "PNG")
        return st.runtime.get_instance().media_file_mgr.add(
            buffer.getvalue(),
            f"image/{(output_format or 'png').lower()}",
            image_id,
        )

    st_image.image_to_url = _canvas_image_to_url

# ── 브랜드 색상 ────────────────────────────────────────────────────────────────
C_GREEN  = "#0D7C86"   # 브랜드 / 정상
C_ORANGE = "#A55A0A"   # 주의 / 경고
C_YELLOW = "#EFCA08"   # 강조 / 포인트
C_RED    = "#E63946"   # 이상 / 위험
C_DARK   = "#1A1A2E"   # 텍스트 다크
C_BG     = "#FAF7F5"   # 배경

# ── 상수 ──────────────────────────────────────────────────────────────────────
MODEL_DIR          = Path(os.environ.get("MODEL_DIR", APP_DIR / "modeling")).expanduser().resolve()
CUSTOMER_DB_PATH   = MODEL_DIR / "customer_records.db"
OLF_MODEL_FILE     = "olf_con_model.pkl"
IMAGE_MODEL_FILE   = "drawing_cnn_model.h5"
KIN_MODEL_FILE     = "drawing_kinematic_model.pkl"
FUSION_CONFIG_FILE = "fusion_config.json"
CANVAS_SIZE        = 400
IMAGE_MODEL_SIZE   = (224, 224)

T_OLF = 0.4617
T_IMG = 0.4510
T_KIN = 0.5418
NORMAL_OLF_PCT = 90.0
NORMAL_BOWEL_SCORE = 1.0

FEATURE_COLS = [
    'velocity_mean','velocity_std','velocity_max','velocity_cv',
    'acceleration_mean','acceleration_std','acceleration_max',
    'jerk_mean','jerk_std','jerk_max','total_distance','duration',
]

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700;800;900&display=swap');
* {{ font-family: 'Noto Sans KR', sans-serif !important; }}
.stApp {{ background-color: {C_BG}; }}
.block-container {{
    max-width: 1280px;
    padding-top: 1.2rem;
    padding-bottom: 7rem;
}}

/* 전체 기본 폰트 크기 */
html, body, [class*="css"] {{
    font-size: 22px !important;
}}
div[data-testid="stMarkdownContainer"] p {{ font-size:22px!important; }}
div[data-testid="stMarkdownContainer"] li {{ font-size:22px!important; }}
div[data-testid="stRadio"] label p {{ font-size:22px!important; }}
div[data-testid="stTextInput"] label p {{
    font-size:22px!important;
    font-weight:800!important;
}}
div[data-testid="stTextInput"] input {{
    font-size:22px!important;
    min-height:58px!important;
}}

div[data-testid="stButton"] button {{
    font-size:22px!important; font-weight:800!important;
    border-radius:14px!important; padding:20px 34px!important;
    transition:all 0.2s ease!important;
    min-height:64px!important;
}}
div[data-testid="stButton"] button[kind="primary"] {{
    background-color:{C_GREEN}!important;
    border-color:{C_GREEN}!important;
    color:white!important;
    box-shadow:0 10px 24px rgba(13,124,134,0.22)!important;
}}
div[data-testid="stButton"] button:hover {{
    transform:translateY(-1px);
    box-shadow:0 6px 20px rgba(0,0,0,0.12)!important;
}}
header[data-testid="stHeader"] {{ display:none; }}
.park-topbar {{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:16px;
    padding:10px 0 22px;
}}
.park-brand {{
    display:flex;
    align-items:center;
    gap:12px;
    font-size:34px;
    font-weight:900;
    color:#075E66;
}}
.park-brand img {{
    width:48px;
    height:48px;
    border-radius:50%;
    object-fit:cover;
    border:2px solid #e5e7eb;
}}
.park-user-pill {{
    background:white;
    border:1px solid #e7e1dd;
    color:#374151;
    border-radius:999px;
    padding:10px 16px;
    font-size:20px;
    font-weight:800;
    box-shadow:0 2px 12px rgba(15,23,42,0.05);
}}
.park-hero {{
    background:white;
    border:1px solid #e7e1dd;
    border-radius:22px;
    padding:26px 32px;
    margin-bottom:28px;
    box-shadow:0 10px 28px rgba(15,23,42,0.06);
    display:flex;
    align-items:center;
    gap:28px;
}}
.park-hero-character {{
    flex:0 0 150px;
    text-align:center;
}}
.park-hero-character img {{
    height:150px;
    object-fit:contain;
    filter:drop-shadow(0 8px 18px rgba(15,23,42,0.12));
}}
.park-speech {{
    position:relative;
    flex:1;
    background:#2E8791;
    color:white;
    border-radius:18px;
    padding:26px 32px;
    box-shadow:0 8px 18px rgba(13,124,134,0.18);
}}
.park-speech:before {{
    content:"";
    position:absolute;
    left:-18px;
    top:50%;
    transform:translateY(-50%);
    border-top:16px solid transparent;
    border-bottom:16px solid transparent;
    border-right:18px solid #2E8791;
}}
.park-speech h2 {{
    margin:0 0 12px;
    font-size:32px;
    font-weight:900;
    color:white;
}}
.park-speech p {{
    margin:0;
    font-size:24px!important;
    line-height:1.75;
    color:rgba(255,255,255,0.94);
    font-weight:600;
}}
.park-soft-card {{
    background:white;
    border:1px solid #e7e1dd;
    border-radius:18px;
    box-shadow:0 8px 24px rgba(15,23,42,0.06);
}}
.park-section-title {{
    display:flex;
    align-items:center;
    gap:10px;
    font-size:27px;
    font-weight:900;
    color:#1f2937;
    margin:8px 0 16px;
}}
.park-bottom-bar {{
    position:fixed;
    left:0;
    right:0;
    bottom:0;
    z-index:990;
    background:rgba(255,255,255,0.92);
    border-top:1px solid #e7e1dd;
    backdrop-filter:blur(10px);
    padding:16px max(24px, calc((100vw - 1280px) / 2));
}}
@media (max-width: 900px) {{
    .park-hero {{ flex-direction:column; align-items:stretch; padding:22px; }}
    .park-hero-character {{ flex:auto; }}
    .park-speech:before {{ display:none; }}
    .park-topbar {{ flex-direction:column; align-items:flex-start; }}
}}
</style>
"""

# ── 유틸 ──────────────────────────────────────────────────────────────────────
def rerun():
    st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()

def init_state():
    defaults = {
        "step": 0, "result_page": 1,
        "olf_results": [], "scopa_results": [],
        "canvas_img": None, "canvas_json": None,
        "customer_name": "", "customer_phone": "",
        "customer_record_id": None, "result_saved": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def normalize_phone(phone):
    return re.sub(r"[^0-9]", "", phone or "")

def init_customer_db():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(CUSTOMER_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS customer_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                phone TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                olf_score INTEGER,
                scopa_score INTEGER,
                p_olf REAL,
                p_img REAL,
                p_kin REAL,
                abnormal_count INTEGER,
                risk_level TEXT,
                olf_results_json TEXT,
                scopa_results_json TEXT,
                canvas_json TEXT
            )
        """)
        existing_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(customer_records)").fetchall()
        }
        for col_name, col_type in {
            "olf_results_json": "TEXT",
            "scopa_results_json": "TEXT",
            "canvas_json": "TEXT",
        }.items():
            if col_name not in existing_cols:
                conn.execute(f"ALTER TABLE customer_records ADD COLUMN {col_name} {col_type}")

def create_customer_record(name, phone):
    now = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(CUSTOMER_DB_PATH) as conn:
        cur = conn.execute(
            """
            INSERT INTO customer_records (name, phone, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (name.strip(), normalize_phone(phone), now, now),
        )
        return cur.lastrowid

def update_customer_result(
    record_id,
    *,
    olf_score,
    scopa_score,
    p_olf,
    p_img,
    p_kin,
    abnormal_count,
    risk_level,
    olf_results,
    scopa_results,
    canvas_json,
):
    if not record_id:
        return
    now = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(CUSTOMER_DB_PATH) as conn:
        conn.execute(
            """
            UPDATE customer_records
               SET updated_at = ?, olf_score = ?, scopa_score = ?,
                   p_olf = ?, p_img = ?, p_kin = ?,
                   abnormal_count = ?, risk_level = ?,
                   olf_results_json = ?, scopa_results_json = ?, canvas_json = ?
             WHERE id = ?
            """,
            (
                now,
                olf_score,
                scopa_score,
                p_olf,
                p_img,
                p_kin,
                abnormal_count,
                risk_level,
                json.dumps(olf_results, ensure_ascii=False),
                json.dumps(scopa_results, ensure_ascii=False),
                json.dumps(canvas_json or {}, ensure_ascii=False),
                record_id,
            ),
        )

def get_customer_history(name, phone):
    phone_digits = normalize_phone(phone)
    if not name.strip() or not phone_digits:
        return []
    with sqlite3.connect(CUSTOMER_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, olf_score, scopa_score, canvas_json,
                   p_olf, p_img, p_kin
              FROM customer_records
             WHERE name = ? AND phone = ? AND olf_score IS NOT NULL
             ORDER BY created_at DESC
            """,
            (name.strip(), phone_digits),
        ).fetchall()
    return [dict(row) for row in rows]

def get_previous_customer_result(name, phone, current_record_id):
    phone_digits = normalize_phone(phone)
    if not name.strip() or not phone_digits:
        return None
    with sqlite3.connect(CUSTOMER_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT id, created_at, olf_score, scopa_score, canvas_json,
                   p_olf, p_img, p_kin
              FROM customer_records
             WHERE name = ? AND phone = ?
               AND olf_score IS NOT NULL
               AND (? IS NULL OR id != ?)
             ORDER BY created_at DESC
             LIMIT 1
            """,
            (name.strip(), phone_digits, current_record_id, current_record_id),
        ).fetchone()
    return dict(row) if row else None

def format_history_rows(rows):
    formatted = []
    for row in rows:
        raw_dt = row["created_at"] or ""
        display_dt = raw_dt.replace("T", " ")
        try:
            from datetime import datetime
            display_dt = datetime.fromisoformat(raw_dt).strftime("%Y년 %m월 %d일 %H:%M")
        except Exception:
            pass
        try:
            canvas = json.loads(row["canvas_json"] or "{}")
            point_count = len(canvas.get("x", []))
        except Exception:
            point_count = 0
        formatted.append({
            "id": row["id"],
            "검사일시": display_dt,
            "후각 점수": row["olf_score"],
            "배변 불편 점수": row["scopa_score"],
            "후각+배변 모델 확률": row["p_olf"],
            "손그림 이미지 모델 확률": row["p_img"],
            "손그림 운동학 모델 확률": row["p_kin"],
            "손그림 좌표 수": point_count,
        })
    return formatted

def format_model_signal(value):
    if value is None:
        return "-"
    return f"{float(value) * 100:.1f}%"

def signal_color(value, threshold):
    if value is None:
        return "#94a3b8"
    return C_RED if float(value) >= threshold else C_GREEN

def report_label(label):
    return f"<strong>{label}:</strong>"

def red_text(text):
    return f"<span style='color:{C_RED};font-weight:900;'>{text}</span>"

def percent_badge(value):
    return red_text(f"{value * 100:.1f}%")

def emphasize_symptoms(text):
    keywords = [
        "불규칙함",
        "불규칙한",
        "뚜렷하지 않습니다",
        "주의 신호",
        "떨림",
        "느려짐",
        "느린 편",
        "변비",
        "냄새가 둔해짐",
        "냄새를 잘 못 맡음",
        "힘을 많이 줌",
        "불편 신호",
        "이상 신호",
        "이상 패턴",
        "속도 변동",
        "배변 불편",
        "어려움",
    ]
    for keyword in keywords:
        text = text.replace(keyword, red_text(keyword))
    return text

def render_history_card(index, row):
    p_olf = row["후각+배변 모델 확률"]
    p_img = row["손그림 이미지 모델 확률"]
    p_kin = row["손그림 운동학 모델 확률"]
    st.markdown(f"""
<div style="background:white;border:1px solid #dbeafe;border-radius:16px;
            padding:18px 20px;margin-bottom:16px;
            box-shadow:0 2px 10px rgba(15,23,42,0.06);">
    <div style="display:flex;align-items:center;justify-content:space-between;
                gap:12px;flex-wrap:wrap;margin-bottom:12px;">
        <div style="font-size:22px;font-weight:900;color:{C_DARK};">
            {index}. {row['검사일시']}
        </div>
        <div style="font-size:16px;color:#64748b;background:#f1f5f9;
                    border-radius:999px;padding:5px 12px;">
            저장된 검사 결과
        </div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;
                margin-bottom:14px;">
        <div style="background:#f8fafc;border-radius:12px;padding:12px 14px;">
            <div style="font-size:17px;color:#64748b;">후각 점수</div>
            <div style="font-size:25px;font-weight:900;color:{C_DARK};">{row['후각 점수']} / 12</div>
        </div>
        <div style="background:#f8fafc;border-radius:12px;padding:12px 14px;">
            <div style="font-size:17px;color:#64748b;">배변 불편 점수</div>
            <div style="font-size:25px;font-weight:900;color:{C_DARK};">{row['배변 불편 점수']} / 9</div>
            <div style="font-size:14px;color:#ef4444;margin-top:4px;">높을수록 불편이 큽니다</div>
        </div>
    </div>
    <div style="font-size:18px;color:#475569;line-height:1.8;margin-bottom:10px;">
        아래 퍼센트는 <strong>AI 모델이 각 항목에서 주의 신호를 감지한 정도</strong>입니다.
        병을 진단하는 확률은 아닙니다.
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;">
        <div style="background:#f8fafc;border-radius:12px;padding:12px;">
            <div style="font-size:16px;color:#64748b;">후각+배변 주의 신호</div>
            <div style="font-size:22px;font-weight:900;color:{signal_color(p_olf, T_OLF)};">
                {format_model_signal(p_olf)}
            </div>
        </div>
        <div style="background:#f8fafc;border-radius:12px;padding:12px;">
            <div style="font-size:16px;color:#64748b;">그림 모양 주의 신호</div>
            <div style="font-size:22px;font-weight:900;color:{signal_color(p_img, T_IMG)};">
                {format_model_signal(p_img)}
            </div>
        </div>
        <div style="background:#f8fafc;border-radius:12px;padding:12px;">
            <div style="font-size:16px;color:#64748b;">손 움직임 주의 신호</div>
            <div style="font-size:22px;font-weight:900;color:{signal_color(p_kin, T_KIN)};">
                {format_model_signal(p_kin)}
            </div>
        </div>
    </div>
    <div style="font-size:17px;color:#64748b;margin-top:12px;">
        손그림 좌표는 안전하게 저장되었습니다. 저장된 점 개수: {row['손그림 좌표 수']}개
    </div>
</div>""", unsafe_allow_html=True)

def trend_text(current, previous, lower_is_better=True, unit="점"):
    if current is None or previous is None:
        return "비교할 이전 값이 부족합니다."
    diff = float(current) - float(previous)

    def fmt(value):
        if unit == "%p":
            return f"{float(value):.1f}%"
        if unit == "점":
            return f"{float(value):.1f}점"
        return f"{float(value):.1f}{unit}"

    if abs(diff) < 0.01:
        return f"지난 검사와 거의 비슷합니다. ({red_text(fmt(previous))} → {red_text(fmt(current))})"
    improved = diff < 0 if lower_is_better else diff > 0
    direction = "증가" if diff > 0 else "감소"
    change = red_text(f"{abs(diff):.1f}{unit} {direction}")
    values = f"{red_text(fmt(previous))} → {red_text(fmt(current))}, {change}"
    if improved:
        return f"지난 검사보다 좋아졌습니다. ({values})"
    return f"지난 검사보다 주의가 늘었습니다. ({values})"

def render_previous_comparison(previous_row, *, olf_score, con_score, p_olf, p_img, p_kin):
    if not previous_row:
        local_comment_card(
            "이전 검사와 비교",
            "비교할 이전 완료 기록이 아직 없습니다.<br>다음 검사부터는 변화 흐름을 함께 보여드릴 수 있습니다.",
            "#64748b",
        )
        return

    prev = format_history_rows([previous_row])[0]
    prev_dt = prev["검사일시"]
    comparison_lines = [
        trend_text(olf_score, previous_row.get("olf_score"), lower_is_better=False, unit="점"),
        trend_text(con_score, previous_row.get("scopa_score"), lower_is_better=True, unit="점"),
        trend_text(p_olf * 100, (previous_row.get("p_olf") or 0) * 100, lower_is_better=True, unit="%p"),
        trend_text(p_img * 100, (previous_row.get("p_img") or 0) * 100, lower_is_better=True, unit="%p"),
        trend_text(p_kin * 100, (previous_row.get("p_kin") or 0) * 100, lower_is_better=True, unit="%p"),
    ]
    body = (
        f"{report_label('이전 검사일')} {prev_dt}<br>"
        f"{report_label('후각')} {comparison_lines[0]}<br>"
        f"{report_label('배변 불편 점수')} {comparison_lines[1]}<br>"
        f"{report_label('후각+배변 주의 신호')} {comparison_lines[2]}<br>"
        f"{report_label('그림 모양 주의 신호')} {comparison_lines[3]}<br>"
        f"{report_label('손 움직임 주의 신호')} {comparison_lines[4]}"
    )
    local_comment_card("이전 검사와 비교", body, C_ORANGE)

def progress(cur):
    steps = ["소개", "후각", "배변", "손그림", "결과"]
    dots = ""
    for i, lbl in enumerate(steps):
        done = i < cur; active = i == cur
        dc = C_GREEN if (done or active) else "#f0eeee"
        ring = "0 0 0 9px #e8f4f5" if active else "none"
        lc = C_GREEN if active else ("#4b5563" if done else "#9ca3af")
        fw = "900" if active else "700"
        inn = "✓" if done else str(i + 1)
        tc = "white" if (done or active) else "#9ca3af"
        dots += f"""<div style="display:flex;flex-direction:column;align-items:center;gap:4px;flex:1;">
            <div style="width:46px;height:46px;border-radius:50%;background:{dc};
                display:flex;align-items:center;justify-content:center;
                font-size:22px;font-weight:900;color:{tc};box-shadow:{ring};">{inn}</div>
            <div style="font-size:21px;color:{lc};font-weight:{fw};">{lbl}</div></div>"""
        if i < len(steps) - 1:
            lc2 = C_GREEN if done else "#eee9e6"
            dots += f'<div style="flex:1;height:4px;background:{lc2};margin-top:21px;border-radius:999px;"></div>'
    st.markdown(
        f'<div class="park-soft-card" style="padding:24px 42px;'
        f'margin-bottom:34px;'
        f'display:flex;align-items:flex-start;">{dots}</div>',
        unsafe_allow_html=True
    )

# ── 나선 가이드라인 ────────────────────────────────────────────────────────────
@st.cache_data
def spiral_guide_img(w=CANVAS_SIZE, h=CANVAS_SIZE):
    """왼쪽 참고 가이드 - 검은 실선"""
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2
    nt = 3; mr = min(w, h) // 2 - 15
    tt = nt * 2 * np.pi; a = mr / tt
    px = py = None
    for deg in range(nt * 360 + 1):
        th = np.radians(deg); r = a * th
        x = cx + r * np.cos(th); y = cy + r * np.sin(th)
        if px is not None:
            draw.line([px, py, x, y], fill=(0, 0, 0), width=2)
        px, py = x, y
    return img

@st.cache_data
def spiral_img(w=CANVAS_SIZE, h=CANVAS_SIZE):
    """캔버스 배경 - 빨간 점선"""
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2
    nt = 3; mr = min(w, h) // 2 - 15
    tt = nt * 2 * np.pi; a = mr / tt
    ds, dg = 2, 8; px = py = None; acc = 0.0
    for deg in range(nt * 360 + 1):
        th = np.radians(deg); r = a * th
        x = cx + r * np.cos(th); y = cy + r * np.sin(th)
        if px is not None:
            acc += np.hypot(x - px, y - py)
            if acc >= dg:
                draw.ellipse([x - ds, y - ds, x + ds, y + ds], fill=(220, 0, 0))
                acc = 0.0
        px, py = x, y
    return img

# ── 모델 로드 ──────────────────────────────────────────────────────────────────
def req(p):
    if not p.exists(): raise FileNotFoundError(f"파일 없음: {p}")
    return p

def force_single_thread_prediction(model):
    """Windows 제한 환경에서 sklearn/joblib 병렬 예측이 막히지 않도록 합니다."""
    seen = set()

    def visit(obj):
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        if hasattr(obj, "n_jobs"):
            try:
                obj.n_jobs = 1
            except Exception:
                pass
        if hasattr(obj, "steps"):
            for _, step in obj.steps:
                visit(step)
        if hasattr(obj, "estimators_"):
            for estimator in obj.estimators_:
                visit(estimator)

    visit(model)
    return model

@st.cache_resource(show_spinner="모델을 불러오는 중...")
def load_models(md):
    import sys
    sys.setrecursionlimit(100000)
    
    mp = Path(md)
    m_olf = joblib.load(req(mp / OLF_MODEL_FILE))
    m_olf = force_single_thread_prediction(m_olf)
    
    # InputLayer 호환성 패치
    import tensorflow as tf
    from tensorflow.python.keras.layers import InputLayer
    
    class CompatInputLayer(tf.keras.layers.InputLayer):
        def __init__(self, **kwargs):
            shape_val = kwargs.pop('batch_shape', None)
            if shape_val is None:
                shape_val = kwargs.pop('shape', None)
            if shape_val is not None:
                if len(shape_val) == 4 and shape_val[0] is None:
                    shape_val = shape_val[1:]
                kwargs['input_shape'] = shape_val
            kwargs.pop('sparse', None)
            kwargs.pop('ragged', None)
            super().__init__(**kwargs)
    
    with tf.keras.utils.custom_object_scope({'InputLayer': CompatInputLayer}):
        import tempfile, shutil, os
        # 텐서플로우의 한글 경로 인코딩 버그를 완벽히 피하기 위해 임시 폴더(영문 경로)로 복사 후 로드
        temp_path = os.path.join(tempfile.gettempdir(), IMAGE_MODEL_FILE)
        shutil.copy2(str(mp / IMAGE_MODEL_FILE), temp_path)
        m_img = tf.keras.models.load_model(temp_path, compile=False)
        try:
            os.remove(temp_path)
        except:
            pass
    
    kd = joblib.load(req(mp / KIN_MODEL_FILE))
    m_kin = kd["pipeline"] if isinstance(kd, dict) and "pipeline" in kd else kd
    m_kin = force_single_thread_prediction(m_kin)
    with open(req(mp / FUSION_CONFIG_FILE), encoding="utf-8") as f:
        weights = json.load(f)["weights"]
    return m_olf, m_img, m_kin, weights

def get_img_base64(path):
    import base64
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

def app_header():
    char_b64 = get_img_base64(APP_DIR / "character.png")
    logo = (
        f'<img src="data:image/png;base64,{char_b64}" />'
        if char_b64 else
        '<div style="width:48px;height:48px;border-radius:50%;background:#0D7C86;color:white;display:flex;align-items:center;justify-content:center;font-weight:900;">P</div>'
    )
    name = st.session_state.get("customer_name", "").strip()
    user_text = f"{name} 님" if name else "검사 준비 중"
    st.markdown(
        f"""
<div class="park-topbar">
    <div class="park-brand">{logo}<span>ParkIN Soon</span></div>
    <div class="park-user-pill">{user_text}</div>
</div>
        """,
        unsafe_allow_html=True,
    )

def voice_guide_button(text):
    speech_text = re.sub(r"<[^>]+>", " ", text)
    speech_text = re.sub(r"\s+", " ", speech_text).strip()
    if not speech_text:
        return
    components.html(
        f"""
<script>
const parkinsoonSpeechText = {json.dumps(speech_text, ensure_ascii=False)};
function speakParkinsoonStepGuide() {{
  if (!("speechSynthesis" in window)) {{
    alert("이 브라우저에서는 음성 안내를 지원하지 않습니다.");
    return;
  }}
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(parkinsoonSpeechText);
  u.lang = "ko-KR";
  u.rate = 0.86;
  u.pitch = 1.0;
  window.speechSynthesis.speak(u);
}}
</script>
<div style="display:flex;justify-content:flex-end;margin:0 0 12px 0;">
  <button onclick="speakParkinsoonStepGuide()" style="
    border:1px solid #d9e8ea;
    background:#ffffff;
    border-radius:999px;
    padding:12px 20px;
    font-size:18px;
    font-weight:800;
    color:#0D7C86;
    box-shadow:0 6px 18px rgba(15,23,42,0.10);
    cursor:pointer;">
    🔊 박인순 설명 다시 듣기
  </button>
</div>
        """,
        height=74,
    )

def step_guide_card(title, body):
    char_b64 = get_img_base64(APP_DIR / "character.png")
    char_html = (
        f'<img src="data:image/png;base64,{char_b64}" />'
        if char_b64 else
        '<div style="height:140px;"></div>'
    )
    voice_guide_button(f"{title}. {body}")
    st.markdown(
        f"""
<div class="park-hero">
    <div class="park-hero-character">{char_html}</div>
    <div class="park-speech">
        <h2>{title}</h2>
        <p>{body}</p>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    
@st.cache_data
def load_hospitals():
    import pandas as pd
    # modeling 폴더와 프로젝트 루트 모두 지원
    csv_path = MODEL_DIR / "hospital_data.csv"
    xlsx_path = MODEL_DIR / "hospital_data.xlsx"
    root_csv_path = APP_DIR / "hospital_data.csv"
    root_xlsx_path = APP_DIR / "hospital_data.xlsx"
    if csv_path.exists():
        df = pd.read_csv(str(csv_path), encoding='utf-8-sig')
    elif xlsx_path.exists():
        df = pd.read_excel(str(xlsx_path))
    elif root_csv_path.exists():
        df = pd.read_csv(str(root_csv_path), encoding='utf-8-sig')
    elif root_xlsx_path.exists():
        df = pd.read_excel(str(root_xlsx_path))
    else:
        return pd.DataFrame()
    df = df[df['요양기관명'].str.contains('신경과', na=False)].copy()
    return df.dropna(subset=['좌표(X)', '좌표(Y)']).reset_index(drop=True)

# ── 예측 ──────────────────────────────────────────────────────────────────────
def pred_proba(model, features):
    x = np.asarray(features, dtype=np.float64).reshape(1, -1)
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(x))
        return float(p[0, 1] if p.ndim == 2 and p.shape[1] >= 2 else p.ravel()[0])
    return float(np.asarray(model.predict(x)).ravel()[0])

def extract_drawing_only(img_arr):
    """캔버스에서 사용자가 그린 선만 추출합니다."""
    rgba = np.asarray(img_arr, dtype=np.float32)
    if rgba.shape[2] == 3:
        return rgba.astype(np.uint8)
    alpha = rgba[:, :, 3:4] / 255.0
    white = np.ones_like(rgba[:, :, :3]) * 255.0
    drawing = rgba[:, :, :3] * alpha + white * (1.0 - alpha)
    return drawing.astype(np.uint8)

def preprocess_canvas(img_arr):
    arr = np.asarray(img_arr, dtype=np.float32)
    if arr.shape[2] == 3:
        comp = arr
    else:
        alpha = arr[:, :, 3:4] / 255.0
        comp = arr[:, :, :3] * alpha + 255.0 * (1.0 - alpha)
    t = tf.image.resize(tf.convert_to_tensor(comp, tf.float32), IMAGE_MODEL_SIZE)
    t = tf.repeat(tf.image.rgb_to_grayscale(t), 3, axis=-1) / 255.0
    return np.expand_dims(t.numpy().astype(np.float32), 0)

def extract_kin(x, y, t):
    x, y, t = map(np.array, (x, y, t))
    dt = np.where((d := np.diff(t) / 1000.0) == 0, 1e-6, d)
    dist = np.hypot(np.diff(x), np.diff(y)); vel = dist / dt
    acc = np.abs(np.diff(vel)) / dt[1:]; jrk = np.abs(np.diff(acc)) / dt[2:]
    def s(a): return [float(f(a)) if len(a) else 0.0 for f in (np.mean, np.std, np.max)]
    feats = {}
    for nm, arr in [("velocity", vel), ("acceleration", acc), ("jerk", jrk)]:
        for i, sf in enumerate(["mean", "std", "max"]): feats[f"{nm}_{sf}"] = s(arr)[i]
    feats["total_distance"] = float(np.sum(dist))
    feats["duration"] = float((t[-1] - t[0]) / 1000.0)
    feats["velocity_cv"] = feats["velocity_std"] / (feats["velocity_mean"] + 1e-6)
    return feats

def get_kin_feats(cj):
    try:
        x, y, t = cj.get("x", []), cj.get("y", []), cj.get("t", [])
        if len(x) < 10: return [0.5] * len(FEATURE_COLS)
        f = extract_kin(x, y, t)
        return [f[c] for c in FEATURE_COLS]
    except: return [0.5] * len(FEATURE_COLS)

def make_gradcam(img_array, model):
    img_var = tf.Variable(img_array.astype(np.float32))
    with tf.GradientTape() as tape:
        preds = model(img_var, training=False)
        loss = preds[0, 0]
    grads = tape.gradient(loss, img_var).numpy()[0]
    heatmap = np.max(np.abs(grads), axis=-1)
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
    p_low = np.percentile(heatmap, 70)
    p_high = np.percentile(heatmap, 99)
    if p_high > p_low:
        heatmap = np.clip((heatmap - p_low) / (p_high - p_low), 0, 1)
    elif heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    orig = (img_array[0] * 255).astype(np.uint8)
    return cv2.addWeighted(orig, 0.5, heatmap_colored, 0.5, 0)

def local_comment_card(title, body, color=C_GREEN):
    st.markdown(f"""
<div style="background:#ffffff;border-left:5px solid {color};border-radius:12px;
            padding:16px 18px;margin:14px 0 20px;
            box-shadow:0 2px 10px rgba(15,23,42,0.06);">
    <div style="font-size:21px;font-weight:800;color:{C_DARK};margin-bottom:8px;">
        {title}
    </div>
    <div style="font-size:20px;color:#334155;line-height:1.75;">
        {body}
    </div>
</div>""", unsafe_allow_html=True)

def olf_local_comment(score):
    pct = score / 12 * 100
    gap = NORMAL_OLF_PCT - pct
    avg_text = (
        f"정상군 평균 {NORMAL_OLF_PCT:.0f}%보다 {gap:.0f}%p 낮습니다."
        if gap > 0 else
        f"정상군 평균 {NORMAL_OLF_PCT:.0f}%보다 {abs(gap):.0f}%p 높거나 비슷합니다."
    )
    if pct >= 75:
        body = (
            f"12개 냄새 중 {score}개를 맞히셨어요.<br>"
            f"{avg_text}<br>"
            "냄새를 구별하는 힘이 비교적 잘 유지된 것으로 보입니다.<br>"
            "이 결과만으로 건강 상태를 단정하지는 않습니다."
        )
        return C_GREEN, emphasize_symptoms(body)
    if pct >= 50:
        body = (
            f"12개 냄새 중 {score}개를 맞히셨어요.<br>"
            f"{avg_text}<br>"
            "냄새를 구별하는 힘이 조금 낮게 보일 수 있습니다.<br>"
            "컨디션, 감기, 비염도 영향을 줄 수 있어요."
        )
        return C_ORANGE, emphasize_symptoms(body)
    body = (
        f"12개 냄새 중 {score}개를 맞히셨어요.<br>"
        f"{avg_text}<br>"
        "냄새를 구별하는 데 어려움이 있었던 것으로 보입니다.<br>"
        "걱정이 되시면 신경과나 이비인후과 상담을 권합니다."
    )
    return C_RED, emphasize_symptoms(body)

def bowel_local_comment(score):
    gap = score - NORMAL_BOWEL_SCORE
    avg_text = (
        f"일반 평균 {NORMAL_BOWEL_SCORE:.0f}점보다 {gap:.0f}점 높습니다."
        if gap > 0 else
        f"일반 평균 {NORMAL_BOWEL_SCORE:.0f}점보다 낮거나 비슷합니다."
    )
    if score < 3:
        body = (
            f"배변 불편 점수는 {score}점입니다. 높을수록 불편이 큰 점수입니다.<br>"
            f"{avg_text}<br>"
            "최근 한 달 동안 큰 배변 불편은 적었던 것으로 보입니다.<br>"
            "물을 충분히 드시고, 규칙적인 식사를 유지해 주세요."
        )
        return C_GREEN, emphasize_symptoms(body)
    if score < 6:
        body = (
            f"배변 불편 점수는 {score}점입니다. 높을수록 불편이 큰 점수입니다.<br>"
            f"{avg_text}<br>"
            "배변 불편이 조금 있었던 것으로 보입니다.<br>"
            "증상이 계속되면 의료진과 상담해 보세요."
        )
        return C_ORANGE, emphasize_symptoms(body)
    body = (
        f"배변 불편 점수는 {score}점입니다. 높을수록 불편이 큰 점수입니다.<br>"
        f"{avg_text}<br>"
        "배변 불편이 뚜렷하게 있었던 것으로 보입니다.<br>"
        "무리하지 마시고 전문의 상담을 받아보시는 것이 좋습니다."
    )
    return C_RED, emphasize_symptoms(body)

def drawing_local_comment(p_img, p_kin, shape_comment, kin_comment):
    if p_img >= T_IMG or p_kin >= T_KIN:
        color = C_ORANGE if not (p_img >= T_IMG and p_kin >= T_KIN) else C_RED
        body = (
            "AI가 그림의 선 모양과 움직임을 함께 살펴봤습니다.<br>"
            "색이 진한 부분은 AI가 중요하게 본 부분입니다.<br>"
            f"{shape_comment}<br>{kin_comment}<br>"
            "이 결과는 참고용입니다. 필요하면 전문의에게 보여주세요."
        )
    else:
        color = C_GREEN
        body = (
            "AI가 그림의 선 모양과 움직임을 함께 살펴봤습니다.<br>"
            "색이 진한 부분은 AI가 중요하게 본 부분입니다.<br>"
            "이번 손그림 결과에서는 큰 이상 신호가 뚜렷하지 않습니다.<br>"
            "다만 이 검사는 진단이 아니라 참고용입니다."
        )
    return color, emphasize_symptoms(body)

def final_local_comment(cnt, olf_score, con_score, p_olf, p_img, p_kin):
    olf_bowel_note = []
    if olf_score < 9:
        olf_bowel_note.append("냄새를 구별하는 힘이 낮아진 신호가 있을 수 있습니다")
    if con_score >= 3:
        olf_bowel_note.append("변비, 힘주기, 배변 불편 같은 자율신경 증상이 있을 수 있습니다")
    if not olf_bowel_note:
        olf_bowel_note.append("후각과 배변 항목에서는 큰 불편 신호가 뚜렷하지 않습니다")

    drawing_notes = []
    if p_img >= T_IMG:
        drawing_notes.append("나선 그림의 모양에서 선의 흔들림이나 불규칙함이 보일 수 있습니다")
    if p_kin >= T_KIN:
        drawing_notes.append("손의 움직임 속도 변화나 떨림 신호가 보일 수 있습니다")
    if not drawing_notes:
        drawing_notes.append("손그림 항목에서는 큰 이상 신호가 뚜렷하지 않습니다")

    if cnt == 0:
        color = C_GREEN
        body = (
            f"{report_label('위험도 점수')} 낮음<br>"
            f"{report_label('설명')} 세 가지 검사에서 큰 주의 신호는 뚜렷하지 않습니다.<br>"
            f"{report_label('모델 점수')} 후각+배변 {percent_badge(p_olf)}, 그림 모양 {percent_badge(p_img)}, 손 움직임 {percent_badge(p_kin)}입니다.<br>"
            f"{report_label('검사 결과')} 후각은 {olf_score}/12점, 배변 불편 점수는 {con_score}/9점입니다.<br>"
            f"{report_label('증상')} {olf_bowel_note[0]}. {drawing_notes[0]}.<br>"
            f"{report_label('예시 증상')} 냄새 저하, 변비, 손 떨림이 새로 생기는지 살펴보세요.<br>"
            f"{report_label('다음 행동')} 평소와 다른 증상이 생기면 의료진과 상담해 주세요."
        )
    elif cnt == 1:
        color = C_ORANGE
        body = (
            f"{report_label('위험도 점수')} 중간<br>"
            f"{report_label('설명')} 검사 중 한 항목에서 주의 신호가 보입니다.<br>"
            f"{report_label('모델 점수')} 후각+배변 {percent_badge(p_olf)}, 그림 모양 {percent_badge(p_img)}, 손 움직임 {percent_badge(p_kin)}입니다.<br>"
            f"{report_label('증상')} {' / '.join(olf_bowel_note)}. {' / '.join(drawing_notes)}.<br>"
            f"{report_label('예시 증상')} 냄새가 둔해짐, 변비가 이어짐, 선이 흔들림, 손 움직임이 느려짐이 있을 수 있습니다.<br>"
            f"{report_label('다음 행동')} 이 결과는 진단이 아닙니다. 비슷한 증상이 계속되면 신경과 상담을 권합니다."
        )
    else:
        color = C_RED
        body = (
            f"{report_label('위험도 점수')} 높음<br>"
            f"{report_label('설명')} 여러 항목에서 주의가 필요한 신호가 함께 보입니다.<br>"
            f"{report_label('모델 점수')} 후각+배변 {percent_badge(p_olf)}, 그림 모양 {percent_badge(p_img)}, 손 움직임 {percent_badge(p_kin)}입니다.<br>"
            f"{report_label('증상')} {' / '.join(olf_bowel_note)}. {' / '.join(drawing_notes)}.<br>"
            f"{report_label('예시 증상')} 냄새를 잘 못 맡음, 변비가 오래감, 대변을 볼 때 힘을 많이 줌, 손이 떨림, 글씨나 선이 불규칙해짐이 있을 수 있습니다.<br>"
            f"{report_label('다음 행동')} 검사 결과만으로 파킨슨병이라고 말할 수는 없습니다. 보호자나 보건소 담당자에게 이 결과를 보여주시고 신경과 전문의 상담을 받아보시는 것이 좋습니다."
        )
    return color, emphasize_symptoms(body)
 
# ── 앱 시작 ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ParkIN Soon · 박인순", layout="centered", page_icon="🧠")
st.markdown(CSS, unsafe_allow_html=True)
init_state()
init_customer_db()

try:
    m_olf, m_img, m_kin, weights = load_models(str(MODEL_DIR))
except Exception as e:
    st.error(f"모델 로드 실패: {e}\n모델 폴더: {MODEL_DIR}"); st.stop()

hdf = load_hospitals()
app_header()

# ══════════════════════════════════════════════════════════════════════════════
# Step 0: 인트로
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.step == 0:
    step_guide_card(
        "ParkIN Soon 조기 선별 검사",
        "후각, 배변, 손그림 세 가지 검사를 차례대로 진행합니다.<br>"
        "검사 결과는 진단이 아니라 참고용이며, 걱정되는 증상이 있으면 신경과 상담을 권합니다.",
    )

    st.markdown(f"""<div style="margin-top:20px"></div>
<div style="background:#fffbeb;border:1.5px solid {C_YELLOW};border-radius:14px;
            padding:16px 20px;margin-bottom:24px;font-size:20px;
            color:#78350f;line-height:1.8;">
    ⚠️ <strong>안내사항</strong><br>
    본 검사는 의학적 진단을 대체하지 않으며, 연구 및 참고용으로만 활용됩니다.<br>
    검사 결과가 걱정되신다면 반드시 전문의 상담을 받으시기 바랍니다.
</div>""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="park-soft-card" style="
            padding:22px 24px;margin-bottom:24px;">
    <h3 style="font-size:24px;font-weight:800;color:{C_DARK};margin:0 0 8px;">
        고객 정보 입력
    </h3>
    <p style="font-size:18px;color:#64748b;margin:0 0 16px;">
        검사 기록 저장을 위해 성함과 전화번호를 입력해주세요.
    </p>
</div>""", unsafe_allow_html=True)
    name_col, phone_col = st.columns(2)
    with name_col:
        customer_name = st.text_input("성함", value=st.session_state.customer_name, placeholder="홍길동")
    with phone_col:
        customer_phone = st.text_input("전화번호", value=st.session_state.customer_phone, placeholder="01012345678")

    history_rows = get_customer_history(customer_name, customer_phone)
    if history_rows:
        st.markdown(f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;
            padding:18px 20px;margin:10px 0 22px;">
    <div style="font-size:22px;font-weight:800;color:{C_DARK};margin-bottom:6px;">
        최근 검사 기록
    </div>
    <div style="font-size:18px;color:#64748b;">
        같은 성함과 전화번호로 저장된 가장 최근 결과입니다.
    </div>
</div>""", unsafe_allow_html=True)
        render_history_card(1, format_history_rows(history_rows[:1])[0])

    _, c2, _ = st.columns([0.5, 2, 0.5])
    with c2:
        if st.button("검사 시작하기 →", use_container_width=True, type="primary"):
            phone_digits = normalize_phone(customer_phone)
            if not customer_name.strip():
                st.warning("성함을 입력해주세요.")
            elif len(phone_digits) < 9:
                st.warning("전화번호를 정확히 입력해주세요.")
            else:
                st.session_state.customer_name = customer_name.strip()
                st.session_state.customer_phone = phone_digits
                st.session_state.customer_record_id = create_customer_record(customer_name, customer_phone)
                st.session_state.result_saved = False
                st.session_state.step = 1; rerun()
    st.markdown("<p style='text-align:center;color:#94a3b8;font-size:20px;margin-top:12px;'>소요 시간: 약 5~10분</p>",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: 후각
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 1:
    progress(1)
    step_guide_card(
        "Step 1. 후각 인지 테스트",
        "스크래치 카드의 향기를 맡아보고, 선명하게 구별이 되면 <strong>구별됨</strong>을 선택해주세요.<br>"
        "잘 모르겠으면 <strong>모르겠음</strong>을 선택하셔도 괜찮습니다.",
    )

    bsit = {
        "CHERRY": ("체리", "🍒"), "DILL_PICKLE": ("피클", "🥒"), "BANANA": ("바나나", "🍌"),
        "CHOCOLATE": ("초콜릿", "🍫"), "CINNAMON": ("계피", "🫚"), "GASOLINE": ("가솔린", "⛽"),
        "LEMON": ("레몬", "🍋"), "ONION": ("양파", "🧅"), "PINEAPPLE": ("파인애플", "🍍"),
        "ROSE": ("장미", "🌹"), "SOAP": ("비누", "🧼"), "SMOKE": ("연기", "💨"),
    }
    temp = []; keys = list(bsit.keys())
    for rs in range(0, len(keys), 3):
        for ci, col in enumerate(st.columns(3)):
            ki = rs + ci
            if ki >= len(keys): break
            lbl, icon = bsit[keys[ki]]
            with col:
                st.markdown(f"""
<div class="park-soft-card" style="
            padding:18px 12px 12px;text-align:center;margin-bottom:6px;">
    <div style="background:#f3f0ee;border-radius:12px;padding:24px 0;font-size:34px;margin-bottom:12px;">{icon}</div>
            <div style="font-size:23px;font-weight:900;color:{C_DARK};">{lbl}</div>
</div>""", unsafe_allow_html=True)
                ans = st.radio("구별?", ["○ 구별됨", "✗ 모르겠음"], horizontal=True,
                               key=f"olf_{keys[ki]}", label_visibility="collapsed")
                temp.append(1 if "○" in ans else 0)

    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
    if st.button("다음 단계 →  배변 상태 체크", use_container_width=True, type="primary"):
        st.session_state.olf_results = temp
        st.session_state.step = 2; rerun()

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: 배변
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    progress(2)
    step_guide_card(
        "Step 2. 배변 상태 체크",
        "최근 한 달 동안의 배변 불편에 대해 솔직하게 답변해주세요.<br>"
        "높은 점수는 불편이 크다는 뜻이며, 정확한 상태 파악에 도움이 됩니다.",
    )

    qs = [
        ("scopa_q5", "지난 한 달 동안 변비 증상을 겪은 적이 있었습니까?", "변비: 일주일에 두 번 이하로 대변을 보는 상태"),
        ("scopa_q6", "지난 한 달 동안 안간힘을 써서 대변을 본 적이 있었습니까?", "배변 시 과도하게 힘을 줘야 했던 경험"),
        ("scopa_q7", "지난 한 달 동안 자신의 의지와 상관없이 대변을 지린 적이 있었습니까?", "원하지 않는데 대변이 새어 나온 경우"),
    ]
    so = {0: "없음", 1: "가끔", 2: "자주", 3: "항상"}; res = []
    for i, (key, q, hint) in enumerate(qs):
        st.markdown(f"""
<div class="park-soft-card" style="
            padding:22px 24px 16px;margin-bottom:18px;
            ">
    <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:10px;">
        <div style="background:{C_ORANGE};color:white;border-radius:50%;
                    width:34px;height:34px;min-width:34px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:16px;font-weight:900;margin-top:2px;">{i + 1}</div>
        <div>
            <div style="font-size:23px;font-weight:800;color:{C_DARK};line-height:1.55;">{q}</div>
            <div style="font-size:22px;color:#64748b;margin-top:6px;">{hint}</div>
        </div>
    </div>
</div>""", unsafe_allow_html=True)
        val = st.radio("점수", [0, 1, 2, 3], format_func=lambda x: so[x],
                       horizontal=True, key=key, label_visibility="collapsed")
        res.append(val)

    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
    if st.button("다음 단계 →  나선 그리기", use_container_width=True, type="primary"):
        st.session_state.scopa_results = res
        st.session_state.step = 3; rerun()

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: 손그림
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    progress(3)
    step_guide_card(
        "Step 3. 나선 그리기 테스트",
        "손의 미세한 떨림과 운동 속도를 살펴봅니다.<br>"
        "오른쪽 캔버스의 점선 가이드를 따라 <strong>안쪽 중심에서 바깥쪽 방향</strong>으로 천천히 그려주세요.",
    )

    cl, cr = st.columns([1, 2])
    with cl:
        st.markdown("<div class='park-section-title'>📋 참고 가이드</div>", unsafe_allow_html=True)
        st.image(spiral_guide_img(), use_container_width=True)
    with cr:
        st.markdown("<div class='park-section-title'>✎ 여기에 그려주세요</div>", unsafe_allow_html=True)
        bg_img = spiral_img(w=CANVAS_SIZE, h=CANVAS_SIZE)
        cr_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=3,
            stroke_color=C_DARK,
            background_color="#FFFFFF",
            background_image=bg_img,
            height=CANVAS_SIZE,
            width=CANVAS_SIZE,
            drawing_mode="freedraw",
            key="spiral_canvas",
            display_toolbar=True
        )

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    if st.button("✅ 분석 결과 확인", use_container_width=True, type="primary"):
        if (cr_result.image_data is not None and cr_result.json_data is not None
                and cr_result.json_data.get("objects")):
            st.session_state.canvas_img = extract_drawing_only(cr_result.image_data)
            xa, ya, ta, ts = [], [], [], 0
            for obj in cr_result.json_data.get("objects", []):
                if obj.get("type") != "path": continue
                for cmd in obj.get("path", []):
                    if cmd[0] in ("M", "L", "Q"):
                        xa.append(float(cmd[-2]))
                        ya.append(float(cmd[-1]))
                        ta.append(ts); ts += 16
            st.session_state.canvas_json = {"x": xa, "y": ya, "t": ta} if len(xa) >= 10 else {}
            st.session_state.result_page = 1
            st.session_state.step = 4; rerun()
        else:
            st.warning("나선을 먼저 그려주세요.")

# ══════════════════════════════════════════════════════════════════════════════
# Step 4: 결과
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    import pandas as pd
    progress(4)

    # ── 예측 ──────────────────────────────────────────────────────────────────
    combined = st.session_state.olf_results + st.session_state.scopa_results
    olf_score = sum(st.session_state.olf_results)
    con_score = sum(st.session_state.scopa_results)
    p_olf = pred_proba(m_olf, combined)
    if st.session_state.canvas_img is None:
        st.error("손그림 이미지가 없습니다. 처음부터 다시 진행해주세요."); st.stop()
    p_img = float(np.asarray(m_img.predict(preprocess_canvas(st.session_state.canvas_img), verbose=0)).ravel()[0])
    p_kin = pred_proba(m_kin, get_kin_feats(st.session_state.get("canvas_json", {})))

    cnt = int(p_olf >= T_OLF) + int(p_img >= T_IMG) + int(p_kin >= T_KIN)

    # ── 종합 결과 변수 ────────────────────────────────────────────────────────
    if cnt == 0:
        ri = "✅"; rc = C_GREEN; rb = "#f0fdf4"; rbd = "#86efac"; rl = "low"
        rt = "특이 징후가 발견되지 않았습니다"
        rs = "세 가지 검사 항목 모두 정상 범위 이내입니다."
        ra = "정기적인 건강 검진을 꾸준히 받으시길 권장드립니다."
    elif cnt == 1:
        ri = "⚠️"; rc = C_ORANGE; rb = "#fff7ed"; rbd = "#fed7aa"; rl = "mid"
        rt = "일부 항목에서 주의가 필요합니다"
        rs = "한 가지 검사 항목에서 이상 징후가 감지되었습니다."
        ra = "전문의와 상담해 보시는 것을 권장드립니다."
    else:
        ri = "🔴"; rc = C_RED; rb = "#fef2f2"; rbd = "#fca5a5"; rl = "high"
        rt = "복수의 항목에서 이상 징후가 감지되었습니다"
        rs = f"{cnt}가지 검사 항목에서 동시에 이상 징후가 확인되었습니다."
        ra = "전문의의 소견을 받아보시길 권장드립니다."

    if not st.session_state.get("result_saved", False):
        update_customer_result(
            st.session_state.get("customer_record_id"),
            olf_score=olf_score,
            scopa_score=con_score,
            p_olf=p_olf,
            p_img=p_img,
            p_kin=p_kin,
            abnormal_count=cnt,
            risk_level=rl,
            olf_results=st.session_state.get("olf_results", []),
            scopa_results=st.session_state.get("scopa_results", []),
            canvas_json=st.session_state.get("canvas_json", {}),
        )
        st.session_state.result_saved = True

    # ── 종합 소견 카드 (항상 표시) ────────────────────────────────────────────
    st.markdown(f"""
<div style="background:{rb};border:2px solid {rbd};border-radius:20px;
            padding:32px 28px;margin-bottom:28px;text-align:center;">
    <div style="font-size:48px;margin-bottom:12px;">{ri}</div>
    <h2 style="font-size:24px;font-weight:800;color:{rc};margin:0 0 10px;line-height:1.4;">{rt}</h2>
    <p style="font-size:20px;color:#475569;margin:0 0 12px;">{rs}</p>
    <p style="font-size:20px;font-weight:600;color:{rc};margin:0;
              background:rgba(255,255,255,0.7);border-radius:10px;
              padding:10px 16px;display:inline-block;">{ra}</p>
</div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 결과 Page 1: 후각 + 변비
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.result_page == 1:

        # ── 후각 검사 결과 ────────────────────────────────────────────────────
        st.markdown(f"""
<div style="background:white;border:1.5px solid #d1fae5;border-radius:18px;
            padding:24px 28px;margin-bottom:20px;">
    <h3 style="font-size:30px;font-weight:800;color:{C_GREEN};margin:0 0 16px;">
        👃 후각 검사 결과
    </h3>""", unsafe_allow_html=True)

        olf_score = sum(st.session_state.olf_results)
        olf_pct   = olf_score / 12 * 100
        avg_normal_olf = 90.0

        fig_olf, ax_olf = plt.subplots(figsize=(8, 3))
        bar_colors = ['#93c5fd', C_RED if olf_pct < 75 else C_GREEN]
        bars = ax_olf.barh(['정상군 평균', '검사자'], [avg_normal_olf, olf_pct],
                           color=bar_colors, height=0.5)
        ax_olf.set_xlim(0, 105)
        ax_olf.set_xlabel('향기 인지율 (%)')
        ax_olf.axvline(75, color=C_ORANGE, linestyle='--', linewidth=1.5, label='주의 기준 (75%)')
        for bar, val in zip(bars, [avg_normal_olf, olf_pct]):
            ax_olf.text(val + 1, bar.get_y() + bar.get_height() / 2,
                        f'{val:.0f}%', va='center', fontsize=12, fontweight='bold')
        ax_olf.legend(fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_olf, clear_figure=True); plt.close()

        st.markdown(f"""<p style="font-size:20px;color:#475569;margin-top:8px;">
            12가지 향기 중 <strong>{olf_score}개</strong>를 정확히 인지하셨습니다.
            (인지율 {olf_pct:.0f}%)
        </p></div>""", unsafe_allow_html=True)
        olf_color, olf_body = olf_local_comment(olf_score)
        local_comment_card("박인순 한마디", olf_body, olf_color)

        # ── 변비 검사 결과 ────────────────────────────────────────────────────
        st.markdown(f"""
<div style="background:white;border:1.5px solid #fed7aa;border-radius:18px;
            padding:24px 28px;margin-bottom:20px;">
    <h3 style="font-size:30px;font-weight:800;color:{C_ORANGE};margin:0 0 16px;">
        🩺 배변 상태 검사 결과
    </h3>""", unsafe_allow_html=True)

        con_score = sum(st.session_state.scopa_results)
        avg_normal_con = 1.0

        fig_con, ax_con = plt.subplots(figsize=(8, 3))
        bar_colors2 = ['#93c5fd', C_RED if con_score >= 3 else C_GREEN]
        bars2 = ax_con.barh(['정상군 평균', '검사자'], [avg_normal_con, con_score],
                            color=bar_colors2, height=0.5)
        ax_con.set_xlim(0, 10)
        ax_con.set_xlabel('배변 불편 점수 (0~9점, 높을수록 불편 큼)')
        ax_con.axvline(3, color=C_ORANGE, linestyle='--', linewidth=1.5, label='주의 기준 (3점)')
        for bar, val in zip(bars2, [avg_normal_con, con_score]):
            ax_con.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                        f'{val:.0f}점', va='center', fontsize=12, fontweight='bold')
        ax_con.legend(fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_con, clear_figure=True); plt.close()

        st.markdown(f"""<p style="font-size:20px;color:#475569;margin-top:8px;">
            배변 불편 점수: <strong>{con_score}점</strong> / 9점
            <span style="color:#ef4444;font-weight:700;">(높을수록 불편이 큽니다)</span>
        </p></div>""", unsafe_allow_html=True)
        bowel_color, bowel_body = bowel_local_comment(con_score)
        local_comment_card("박인순 한마디", bowel_body, bowel_color)

        # ── 다음 버튼 ─────────────────────────────────────────────────────────
        _, c2, _ = st.columns([1, 2, 1])
        with c2:
            if st.button("손그림 분석 결과 보기 →", use_container_width=True, type="primary"):
                st.session_state.result_page = 2; rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # 결과 Page 2: 손그림 Grad-CAM + 운동학
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.result_page == 2:

        # ── 손그림 Grad-CAM ───────────────────────────────────────────────────
        st.markdown(f"""
<div style="background:white;border:1.5px solid #fce7f3;border-radius:18px;
            padding:28px 28px;margin-bottom:20px;">
    <h3 style="font-size:30px;font-weight:800;color:#be185d;margin:0 0 16px;">
        ✍️ 손그림 분석 결과
    </h3>""", unsafe_allow_html=True)

        img_input = preprocess_canvas(st.session_state.canvas_img)
        try:
            gradcam_img = make_gradcam(img_input, m_img)
            col_orig, col_cam = st.columns(2)
            with col_orig:
                st.markdown("**원본 나선 그림**")
                orig_np = (img_input[0] * 255).astype(np.uint8)
                st.image(orig_np, width=350)
            with col_cam:
                st.markdown("**AI 주목 영역 (Grad-CAM)**")
                st.image(gradcam_img[:, :, ::-1], width=350)

            badge_img = "⚠️ 이상 감지" if p_img >= T_IMG else "✅ 정상 범위"
            color_img = C_RED if p_img >= T_IMG else C_GREEN
            st.markdown(f"""
<p style="font-size:20px;color:#475569;margin-top:8px;">
    붉은 영역일수록 모델이 이상 패턴으로 주목한 부분입니다.
    &nbsp;<span style="background:{color_img};color:white;
    border-radius:12px;padding:3px 10px;font-size:20px;">{badge_img}</span>
</p>""", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Grad-CAM 생성 실패: {e}")
            
        if p_img < T_IMG * 0.6:
            shape_comment = "나선 형태가 전반적으로 안정적이며 이상 패턴이 관찰되지 않았습니다."
        elif p_img < T_IMG:
            shape_comment = "나선 형태는 정상 범위이나 일부 불규칙한 구간이 감지되었습니다."
        elif p_img < T_IMG * 1.3:
            shape_comment = "나선 형태에서 경미한 이상 패턴이 감지되었습니다. 전문의 확인을 권장합니다."
        else:
            shape_comment = "나선 형태에서 뚜렷한 이상 패턴이 감지되었습니다. 전문의 상담을 권장합니다."

        color_img = C_RED if p_img >= T_IMG else C_GREEN
        st.markdown(f"""
        <div style="background:{color_img};border-radius:12px;
                    padding:14px 20px;margin-top:12px;">
            <span style="color:white;font-size:22px;font-weight:700;">
                🖊️ 형태 분석: {shape_comment}
            </span>
        </div>""", unsafe_allow_html=True)
        
        # ── 운동학 수치 ───────────────────────────────────────────────────────
        kin_feats = get_kin_feats(st.session_state.get("canvas_json", {}))
        if kin_feats and any(v != 0.5 for v in kin_feats):
            feat_dict = dict(zip(FEATURE_COLS, kin_feats))
            st.markdown(f"""
<div style="background:#f8fafc;border-radius:12px;padding:16px 20px;margin-top:12px;">
    <div style="font-size:20px;font-weight:700;color:#334155;margin-bottom:10px;">
        📊 손그림 움직임 수치 (참고용)
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;font-size:20px;color:#475569;">
        <div>평균 속도<br><strong>{feat_dict['velocity_mean']:.1f} px/s</strong></div>
        <div>속도 변동성 (떨림)<br><strong>{feat_dict['velocity_cv']:.3f}</strong></div>
        <div>평균 가속도<br><strong>{feat_dict['acceleration_mean']:.1f}</strong></div>
        <div>최대 속도<br><strong>{feat_dict['velocity_max']:.1f} px/s</strong></div>
        <div>최대 가속도<br><strong>{feat_dict['acceleration_max']:.1f}</strong></div>
        <div>총 이동거리<br><strong>{feat_dict['total_distance']:.0f} px</strong></div>
    </div>
    
    """, unsafe_allow_html=True)

            vel_cv   = feat_dict['velocity_cv']
            vel_mean = feat_dict['velocity_mean']
            if vel_cv < 0.4:
                kin_comment = "속도가 전반적으로 일정하게 유지되었습니다."
            elif vel_cv < 0.7:
                kin_comment = "속도 변동이 다소 관찰되었습니다."
            else:
                kin_comment = "속도 변동이 크게 나타났습니다. 떨림 가능성이 있습니다."

            if vel_mean < 50:
                kin_comment += " 전반적인 운동 속도가 느린 편입니다."
            elif vel_mean > 300:
                kin_comment += " 전반적인 운동 속도는 빠른 편입니다."

            st.markdown(f"""
<div style="background:#f0fdf4;border-left:4px solid {C_GREEN};
            border-radius:8px;padding:12px 16px;margin-top:10px;
            font-size:20px;color:#334155;">
    📐 <strong>움직임 평가:</strong> {kin_comment}
</div>
    <div style="font-size:20px;color:#94a3b8;margin-top:10px;">
        ※ 웹 캔버스 환경 특성상 타임스탬프가 근사값이므로 참고용으로만 활용하세요.
    </div>
</div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        if "kin_comment" not in locals():
            kin_comment = "움직임 수치는 충분히 계산되지 않았습니다."
        draw_color, draw_body = drawing_local_comment(p_img, p_kin, shape_comment, kin_comment)
        local_comment_card("박인순 한마디", draw_body, draw_color)

        # ── 판정 기준 안내 ────────────────────────────────────────────────────
        st.markdown(f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
            padding:16px 20px;margin-top:8px;margin-bottom:24px;
            font-size:20px;color:#64748b;line-height:1.8;">
    <strong style="color:#334155;">판정 기준 안내</strong><br>
    각 항목의 임계값은 해당 데이터셋에서 Youden's Index로 산출된 최적값입니다.<br>
    이상 감지된 항목 수를 기준으로 종합 결과를 도출합니다.<br>
    &nbsp;• 0개 이상 → 특이 징후 없음 &nbsp;
    &nbsp;• 1개 이상 → 주의 필요 &nbsp;
    &nbsp;• 2개 이상 → 전문의 소견 권유
</div>""", unsafe_allow_html=True)

        # ── 신경과 찾기 (고위험만) ────────────────────────────────────────────
        if rl == "high":
            st.markdown(f"<h3 style='font-size:20px;font-weight:700;color:{C_RED};margin-bottom:16px;'>🏥 가까운 신경과 찾기</h3>",
                        unsafe_allow_html=True)
            if hdf.empty:
                st.warning("병원 데이터가 없습니다. hospital_data.xlsx 또는 modeling/hospital_data.xlsx를 확인해주세요.")
            else:
                addr = st.text_input("현재 보건소 주소를 입력해주세요", placeholder="예: 서울특별시 강남구")
                if addr:
                    pts = addr.strip().split()
                    sido = pts[0][:2] if pts else ""; sgg = pts[1][:3] if len(pts) > 1 else ""
                    nb = hdf[hdf['주소'].str.contains(sido, na=False) & hdf['주소'].str.contains(sgg, na=False)].head(10)
                    if nb.empty: nb = hdf[hdf['주소'].str.contains(sido, na=False)].head(10)
                    if not nb.empty:
                        # 지도 중심 좌표
                        center_lat = nb['좌표(Y)'].astype(float).mean()
                        center_lon = nb['좌표(X)'].astype(float).mean()
                        
                        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
                        
                        for i, (_, row) in enumerate(nb.iterrows(), 1):
                            ph = row['전화번호'] if pd.notna(row['전화번호']) else '번호 없음'
                            lat = float(row['좌표(Y)'])
                            lon = float(row['좌표(X)'])
                            
                            # 번호 달린 마커
                            folium.Marker(
                                location=[lat, lon],
                                popup=folium.Popup(
                                    f"<b>{i}. {row['요양기관명']}</b><br>{row['주소']}<br>☎ {ph}",
                                    max_width=250
                                ),
                                tooltip=f"{i}. {row['요양기관명']}",
                                icon=folium.DivIcon(
                                    html=f"""
                                    <div style="background:{C_RED};color:white;
                                                border-radius:50%;width:28px;height:28px;
                                                display:flex;align-items:center;justify-content:center;
                                                font-size:18px;font-weight:700;
                                                border:2px solid white;
                                                box-shadow:0 2px 6px rgba(0,0,0,0.3);">
                                        {i}
                                    </div>""",
                                    icon_size=(28, 28),
                                    icon_anchor=(14, 14)
                                )
                            ).add_to(m)
                        
                        st_folium(m, width="100%", height=400)
                        
                        # 병원 목록
                        for i, (_, row) in enumerate(nb.iterrows(), 1):
                            ph = row['전화번호'] if pd.notna(row['전화번호']) else '번호 없음'
                            st.markdown(f"""
                    <div style="background:white;border:1px solid #fecaca;border-radius:10px;
                                padding:12px 16px;margin-bottom:8px;font-size:18px;
                                display:flex;align-items:center;gap:12px;">
                        <div style="background:{C_RED};color:white;border-radius:50%;
                                    width:28px;height:28px;min-width:28px;
                                    display:flex;align-items:center;justify-content:center;
                                    font-size:13px;font-weight:700;">{i}</div>
                        <div>
                            <strong>{row['요양기관명']}</strong><br>
                            <span style="font-size:18px;color:#64748b;">{row['주소']} &nbsp; ☎ {ph}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
                    else:
                        st.warning("해당 지역에서 신경과를 찾을 수 없습니다.")

        # ── 면책 고지 ─────────────────────────────────────────────────────────
        st.markdown("""
<div style="background:#f1f5f9;border-radius:12px;padding:14px 18px;margin-top:16px;
            font-size:20px;color:#64748b;line-height:1.7;text-align:center;">
    ⚕️ 본 결과는 연구 및 프로토타입용 참고 지표이며, 의학적 진단을 대체하지 않습니다.<br>
    검사 결과와 관계없이 건강에 이상이 느껴지시면 전문의와 상담하시기 바랍니다.
</div>""", unsafe_allow_html=True)


        # ── 이전/최종/처음 버튼 ───────────────────────────────────────────────
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        col_prev, col_report, col_home = st.columns([1, 1.4, 1])
        with col_prev:
            if st.button("← 후각/변비 결과 보기", use_container_width=True):
                st.session_state.result_page = 1; rerun()
        with col_report:
            if st.button("최종 레포트 확인하기 →", use_container_width=True, type="primary"):
                st.session_state.result_page = 3; rerun()
        with col_home:
            if st.button("🔄 처음으로 돌아가기", use_container_width=True):
                st.session_state.clear(); rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # 결과 Page 3: 최종 레포트 + Q&A
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.result_page == 3:
        st.markdown(f"""
<div style="background:white;border:1.5px solid #dbeafe;border-radius:20px;
            padding:28px 30px;margin-bottom:24px;">
    <h2 style="font-size:32px;font-weight:900;color:{C_DARK};margin:0 0 8px;">
        최종 레포트
    </h2>
    <p style="font-size:20px;color:#64748b;margin:0;line-height:1.7;">
        후각, 배변, 손그림 결과를 한 화면에 모았습니다.
        아래 내용은 참고용이며, 정확한 진단은 신경과 전문의 상담이 필요합니다.
    </p>
</div>""", unsafe_allow_html=True)

        st.markdown(f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:16px;
            padding:18px 20px;margin-bottom:24px;">
    <div style="font-size:24px;font-weight:900;color:{C_DARK};margin-bottom:14px;">요약</div>
    <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;">
        <div style="background:white;border:1.5px solid #dbeafe;border-radius:14px;padding:16px;">
            <div style="font-size:18px;color:#64748b;margin-bottom:6px;">1. 후각+배변</div>
            <div style="font-size:28px;font-weight:900;color:{signal_color(p_olf, T_OLF)};">{p_olf * 100:.1f}%</div>
            <div style="font-size:17px;color:#475569;margin-top:8px;">후각 {olf_score}/12 · 배변 {con_score}/9</div>
        </div>
        <div style="background:white;border:1.5px solid #dbeafe;border-radius:14px;padding:16px;">
            <div style="font-size:18px;color:#64748b;margin-bottom:6px;">2. 그림 모양</div>
            <div style="font-size:28px;font-weight:900;color:{signal_color(p_img, T_IMG)};">{p_img * 100:.1f}%</div>
            <div style="font-size:17px;color:#475569;margin-top:8px;">AI가 나선 모양에서 본 신호</div>
        </div>
        <div style="background:white;border:1.5px solid #dbeafe;border-radius:14px;padding:16px;">
            <div style="font-size:18px;color:#64748b;margin-bottom:6px;">3. 손 움직임</div>
            <div style="font-size:28px;font-weight:900;color:{signal_color(p_kin, T_KIN)};">{p_kin * 100:.1f}%</div>
            <div style="font-size:17px;color:#475569;margin-top:8px;">이상 감지 {cnt}/3</div>
        </div>
    </div>
    <div style="font-size:17px;color:#64748b;margin-top:12px;">
        퍼센트는 모델이 주의 신호를 감지한 정도입니다. 진단 확률은 아닙니다.
    </div>
</div>""", unsafe_allow_html=True)

        st.markdown("## 1. 후각 결과")
        olf_pct = olf_score / 12 * 100
        fig_olf_r, ax_olf_r = plt.subplots(figsize=(8.5, 3.2))
        ax_olf_r.barh(["정상군 평균", "검사자"], [90.0, olf_pct],
                      color=["#93c5fd", C_RED if olf_pct < 75 else C_GREEN], height=0.5)
        ax_olf_r.set_xlim(0, 105)
        ax_olf_r.axvline(75, color=C_ORANGE, linestyle="--", linewidth=1.5)
        ax_olf_r.set_xlabel("향기 인지율 (%)")
        plt.tight_layout()
        st.pyplot(fig_olf_r, clear_figure=True); plt.close()
        olf_color, olf_body = olf_local_comment(olf_score)
        local_comment_card("후각 코멘트", olf_body, olf_color)

        st.markdown("## 2. 배변 결과")
        fig_con_r, ax_con_r = plt.subplots(figsize=(8.5, 3.2))
        fig_con_r.subplots_adjust(left=0.22)
        ax_con_r.barh(["정상군 평균", "검사자"], [1.0, con_score],
                      color=["#93c5fd", C_RED if con_score >= 3 else C_GREEN], height=0.5)
        ax_con_r.set_xlim(0, 10)
        ax_con_r.axvline(3, color=C_ORANGE, linestyle="--", linewidth=1.5)
        ax_con_r.set_xlabel("배변 불편 점수 (0~9점, 높을수록 불편 큼)")
        plt.tight_layout()
        st.pyplot(fig_con_r, clear_figure=True); plt.close()
        bowel_color, bowel_body = bowel_local_comment(con_score)
        local_comment_card("배변 코멘트", bowel_body, bowel_color)

        st.markdown("## 3. 손그림 결과")
        img_input = preprocess_canvas(st.session_state.canvas_img)
        try:
            gradcam_img = make_gradcam(img_input, m_img)
            st.markdown("**원본 나선 그림**")
            st.image((img_input[0] * 255).astype(np.uint8), width=360)
            st.markdown("**AI가 중요하게 본 부분**")
            st.image(gradcam_img[:, :, ::-1], width=360)
            st.caption("색이 진한 부분은 AI가 그림에서 중요하게 본 부분입니다.")
        except Exception as e:
            st.warning(f"Grad-CAM 생성 실패: {e}")

        kin_feats_report = get_kin_feats(st.session_state.get("canvas_json", {}))
        kin_comment_report = "움직임 수치는 충분히 계산되지 않았습니다."
        if kin_feats_report and any(v != 0.5 for v in kin_feats_report):
            feat_report = dict(zip(FEATURE_COLS, kin_feats_report))
            vel_cv = feat_report["velocity_cv"]
            vel_mean = feat_report["velocity_mean"]
            if vel_cv < 0.4:
                kin_comment_report = "속도가 전반적으로 일정하게 유지되었습니다."
            elif vel_cv < 0.7:
                kin_comment_report = "속도 변동이 다소 관찰되었습니다."
            else:
                kin_comment_report = "속도 변동이 크게 나타났습니다. 떨림 가능성이 있습니다."
            if vel_mean < 50:
                kin_comment_report += " 전반적인 운동 속도가 느린 편입니다."
            elif vel_mean > 300:
                kin_comment_report += " 전반적인 운동 속도는 빠른 편입니다."

            st.markdown(f"""
<div style="background:#f8fafc;border-radius:12px;padding:14px 16px;margin-top:12px;
            font-size:20px;color:#334155;line-height:1.8;">
    평균 속도: <strong>{feat_report['velocity_mean']:.1f} px/s</strong><br>
    속도 변동성: <strong>{feat_report['velocity_cv']:.3f}</strong><br>
    총 이동거리: <strong>{feat_report['total_distance']:.0f} px</strong>
</div>""", unsafe_allow_html=True)

        if p_img < T_IMG * 0.6:
            shape_report = "나선 형태가 전반적으로 안정적이며 이상 패턴이 관찰되지 않았습니다."
        elif p_img < T_IMG:
            shape_report = "나선 형태는 정상 범위이나 일부 불규칙한 구간이 감지되었습니다."
        elif p_img < T_IMG * 1.3:
            shape_report = "나선 형태에서 경미한 이상 패턴이 감지되었습니다."
        else:
            shape_report = "나선 형태에서 뚜렷한 이상 패턴이 감지되었습니다."
        draw_color, draw_body = drawing_local_comment(p_img, p_kin, shape_report, kin_comment_report)
        local_comment_card("손그림 코멘트", draw_body, draw_color)

        st.markdown(f"""
<div style="background:{rb};border:2px solid {rbd};border-radius:18px;
            padding:20px 22px;margin:24px 0;text-align:center;">
    <div style="font-size:28px;margin-bottom:8px;">{ri}</div>
    <div style="font-size:24px;font-weight:900;color:{rc};margin-bottom:8px;">{rt}</div>
    <div style="font-size:20px;color:#475569;line-height:1.7;">{rs}<br>{ra}</div>
</div>""", unsafe_allow_html=True)

        final_color, final_body = final_local_comment(cnt, olf_score, con_score, p_olf, p_img, p_kin)
        local_comment_card("박인순의 최종 코멘트", final_body, final_color)
        previous_result = get_previous_customer_result(
            st.session_state.get("customer_name", ""),
            st.session_state.get("customer_phone", ""),
            st.session_state.get("customer_record_id"),
        )
        render_previous_comparison(
            previous_result,
            olf_score=olf_score,
            con_score=con_score,
            p_olf=p_olf,
            p_img=p_img,
            p_kin=p_kin,
        )

        col_prev, col_home = st.columns(2)
        with col_prev:
            if st.button("← 손그림 결과로 돌아가기", use_container_width=True):
                st.session_state.result_page = 2; rerun()
        with col_home:
            if st.button("🔄 처음으로 돌아가기", use_container_width=True):
                st.session_state.clear(); rerun()

# ── 전역 사이드바 챗봇(AI 박인순) 모듈 호출 ──────────────────────────────
import ParkinsoonAI
tr = {
    "step": st.session_state.get("step", 0),
    "olf_score": sum(st.session_state.get("olf_results", [])) if st.session_state.get("olf_results") else None,
    "const_score": sum(st.session_state.get("scopa_results", [])) if st.session_state.get("scopa_results") else None,
    "olf_results": st.session_state.get("olf_results", []),
    "scopa_results": st.session_state.get("scopa_results", []),
    "customer_name": st.session_state.get("customer_name", ""),
    "customer_phone": st.session_state.get("customer_phone", ""),
    "p_olf": locals().get('p_olf', None),
    "img_prob": locals().get('p_img', None),
    "p_img": locals().get('p_img', None),
    "kin_prob": locals().get('p_kin', None),
    "p_kin": locals().get('p_kin', None),
    "kin_feats": locals().get('kin_feats', None),
    "result_page": st.session_state.get("result_page", 1),
    "final_risk": (locals().get('cnt', 0) / 3.0) if st.session_state.get("step") == 4 else None
}
ParkinsoonAI.render_parkinson_chatbot(tr)
