import json
from pathlib import Path
from typing import Dict, Optional, Tuple


APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_DIR / "fusion_config.json"
REQUIRED_MODELS = ("olf", "img", "kin")


def load_fusion_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict:
    """검증 AUC 기반 late-fusion 설정을 읽고 기본 무결성을 확인합니다."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = json.load(file)

    if config.get("method") not in {"auc_weighted_average", "reliability_uncertainty_dynamic"}:
        raise ValueError("지원하지 않는 late-fusion 방식입니다.")

    weights = config.get("weights", {})
    if any(name not in weights for name in REQUIRED_MODELS):
        raise ValueError("fusion_config.json에 필요한 모델 가중치가 없습니다.")
    if any(float(weights[name]) < 0 for name in REQUIRED_MODELS):
        raise ValueError("late-fusion 가중치는 음수일 수 없습니다.")
    if sum(float(weights[name]) for name in REQUIRED_MODELS) <= 0:
        raise ValueError("late-fusion 가중치 합은 0보다 커야 합니다.")
    return config


def load_stage_thresholds(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, float]:
    """Load the validated deployment threshold for each screening modality."""
    config = load_fusion_config(config_path)
    thresholds = config.get("thresholds", {})
    missing = [name for name in REQUIRED_MODELS if name not in thresholds]
    if missing:
        raise ValueError(f"fusion_config.json에 단계 임계값이 없습니다: {missing}")
    parsed = {name: float(thresholds[name]) for name in REQUIRED_MODELS}
    if any(not 0.0 < value < 1.0 for value in parsed.values()):
        raise ValueError("단계 임계값은 0과 1 사이여야 합니다.")
    return parsed


def calculate_fusion_score(
    p_olf: float,
    p_img: float,
    p_kin: float,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Tuple[float, Dict[str, float]]:
    """세 모델 출력을 AUC 가중 평균한 통합 주의 점수로 계산합니다.

    반환값은 질병 확률로 보정된 값이 아니라 서로 다른 선별 모델의 상대적인
    주의 신호를 하나로 요약한 점수입니다.
    """
    probabilities = {"olf": float(p_olf), "img": float(p_img), "kin": float(p_kin)}
    if any(not 0.0 <= value <= 1.0 for value in probabilities.values()):
        raise ValueError("각 모델 점수는 0과 1 사이여야 합니다.")

    config = load_fusion_config(config_path)
    raw_weights = {name: float(config["weights"][name]) for name in REQUIRED_MODELS}
    weight_sum = sum(raw_weights.values())
    weights = {name: value / weight_sum for name, value in raw_weights.items()}
    score = sum(probabilities[name] * weights[name] for name in REQUIRED_MODELS)
    return float(score), weights


def _threshold_distance(probability: float, threshold: float) -> float:
    """판정 경계로부터의 거리를 0~1로 정규화합니다."""
    denominator = max(threshold, 1.0 - threshold, 1e-6)
    return min(abs(probability - threshold) / denominator, 1.0)


def calculate_dynamic_fusion_score(
    p_olf: Optional[float],
    p_img: Optional[float],
    p_kin: Optional[float],
    quality_scores: Optional[Dict[str, float]] = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Tuple[float, Dict[str, float], Dict[str, Dict[str, float]]]:
    """입력 품질과 예측 불확실도를 반영한 규칙 기반 동적 late fusion입니다.

    결합 코호트에서 학습된 모델이 아니므로 결과는 질병 확률이 아니라 위험 신호
    선별 보조 점수입니다. 누락되거나 품질이 0인 modality는 자동 제외합니다.
    """
    config = load_fusion_config(config_path)
    probabilities = {"olf": p_olf, "img": p_img, "kin": p_kin}
    qualities = quality_scores or {}
    auc_scores = config.get("auc_scores", {})
    thresholds = config.get("thresholds", {})
    raw_weights: Dict[str, float] = {}
    details: Dict[str, Dict[str, float]] = {}

    for name in REQUIRED_MODELS:
        probability = probabilities[name]
        quality = min(max(float(qualities.get(name, 1.0)), 0.0), 1.0)
        if probability is None or quality == 0.0:
            raw_weights[name] = 0.0
            details[name] = {"quality": quality, "reliability": 0.0, "confidence": 0.0}
            continue
        probability = float(probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} 모델 점수는 0과 1 사이여야 합니다.")
        threshold = float(thresholds.get(name, 0.5))
        reliability = max(float(auc_scores.get(name, 0.5)) - 0.5, 0.01)
        confidence = _threshold_distance(probability, threshold)
        uncertainty_factor = 0.5 + 0.5 * confidence
        raw_weights[name] = reliability * quality * uncertainty_factor
        details[name] = {
            "quality": quality,
            "reliability": reliability,
            "confidence": confidence,
            "uncertainty_factor": uncertainty_factor,
        }

    total_weight = sum(raw_weights.values())
    if total_weight <= 0.0:
        raise ValueError("사용 가능한 검사 결과가 없어 통합 점수를 계산할 수 없습니다.")
    weights = {name: raw_weights[name] / total_weight for name in REQUIRED_MODELS}
    score = sum(float(probabilities[name]) * weights[name] for name in REQUIRED_MODELS if probabilities[name] is not None)
    return float(score), weights, details


def majority_vote_stage(
    probabilities: Dict[str, Optional[float]],
    thresholds: Dict[str, float],
    quality_scores: Optional[Dict[str, float]] = None,
    minimum_quality: float = 0.35,
) -> Dict:
    """검사별 threshold 투표로 위험 신호 단계를 정합니다.

    품질이 너무 낮거나 값이 없는 검사는 기권 처리합니다. 세 검사 중 두 검사
    이상이 유효하지 않으면 안전하게 재검사 단계로 보류합니다.
    """
    qualities = quality_scores or {}
    votes: Dict[str, Optional[int]] = {}
    for name in REQUIRED_MODELS:
        probability = probabilities.get(name)
        quality = float(qualities.get(name, 1.0))
        if probability is None or quality < minimum_quality:
            votes[name] = None
        else:
            votes[name] = int(float(probability) >= float(thresholds[name]))

    valid_votes = [vote for vote in votes.values() if vote is not None]
    positive_count = sum(valid_votes)
    valid_count = len(valid_votes)

    if valid_count < 2:
        stage = "retest"
        label = "검사 품질 확인 필요"
    elif positive_count == 0:
        stage = "low"
        label = "낮은 주의 단계"
    elif positive_count == 1:
        stage = "observe"
        label = "단일 위험 신호 관찰"
    elif positive_count == 2:
        stage = "caution"
        label = "다수결 위험 신호 주의"
    else:
        stage = "high_caution"
        label = "세 검사 모두 위험 신호"

    return {
        "stage": stage,
        "label": label,
        "votes": votes,
        "positive_count": int(positive_count),
        "valid_count": int(valid_count),
        "majority_positive": bool(valid_count >= 2 and positive_count >= 2),
    }
