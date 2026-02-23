import math

import pandas as pd

# Angle range constants (tune once model is finalized)
ELBOW_MIN = 80
ELBOW_MAX = 110
SHOULDER_MIN = 90
SHOULDER_MAX = 130
KNEE_MIN = 140
KNEE_MAX = 170
HIP_MIN = 150
HIP_MAX = 175

# Penalty points per angle when outside range (optional to tune)
PENALTY_ELBOW_TUCKED = 15
PENALTY_ELBOW_EXTENDED = 10
PENALTY_SHOULDER = 10
PENALTY_KNEE = 10
PENALTY_HIP = 10


def _valid_angle(val) -> bool:
    return val is not None and not math.isnan(val)


def score_shot(stats: dict) -> dict:
    score = 100
    strengths = []
    weaknesses = []

    elbow = stats.get("elbow_angle")
    if _valid_angle(elbow):
        if ELBOW_MIN <= elbow <= ELBOW_MAX:
            strengths.append("Good elbow angle at release")
        elif elbow < ELBOW_MIN:
            score -= PENALTY_ELBOW_TUCKED
            weaknesses.append("Elbow too tucked — try keeping it at 90°")
        else:
            score -= PENALTY_ELBOW_EXTENDED
            weaknesses.append("Elbow too extended — chicken wing form")

    shoulder = stats.get("shoulder_angle")
    if _valid_angle(shoulder):
        if SHOULDER_MIN <= shoulder <= SHOULDER_MAX:
            strengths.append("Good shoulder angle (arm raised toward basket)")
        elif shoulder < SHOULDER_MIN:
            score -= PENALTY_SHOULDER
            weaknesses.append("Shoulder angle too low — raise arm toward basket")
        else:
            score -= PENALTY_SHOULDER
            weaknesses.append("Shoulder over-extended — keep release natural")

    knee = stats.get("knee_angle")
    if _valid_angle(knee):
        if KNEE_MIN <= knee <= KNEE_MAX:
            strengths.append("Good knee extension at follow-through")
        elif knee < KNEE_MIN:
            score -= PENALTY_KNEE
            weaknesses.append("Knees too bent — extend into follow-through")
        else:
            score -= PENALTY_KNEE
            weaknesses.append("Knees over-extended — slight bend is OK")

    hip = stats.get("hip_angle")
    if _valid_angle(hip):
        if HIP_MIN <= hip <= HIP_MAX:
            strengths.append("Good hip posture (upright)")
        elif hip < HIP_MIN:
            score -= PENALTY_HIP
            weaknesses.append("Hips too bent — stand more upright at release")
        else:
            score -= PENALTY_HIP
            weaknesses.append("Hip angle too open — check posture")

    return {
        "score": max(0, score),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "metadata": stats,
    }

def main():
    dummy_stats = {"elbow_angle":30}
    result = score_shot(dummy_stats)
    print(result)

if __name__ == "__main__":
    main()