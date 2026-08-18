"""
Safe-Logic 2.0 — L3 결정 로직 (N-of-M 투표 + 히스테리시스 + 채널 다중화)
PSIntelligence Step 5.5 (2026-08-18 결정 D-A)

3-Layer FDIR:
  L0: 물리 경계 (하드 바운드)   -> 단발 플래그로 즉시 abort (결정적)
  L1: 필터 일관성 (혁신 χ²)     -> 단발 플래그로 즉시 abort (결정적, exact)
  L2: ML anomaly (OC-SVM + CP)  -> N-of-M 투표 (기본 N=3, M=5)

히스테리시스: abort 래치 후 해제는 M hysteresis_clear(=5) 연속 전 채널 정상 필요.
abort는 되돌릴 수 없는 결정(낙하산/중단)이므로 래치가 기본 동작.

채널 플래그 생성은 외부(채널별 스코어 함수) 책임 — 이 모듈은 결정 로직만 담당.
CP 보장과의 정렬: N-of-M 규칙 전체를 하나의 conformal 객체로 calibration하는
방식은 [[conformal-safety-monitoring]] novelty 1위(비행 단위 FPR)와 정렬.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List


@dataclass
class Decision:
    """단일 스텝의 L3 결정 출력."""
    action: str                # "abort" | "watch" | "safe"
    state: str                 # "normal" | "aborted"
    l2_votes: int              # 현재 M-윈도우 내 L2 플래그 수
    normal_streak: int         # abort 후 연속 정상 스텝 수 (히스테리시스)
    channels: Dict[str, bool] = field(default_factory=dict)  # 입력 플래그 기록
    m: int = 5                 # N-of-M 윈도우 크기 (표시용)

    def __str__(self) -> str:
        return (f"Decision(action={self.action}, state={self.state}, "
                f"l2_votes={self.l2_votes}/{self.m}, "
                f"normal_streak={self.normal_streak})")


class SafeLogic:
    """L3 결정 로직: L0/L1 즉시 abort + L2 N-of-M + 히스테리시스 래치."""

    def __init__(self, n: int = 3, m: int = 5, hysteresis_clear: int = 5):
        if not (1 <= n <= m):
            raise ValueError(f"invalid N-of-M: n={n}, m={m} (require 1<=n<=m)")
        self.n = n
        self.m = m
        self.hysteresis_clear = hysteresis_clear
        self._l2_buffer: Deque[bool] = deque(maxlen=m)
        self._aborted: bool = False
        self._normal_streak: int = 0
        self._history: List[Decision] = []

    # ------------------------------------------------------------------
    # 상태
    # ------------------------------------------------------------------
    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def normal_streak(self) -> int:
        return self._normal_streak

    @property
    def l2_votes(self) -> int:
        return sum(self._l2_buffer)

    @property
    def history(self) -> List[Decision]:
        return list(self._history)

    def reset(self) -> None:
        self._l2_buffer.clear()
        self._aborted = False
        self._normal_streak = 0
        self._history.clear()

    # ------------------------------------------------------------------
    # 결정
    # ------------------------------------------------------------------
    def decide(self, l0: bool = False, l1: bool = False, l2: bool = False) -> Decision:
        """한 스텝의 채널 플래그를 받아 결정을 반환한다.

        Parameters
        ----------
        l0 : 물리 경계 위반 (하드 바운드)
        l1 : 혁신 χ² 검정 실패 (결정적)
        l2 : ML 이상 플래그 (OC-SVM + CP 스코어 > 임계)
        """
        channels = {"l0": bool(l0), "l1": bool(l1), "l2": bool(l2)}

        # L0/L1: 결정적 채널 — 즉시 abort (래치)
        if l0 or l1:
            self._aborted = True
            self._normal_streak = 0
            action = "abort"
        elif self._aborted:
            # 히스테리시스: 해제 조건 = hysteresis_clear 연속 전 채널 정상
            if not l2:
                self._normal_streak += 1
                if self._normal_streak >= self.hysteresis_clear:
                    self._aborted = False
                    self._normal_streak = 0
                    self._l2_buffer.clear()  # 이전 플래그 잔존으로 재-abort 방지
                    action = "safe"
                else:
                    action = "abort"  # 아직 래치 유지
            else:
                self._normal_streak = 0
                action = "abort"
        else:
            # L2: N-of-M 투표
            self._l2_buffer.append(l2)
            votes = self.l2_votes
            if votes >= self.n:
                self._aborted = True
                self._normal_streak = 0
                action = "abort"
            elif votes > 0:
                action = "watch"     # 플래그 있으나 아직 임계 미만
            else:
                action = "safe"

        dec = Decision(
            action=action,
            state="aborted" if self._aborted else "normal",
            l2_votes=self.l2_votes,
            normal_streak=self._normal_streak,
            channels=channels,
            m=self.m,
        )
        self._history.append(dec)
        return dec


# ----------------------------------------------------------------------
# 채널 플래그 생성 헬퍼 (L0/L1/L2 스코어 -> boolean)
# ----------------------------------------------------------------------
def l0_physical_bound(sensor_value: float, min_bound: float, max_bound: float) -> bool:
    """L0: 물리 경계 검사 — 결정적 하드 바운드."""
    return not (min_bound <= sensor_value <= max_bound)


def l1_chi2_flag(chi2_stat: float, dof: int, alpha: float = 0.01) -> bool:
    """L1: 혁신 χ² 검정 — 결정적 exact 보장.

    H0: 혁신이 영평균 백색 가우시안 -> χ²(dof) 분포.
    임계 초과 시 필터 일관성 위반으로 abort.
    """
    from scipy.stats import chi2
    threshold = chi2.ppf(1 - alpha, df=dof)
    return bool(chi2_stat > threshold)


def l2_cp_flag(score: float, cp_threshold: float) -> bool:
    """L2: ML 비적합 스코어와 CP 임계 비교.

    cp_threshold: split conformal calibration에서 얻은 (1-α)(1+1/n) quantile.
    주의: ONNX 배포 시 sklearn score_samples(=decision_function+offset)가 아니라
    decision_function(ONNX 출력) 기준으로 스코어 정의를 통일할 것 (2026-08-18 검증).
    """
    return bool(score > cp_threshold)
