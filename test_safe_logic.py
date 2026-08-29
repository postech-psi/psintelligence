"""
SafeLogic L3 단위 테스트 — N-of-M 투표 + 히스테리시스 + 채널 다중화.
pytest 없이 assert 기반 실행 가능. python test_safe_logic.py
"""
import sys
sys.path.insert(0, ".")

from safe_logic import SafeLogic, l0_physical_bound, l1_chi2_flag, l2_cp_flag


def run(name, fn):
    fn()
    print(f"  PASS  {name}")


# ----------------------------------------------------------------------
def test_normal_stream_no_abort():
    sl = SafeLogic(n=3, m=5)
    for _ in range(10):
        d = sl.decide(l0=False, l1=False, l2=False)
        assert d.action == "safe", d
        assert d.state == "normal"
    assert not sl.aborted


def test_l2_sparse_flags_no_abort():
    # 5윈도우 중 2개만 플래그 -> N=3 미달 -> watch/safe 유지
    sl = SafeLogic(n=3, m=5)
    flags = [False, True, False, True, False]  # 2/5
    for f in flags:
        d = sl.decide(l2=f)
        assert d.action in ("safe", "watch")
        assert d.state == "normal"
    assert not sl.aborted
    assert sl.l2_votes == 2


def test_l2_n_of_m_triggers_abort():
    # 5윈도우 중 3개 플래그 -> abort
    sl = SafeLogic(n=3, m=5)
    flags = [False, True, True, True, False]  # 3/5
    aborted_at = None
    for i, f in enumerate(flags):
        d = sl.decide(l2=f)
        if d.action == "abort":
            aborted_at = i
            break
    assert aborted_at is not None
    assert sl.aborted
    assert aborted_at <= 3  # 3번째 True에서 abort 가능해야 함


def test_l0_immediate_abort():
    sl = SafeLogic()
    for _ in range(3):
        sl.decide(l2=False)
    d = sl.decide(l0=True)  # 단발 물리 경계 -> 즉시 abort
    assert d.action == "abort"
    assert sl.aborted


def test_l1_immediate_abort():
    sl = SafeLogic()
    for _ in range(3):
        sl.decide(l2=False)
    d = sl.decide(l1=True)  # 단발 χ² 실패 -> 즉시 abort
    assert d.action == "abort"
    assert sl.aborted


def test_hysteresis_requires_5_clean():
    # abort 후 4연속 정상 -> 아직 abort, 5연속 -> 해제
    sl = SafeLogic(n=3, m=5, hysteresis_clear=5)
    for f in [False, True, True, True]:  # 3/4 -> abort
        d = sl.decide(l2=f)
    assert sl.aborted

    for i in range(4):
        d = sl.decide(l2=False)
        assert d.action == "abort", f"step {i}: 히스테리시스 유지 실패"
        assert d.state == "aborted"
    d = sl.decide(l2=False)  # 5번째 연속 정상
    assert d.action == "safe"
    assert not sl.aborted


def test_hysteresis_reset_on_new_flag():
    # abort 후 정상 2스텝 -> 새 L2 플래그 -> streak 리셋, 계속 abort
    sl = SafeLogic(n=3, m=5, hysteresis_clear=5)
    for f in [True, True, True]:
        sl.decide(l2=f)
    assert sl.aborted
    sl.decide(l2=False)
    sl.decide(l2=False)
    assert sl.normal_streak == 2
    d = sl.decide(l2=True)
    assert d.action == "abort"
    assert sl.normal_streak == 0


def test_hysteresis_release_clears_buffer():
    # 해제 후 이전 플래그가 버퍼에 남아 재-abort되는 회귀 방지 (2026-08-18 발견)
    sl = SafeLogic(n=3, m=5, hysteresis_clear=5)
    for f in [True, True, True]:
        sl.decide(l2=f)
    assert sl.aborted
    d = None
    for _ in range(5):  # 5연속 정상 -> 해제
        d = sl.decide(l2=False)
    assert d is not None and d.action == "safe" and not sl.aborted
    # 해제 직후 정상 스트림에서 즉시 재-abort되면 안 됨 (버퍼 클리어 검증)
    for _ in range(10):
        d = sl.decide(l2=False)
        assert d.action == "safe", d
        assert not sl.aborted


def test_l0_helpers():
    assert l0_physical_bound(15.0, 0.0, 100.0) is False
    assert l0_physical_bound(-1.0, 0.0, 100.0) is True
    assert l0_physical_bound(101.0, 0.0, 100.0) is True


def test_l1_chi2_helper():
    # 자유도 3, α=0.01 임계 ≈ 11.34
    assert l1_chi2_flag(5.0, dof=3, alpha=0.01) is False
    assert l1_chi2_flag(20.0, dof=3, alpha=0.01) is True


def test_l2_cp_helper():
    assert l2_cp_flag(0.5, cp_threshold=1.0) is False
    assert l2_cp_flag(1.5, cp_threshold=1.0) is True


def test_invalid_params():
    try:
        SafeLogic(n=6, m=5)
        assert False, "n>m 허용 안 됨"
    except ValueError:
        pass


def test_reset():
    sl = SafeLogic()
    for f in [True, True, True]:
        sl.decide(l2=f)
    assert sl.aborted
    sl.reset()
    assert not sl.aborted
    assert sl.l2_votes == 0
    assert len(sl.history) == 0


# ----------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_normal_stream_no_abort,
        test_l2_sparse_flags_no_abort,
        test_l2_n_of_m_triggers_abort,
        test_l0_immediate_abort,
        test_l1_immediate_abort,
        test_hysteresis_requires_5_clean,
        test_hysteresis_reset_on_new_flag,
        test_hysteresis_release_clears_buffer,
        test_l0_helpers,
        test_l1_chi2_helper,
        test_l2_cp_helper,
        test_invalid_params,
        test_reset,
    ]
    print(f"SafeLogic 테스트 {len(tests)}건 실행")
    for t in tests:
        run(t.__name__, t)
    print("\n전체 통과 ✅")
