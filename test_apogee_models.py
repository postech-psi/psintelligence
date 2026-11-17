"""PyTorch contract tests. Run only in a torch-enabled environment."""
from __future__ import annotations

import torch

from apogee_models import (
    GRUApogeeModel,
    SnapshotApogeeModel,
    _state_dict_from_physics_wrapper,
    project_apogee,
    raw_apogee,
    raw_apogee_loss,
    score_state_dict,
    validate_physics_inputs,
)


def test_gru_returns_raw_ratio_and_keeps_legacy_keys():
    model = GRUApogeeModel(input_dim=6, hidden_dim=8, num_layers=2)
    model.eval()
    x = torch.randn(3, 100, 6)
    r_raw, _ = model(x)
    assert r_raw.shape == (3,)
    assert torch.all((r_raw > 0.5) & (r_raw < 1.5))
    legacy = {f"gru.{k}": v for k, v in model.state_dict().items()}
    clone = GRUApogeeModel(input_dim=6, hidden_dim=8, num_layers=2)
    clone.load_legacy_state_dict(legacy)
    clone.eval()
    assert torch.allclose(r_raw, clone(x)[0])


def test_bare_state_dict_is_not_mistaken_for_a_wrapper():
    model = GRUApogeeModel(input_dim=6, hidden_dim=8, num_layers=2)
    state = model.state_dict()
    cleaned = _state_dict_from_physics_wrapper(state)
    assert list(cleaned) == list(state)


def test_raw_loss_receives_gradient_when_raw_ratio_is_below_height_floor():
    model = SnapshotApogeeModel(input_dim=6, hidden_dim=8)
    with torch.no_grad():
        model.net[-2].bias.fill_(-2.0)
    x = torch.zeros(2, 6)
    h = torch.tensor([100.0, 100.0])
    v = torch.tensor([10.0, 10.0])
    r_target = torch.tensor([0.9, 0.9])
    h_target = (h + v**2 / (2 * 9.81)) * r_target
    r_pred = model(x)
    h_pred = raw_apogee(h, v, r_pred)
    assert torch.all(h_pred < h)
    loss = raw_apogee_loss(r_pred, r_target, h_pred, h_target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    projected, _, below = project_apogee(h, v, r_pred)
    assert projected.shape == r_pred.shape
    assert below.dtype == torch.bool
    assert torch.isfinite(projected).all()


def test_projection_flags_below_current_altitude_without_hiding_raw():
    h = torch.tensor([100.0])
    v = torch.tensor([10.0])
    r_raw = torch.tensor([0.5])
    h_raw = raw_apogee(h, v, r_raw)
    projected, r_proj, below = project_apogee(h, v, r_raw)
    assert h_raw.item() < h.item()
    assert below.item()
    assert projected.item() >= h.item()
    assert r_proj.item() > r_raw.item()


def test_invalid_denominator_is_rejected():
    try:
        validate_physics_inputs(torch.tensor([-1.0]), torch.tensor([0.0]))
    except ValueError:
        return
    raise AssertionError("non-positive denominator must fail")


def test_best_state_copy_does_not_share_storage():
    model = SnapshotApogeeModel(input_dim=6, hidden_dim=8)
    saved = score_state_dict(model.state_dict())
    for value in model.parameters():
        value.data.add_(1.0)
    assert not torch.allclose(saved[next(iter(saved))], model.state_dict()[next(iter(saved))])


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    print(f"apogee_models 테스트 {len(tests)}건 실행")
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print("\n전체 통과 ✅")
