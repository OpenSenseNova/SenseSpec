"""W&B tracking must use caller-owned credentials and respect opt-out modes."""

from types import SimpleNamespace

import pytest

from sensespec.utils import wandb as tracking


@pytest.mark.parametrize(
    "mode,environment_mode",
    [("offline", None), ("disabled", None), (None, "offline"), (None, "disabled")],
)
def test_offline_and_disabled_modes_skip_login(monkeypatch, mode, environment_mode):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    if environment_mode:
        monkeypatch.setenv("WANDB_MODE", environment_mode)
    monkeypatch.setattr(tracking.wandb, "login", lambda **kwargs: pytest.fail("Unexpected login"))
    tracking._login_wandb(SimpleNamespace(wandb_mode=mode))


@pytest.mark.parametrize(
    "argument_key,environment_key,expected_key",
    [
        (None, None, None),
        (None, "test-environment-key", "test-environment-key"),
        ("test-argument-key", "test-environment-key", "test-argument-key"),
    ],
)
def test_online_mode_uses_explicit_or_saved_credentials(
    monkeypatch, argument_key, environment_key, expected_key
):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    if environment_key:
        monkeypatch.setenv("WANDB_API_KEY", environment_key)
    calls = []
    monkeypatch.setattr(tracking.wandb, "login", lambda **kwargs: calls.append(kwargs))
    tracking._login_wandb(
        SimpleNamespace(wandb_mode="online", wandb_key=argument_key, wandb_host=None)
    )
    assert calls == [{"key": expected_key, "host": None}]


def test_disabled_reporter_never_initializes_wandb(monkeypatch):
    monkeypatch.setattr(tracking.wandb, "login", lambda **kwargs: pytest.fail("Unexpected login"))
    monkeypatch.setattr(tracking.wandb, "init", lambda **kwargs: pytest.fail("Unexpected W&B run"))
    args = SimpleNamespace(use_wandb=False)
    tracking.init_wandb_primary(args)
    assert args.wandb_run_id is None
