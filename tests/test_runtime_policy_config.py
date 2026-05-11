from app.runtime_policy_config import (
    load_runtime_policy_config,
)


def test_runtime_policy_config():

    cfg = (
        load_runtime_policy_config()
    )

    assert isinstance(
        cfg,
        dict,
    )
