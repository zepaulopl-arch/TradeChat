from app.runtime_policy import (
    resolve_policy_profile,
)


def test_runtime_policy_default():

    value = (
        resolve_policy_profile(
            "PETR4.SA"
        )
    )

    assert isinstance(
        value,
        str,
    )
