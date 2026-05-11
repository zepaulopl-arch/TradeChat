from app.commands.promote_policy import (
    promote_policy,
)


def test_promote_policy_import():

    assert callable(
        promote_policy
    )
