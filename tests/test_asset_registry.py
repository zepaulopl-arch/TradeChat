from app.asset_registry import canonical_asset_ticker


def test_canonical_asset_ticker_keeps_current_embj3():
    cfg = {
        "assets_registry": {
            "assets": {
                "EMBJ3.SA": {
                    "aliases": ["EMBJ3", "EMBR3", "EMBR3.SA"],
                    "lineage": {
                        "predecessors": [
                            {
                                "ticker": "EMBR3.SA",
                                "b3_code": "EMBR3",
                                "until": "2025-10-31",
                                "reason": "ticker_migration",
                            }
                        ]
                    },
                }
            }
        }
    }

    assert canonical_asset_ticker(cfg, "EMBJ3.SA") == "EMBJ3.SA"


def test_canonical_asset_ticker_maps_embraer_predecessor_to_current_ticker():
    cfg = {
        "assets_registry": {
            "assets": {
                "EMBJ3.SA": {
                    "aliases": ["EMBJ3", "EMBR3", "EMBR3.SA"],
                    "lineage": {
                        "predecessors": [
                            {
                                "ticker": "EMBR3.SA",
                                "b3_code": "EMBR3",
                                "until": "2025-10-31",
                                "reason": "ticker_migration",
                            }
                        ]
                    },
                }
            }
        }
    }

    assert canonical_asset_ticker(cfg, "EMBR3.SA") == "EMBJ3.SA"
    assert canonical_asset_ticker(cfg, "EMBR3") == "EMBJ3.SA"
