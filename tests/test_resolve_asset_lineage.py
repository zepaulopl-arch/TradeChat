from app.data import resolve_asset


def test_resolve_asset_maps_embraer_predecessor():
    cfg = {
        "assets_registry": {
            "assets": {
                "EMBJ3.SA": {
                    "aliases": ["EMBR3", "EMBR3.SA"],
                    "lineage": {
                        "predecessors": [
                            {
                                "ticker": "EMBR3.SA",
                            }
                        ]
                    },
                    "registry_status": "active",
                }
            }
        }
    }

    resolved = resolve_asset(cfg, "EMBR3.SA")

    assert resolved["canonical"] == "EMBJ3.SA"
    assert resolved["changed"] is True
