# Config directory

YAML files were moved out of the project root into `config/`:

- `config/config.yaml`: operational parameters, model, policy, simulation and UI.
- `config/data.yaml`: asset registry, groups, subgroups, CNPJ fields and linked context indices.
- `config/features.yaml`: feature families, windows, presets, preparation and selection.

`load_config()` loads `config/config.yaml` by default and resolves related YAML files relative to that directory.
