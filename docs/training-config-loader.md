# Training Config Loader

`configs/gaia-training.json` is the source of truth for a training line. Use
the loader directly:

```python
from gaiazero.config import load_training_config

cfg = load_training_config("configs/gaia-training.json")
pipeline = cfg.pipeline_config(3)
```

The loader validates `config_version`, `standard-v22`, `action-tuple-v1`, the
versioned setup seed stream, required sections, player profiles and the
single-GPU runtime policy. It computes a canonical SHA-256 `config_hash`.

The generated `PipelineConfig` carries the source path/hash, profile-specific
network ID, network capacity, search budgets, training batch/lr/weight decay,
shuffle interval/window, device, and `seed_stream_version`. Every self-play
setup passes that seed-stream version to `GaiaState.initial`; raw shards and
training checkpoints record the config identity.

Run the complete pipeline from the JSON document without creating a manual
`pipeline.json` first:

```powershell
gaiazero pipeline --players 3 --training-config configs/gaia-training.json
```

The worker entry point also accepts the document directly:

```powershell
python -m gaiazero.distributed selfplay --players 3 --config configs/gaia-training.json --once
```

Operational values can be adjusted under the optional `pipeline` section;
schema, seed-stream, network, search and training values remain in their
respective sections. Explicit `GaiaTrainingConfig.pipeline_config(...,
overrides={...})` values are limited to `PipelineConfig` fields and are
included in the resulting run snapshot.
