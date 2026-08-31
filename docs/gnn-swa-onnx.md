# PyTorch GNN, SWA and ONNX

The repository now includes a CPU-runnable graph network and export path.

Install the optional CPU ONNX validation dependencies with
`pip install -e ".[onnx]"`.  The checked-in golden test can then be run with
`pytest -q tests/test_gnn_onnx.py`.

```python
from gaiazero.config import load_training_config
from gaiazero.game import GaiaState
from gaiazero.gnn import GraphHybridNetwork, graph_inputs_from_state
from gaiazero.onnx_export import export_swa_to_onnx, verify_onnx_cpu_golden

config = load_training_config()
graph_config = config.graph_network_config(3)
model = GraphHybridNetwork(graph_config).eval()
state = GaiaState.initial(3, seed=0, seed_stream_version=config.seed_stream_version)
inputs = graph_inputs_from_state(state, graph_config)
export_swa_to_onnx(model, "model.onnx", inputs)
verify_onnx_cpu_golden(model, "model.onnx", inputs)
```

`GraphHybridNetwork` accepts padded node/edge/player tensors and emits exactly
four ONNX outputs: `action_type_logits`, `action_argument_logits`,
`pairwise_wdl_logits` and `vp_belief_logits`.  The pairwise and VP heads are
the two value components of the three-head production contract; the policy is
factorized into action type and conditional argument slots.

`SWAAccumulator` performs equal-weight rolling averaging after the configured
sample threshold and interval. `AlphaZeroTrainer` updates it after optimizer
steps, and `save_checkpoint(..., swa=trainer.swa)` persists the snapshots. The
averaged weights can be copied to a model or exported through
`export_swa_to_onnx` without mutating the training model.

The fixed CPU golden case is
[`tests/fixtures/gnn_cpu_golden.json`](../tests/fixtures/gnn_cpu_golden.json).
It freezes model/input seeds, tensor shapes and a PyTorch output digest; the
test also compares ONNX Runtime CPU output within the recorded tolerances.
