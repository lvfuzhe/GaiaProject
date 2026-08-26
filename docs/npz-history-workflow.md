# NPZ and History Separation

Training samples and dashboard history are separate data products.

The five training workers only use these paths:

```text
raw/*.npz -> shuffled/*.npz -> PyTorch training -> candidates/*.pt
```

No worker converts samples to replay JSON and no worker imports the history
module. This keeps self-play and training lightweight and prevents replay files
from growing with every training game.

To inspect a training sample in the dashboard, explicitly convert it:

```powershell
gaiazero npz-to-history runs/pipeline-4p/raw/game-0001.npz --history-dir runs/history
```

For a directory:

```powershell
python scripts/npz_to_history.py runs/pipeline-4p/raw --history-dir runs/history
```

Raw self-play NPZ files contain both the four training arrays and a complete
`gaiazero-replay-trace-v1` history: initial setup, every action, post-action
state snapshot, MCTS candidates, and the terminal state. The converter writes
that trace as `source=training_npz` local-history JSON. The existing history
API and page can load it, and the converted run can be deleted from the page or
with:

```powershell
gaiazero delete-training-history npz-game-0001 --history-dir runs/history
```

Deletion affects only the converted history JSON. It does not delete raw or
shuffled NPZ files used by training. Those remain managed by the pipeline data
retention policy and can be removed separately when a training run is stopped.
