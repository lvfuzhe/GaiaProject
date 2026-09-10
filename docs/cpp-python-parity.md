# C++ / Python rules parity

The C++ `GaiaState` core is aligned with the Python `standard-v22` reference
for the currently modelled game state and action contract.

Covered by the parity harness:

- BGA-compatible setup generation for `2p-reduced`, `3p-reduced`,
  `3p-normal`, and `4p-normal`, including seed streams and setup hash.
- Canonical JSON and `state-hash-v1` after every tested transition.
- Semantic `ActionTuple` names, arguments, ordering, and legal-action masks.
- Snake placement, booster selection, income, building, terraforming,
  research, technology, power/QIC actions, passing, and terminal state.
- Federation minimum-satellite paths and federation adjacency/reduction rules.
- Passive charging, Taklons brainstone handling, Terrans Gaia conversion,
  Itars Gaia technology decisions, Gleens federation reward, and other
  special-faction actions represented by the standard rules core.
- GraphBatch input tensors are covered by a byte-for-byte golden digest:
  `float32`/`int64` dtypes, fixed shapes, padding masks, node order, directed
  adjacency, relation IDs, and absolute player order are checked for five
  seeds in each of the 2/3/4-player profiles.

Run the checks after building the probes:

```powershell
python -m pytest tests/test_cpp_setup_parity.py tests/test_cpp_state_parity.py -q
python -m pytest tests/test_step0_contract.py -q
python scripts/verify_cpp_python_parity.py --output tests/fixtures/step0-parity-acceptance.json
python -m pytest -q
ctest --test-dir build/cpp-msvc -C Release --output-on-failure
```

The state parity tests include deterministic traces, alternative-action
traces, targeted special-rule scenarios, and full terminal trajectories for
multiple seeds and player counts. The current acceptance run covers all 1024
stored BGA setup samples and six 2,000-step random traces; no setup, special
rule, legal-action, canonical-state, or state-hash difference was observed.
The machine-readable result is stored in
`tests/fixtures/step0-parity-acceptance.json`.
Production C++ selfplay, TensorRT inference and gatekeeper remain separate
pending execution items.
