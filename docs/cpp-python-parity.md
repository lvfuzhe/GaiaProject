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

Run the checks after building the probes:

```powershell
python -m pytest tests/test_cpp_setup_parity.py tests/test_cpp_state_parity.py -q
python -m pytest -q
ctest --test-dir build/cpp-msvc -C Release --output-on-failure
```

The state parity tests include deterministic traces, alternative-action
traces, targeted special-rule scenarios, and full terminal trajectories for
multiple seeds and player counts. Production C++ selfplay, TensorRT inference
and gatekeeper remain separate pending execution items.
