"""ONNX export and CPU golden verification for the graph network."""

from __future__ import annotations

import hashlib
import json
import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from gaiazero.contracts import ACTION_TUPLE_SCHEMA_VERSION, RULES_VERSION, STATE_HASH_VERSION
from gaiazero.gnn import GraphHybridNetwork, load_graph_checkpoint
from gaiazero.swa import SWAAccumulator


def _tensor_digest(values: Sequence[torch.Tensor | np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def export_graph_onnx(
    model: GraphHybridNetwork,
    destination: str | Path,
    example_inputs: Sequence[torch.Tensor],
    *,
    opset_version: int = 18,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Export the graph model with stable input/output names and a manifest."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    if len(example_inputs) != 8:
        raise ValueError("graph ONNX export requires eight example input tensors")
    with torch.inference_mode():
        expected = model(*example_inputs)
    torch.onnx.export(
        model,
        tuple(example_inputs),
        path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=(
            "node_features",
            "edge_index",
            "edge_type",
            "edge_mask",
            "node_mask",
            "global_features",
            "player_features",
            "player_mask",
        ),
        output_names=model.output_names,
        dynamic_axes={
            name: {0: "batch"}
            for name in (
                "node_features",
                "edge_index",
                "edge_type",
                "edge_mask",
                "node_mask",
                "global_features",
                "player_features",
                "player_mask",
                *model.output_names,
            )
        },
    )
    manifest = {
        "format": "gaiazero-graph-onnx-v1",
        "rules_version": RULES_VERSION,
        "action_schema_version": ACTION_TUPLE_SCHEMA_VERSION,
        "state_hash_version": STATE_HASH_VERSION,
        "opset_version": opset_version,
        "architecture_family": model.architecture_family,
        "network_config": asdict(model.config),
        "input_names": [
            "node_features",
            "edge_index",
            "edge_type",
            "edge_mask",
            "node_mask",
            "global_features",
            "player_features",
            "player_mask",
        ],
        "output_names": list(model.output_names),
        "input_shapes": [list(value.shape) for value in example_inputs],
        "output_shapes": [list(value.shape) for value in expected],
        "golden_output_digest": _tensor_digest(expected),
        "metadata": dict(metadata or {}),
    }
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def export_swa_to_onnx(
    model: GraphHybridNetwork,
    destination: str | Path,
    example_inputs: Sequence[torch.Tensor],
    *,
    swa: SWAAccumulator | None = None,
    opset_version: int = 18,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Export a graph model using averaged weights when SWA is active."""

    exported = copy.deepcopy(model).cpu().eval()
    if swa is not None and swa.active:
        swa.copy_to(exported)
    payload = dict(metadata or {})
    payload["weights_source"] = "swa" if swa is not None and swa.active else "model"
    if swa is not None:
        payload["swa"] = swa.metadata()
    return export_graph_onnx(
        exported,
        destination,
        example_inputs,
        opset_version=opset_version,
        metadata=payload,
    )


def export_swa_checkpoint_to_onnx(
    checkpoint: str | Path,
    destination: str | Path,
    example_inputs: Sequence[torch.Tensor],
    *,
    opset_version: int = 18,
) -> Path:
    """Load a graph checkpoint, restore its SWA snapshots, and export ONNX."""

    model, metadata, swa_state = load_graph_checkpoint(checkpoint, "cpu")
    accumulator = None
    if swa_state is not None:
        from gaiazero.swa import SWAAccumulator, SWAConfig

        accumulator = SWAAccumulator(SWAConfig.from_mapping(swa_state.get("config")))
        accumulator.load_state_dict(swa_state)
    return export_swa_to_onnx(
        model,
        destination,
        example_inputs,
        swa=accumulator,
        opset_version=opset_version,
        metadata={"checkpoint": str(checkpoint), **metadata},
    )


def verify_onnx_cpu_golden(
    model: GraphHybridNetwork,
    onnx_path: str | Path,
    example_inputs: Sequence[torch.Tensor],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    """Compare ONNX Runtime CPU output with PyTorch CPU output."""

    try:
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("onnxruntime is required for CPU golden verification") from error
    model = model.cpu().eval()
    with torch.inference_mode():
        expected = tuple(value.cpu() for value in model(*example_inputs))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        list(model.output_names),
        {
            input_node.name: value.detach().cpu().numpy()
            for input_node, value in zip(session.get_inputs(), example_inputs, strict=True)
        },
    )
    errors = [
        float(np.max(np.abs(value - reference.numpy())))
        for value, reference in zip(actual, expected, strict=True)
    ]
    passed = all(
        np.allclose(value, reference.numpy(), atol=atol, rtol=rtol)
        for value, reference in zip(actual, expected, strict=True)
    )
    return {
        "passed": passed,
        "max_abs_error": max(errors, default=0.0),
        "per_output_max_abs_error": errors,
        "torch_digest": _tensor_digest(expected),
        "onnx_digest": _tensor_digest(actual),
    }
