# C++/CMake 基础工程

仓库根目录现在包含一个独立的 C++20/CMake 工程。首版只冻结跨语言边界：

- `gaiazero::ActionTuple` 与 `standard-v22` action type 审计编号。
- 固定尺寸的 `GraphBatch`，对应 Python GNN/ONNX 的节点、边、玩家和 mask 输入。
- `InferenceBackend` 抽象，后续 ONNX Runtime/TensorRT 后端都通过同一接口接入。
- `OnnxRuntimeCpuBackend` 参考后端：加载模型时校验固定输入/输出签名，执行 CPU 推理并返回四个 head 的 logits。
- CTest smoke 测试，验证契约编号、tuple canonicalization 和张量尺寸。

CUDA、ONNX Runtime 和 TensorRT 都是可选 SDK，默认关闭；因此当前没有 NVIDIA GPU 或 CUDA/TensorRT SDK 也可以编译和测试。

## VS2026 + Ninja

在普通 PowerShell 中先加载 MSVC 环境（路径按本机安装位置调整）：

```powershell
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "call `"$vcvars`" && cmake --preset windows-msvc-ninja"
cmd /c "call `"$vcvars`" && cmake --build --preset windows-msvc-ninja"
cmd /c "call `"$vcvars`" && ctest --preset windows-msvc-ninja"
```

也可以在已打开的 VS Developer PowerShell 中直接运行：

```powershell
cmake --preset windows-msvc-ninja
cmake --build --preset windows-msvc-ninja
ctest --preset windows-msvc-ninja
```

启用可选 SDK wiring 时显式传入其根目录。例如，预留 ONNX Runtime CPU 后端：

```powershell
cmake --preset windows-msvc-ninja -DGAIA_ENABLE_ORT_CPU=ON -DONNXRUNTIME_ROOT="C:\sdk\onnxruntime"
```

`ONNXRUNTIME_ROOT` 必须同时包含 `onnxruntime_cxx_api.h`、对应的 import library（Windows 通常为 `onnxruntime.lib`）和运行时 DLL。配置成功后，C++ 代码可这样创建后端：

```cpp
gaiazero::OnnxRuntimeCpuBackend backend("candidate.onnx");
gaiazero::NetworkOutput output = backend.infer(graph_batch);
```

后端会拒绝不存在的模型、输入/输出数量或名称不匹配、类型/秩不匹配以及运行时 shape 不匹配的模型。模型导出的合法动作过滤仍由 C++ 规则引擎负责，ONNX 后端只返回原始 logits。

启用 CUDA 时需要本机已安装 CUDA Toolkit；没有 Toolkit 时 CMake 会在配置阶段明确报错：

```powershell
cmake --preset windows-msvc-ninja -DGAIA_ENABLE_CUDA=ON
```

当前 C++ 工程尚未迁移完整 Gaia 规则、MCTS、selfplay 或 gatekeeper；这些模块会在契约稳定后逐步接入，Python/PyTorch 训练仍是现阶段的参考实现。
