# torch-nested-ad-narrow-guard

诊断并规避 PyTorch 中两个真实存在、已在本机独立复现的正确性缺陷
（本机验证环境：torch 2.14.0，macOS arm64 CPU）：

## 缺陷一 —— 嵌套前向模式自动微分静默将 slogdet 二阶导数归零

**上游 issue：** [pytorch/pytorch#196697](https://github.com/pytorch/pytorch/issues/196697)（open）

在 `torch.func.jvp` 内部再嵌套调用一次 `torch.func.jvp`（即"前向套前向"
的二阶微分）时，`torch.linalg.slogdet` 的二阶导数会被静默地计算为
`0.0`，而不是数学上正确的非零值。危险之处在于它**不会**报错、不会
产生 NaN、也不会有任何警告。

按 issue 原文精确复现（已在本机验证）：期望值
`-0.17853600476096013`，实际返回 `0.0`。

开发过程中额外验证："前向套反向"（`torch.func.grad` 作用于
`torch.func.jvp`）**同样**是静默错误的（也返回 `0.0`）。只有"反向
优先"的微分顺序才正确：反向套反向（两次
`torch.autograd.grad(create_graph=True)`）和反向套前向
（`torch.func.jvp` 作用于 `torch.func.grad`）均与解析值精确匹配。

`safe_nested_slogdet_second_order_jvp(f, t)` 使用反向套反向的微分
方式，已验证与解析值的误差小于 `1e-9`。

## 缺陷二 —— jagged 嵌套张量的 narrow+unbind 会混入未选中的元素

**上游 issue：** [pytorch/pytorch#196708](https://github.com/pytorch/pytorch/issues/196708)（open）

`torch.nested.narrow(x, dim, starts, lengths, layout=torch.jagged)`
之后再调用 `.unbind()`，至少有一行会静默地包含请求范围
`[start, start+length)` 之外的元素。

按 issue 原文精确复现（已在本机验证）：期望和为 `4.8999999999999995`
(=7*t)，实际返回 `6.999999999999999` (=10*t，混入了未选中的 3*t)。

`safe_jagged_narrow_unbind(dense_x, dim, starts, lengths)` 完全不调用
有缺陷的 `torch.nested.narrow`，而是直接用普通张量索引构造正确的
按行切片结果，已验证与期望选取结果完全一致。

## 安装

```bash
pip install "torch-nested-ad-narrow-guard[torch]"
```

## 使用

```bash
torch-nested-ad-narrow-guard          # 人类可读报告
torch-nested-ad-narrow-guard --json   # 机器可读 JSON
torch-nested-ad-narrow-guard --no-color
```

## 真实局限性

- `safe_jagged_narrow_unbind` 目前仅支持 `dim=1`（与上游复现代码一致）
  及稠密二维输入，未覆盖所有 jagged-narrow 调用形态。
- `safe_nested_slogdet_second_order_jvp` 要求 `f` 可被
  `torch.autograd` 直接微分（内部不能再使用 `torch.func` 变换）。
- 这些是**规避方案**，不是上游修复。如果 PyTorch 未来修复了这两个
  issue，本包自身的回归测试会失败——这是需要重新核实并更新本
  README 的信号，而非本包出现了退化。
- 仅在 torch 2.14.0（macOS arm64 CPU 本机测试 + ubuntu-latest /
  macos-latest CI，均为纯 CPU）上验证过，未在 CUDA/ROCm/MPS 加速
  路径上独立验证。

## 许可证

MIT
