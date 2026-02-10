# 版本归档标准（vxpx）

本文定义 `docs/versions/` 下所有版本留档的强制结构与写作规范。

## 1）目录结构（硬约束）

每个版本 `vxpx` 必须使用独立目录：

- `docs/versions/<version>/README.md`
- `docs/versions/<version>/CHANGES.md`
- `docs/versions/<version>/RESULTS.md`
- `docs/versions/<version>/runs/README.md`

仅保留单文件（如 `docs/versions/<version>.md`）不属于有效正式归档。

## 2）内容契约

### `README.md`
必须包含：
- 版本类型（`v+1` 或 `p+1`）
- 本版目标与摘要
- 推理期口径一致性说明（`strict-argmax` / `shielded/masked/hybrid`）
- 代表运行路径（`run_dir`、`run_json`、KPI CSV）
- 结论与下一步方向

### `CHANGES.md`
必须包含：
- 版本意图
- 相对上一版的具体变更（建议 `old -> new`）
- 受影响文件列表
- 关键参数快照

### `RESULTS.md`
必须包含：
- KPI 源路径（`table2_kpis_mean_raw.csv`）
- short/long 的 CNN 与 Hybrid 对比指标
- 门槛检查（success/path length/path time）
- 基于原始 KPI 的 `failure_reason` 分布摘要
- 最终结论（`通过`/`未通过`）

### `runs/README.md`
必须包含：
- 代表运行绑定（`run_dir`、`run_json`、KPI）
- 该版本已发现/采用运行列表
- 可选：best 软链接或指针

## 3）可追溯性规则

- 全部指标必须可追溯到 `runs/` 下真实文件。
- 指标或路径无法确认时必须写 `N/A` 并说明原因。
- 未经数据证据，不得改写历史结论。

## 4）索引同步

任何版本归档更新后，必须同步更新：

- `docs/versions/README.md`（版本索引表）

索引行至少包含：
- 版本号
- 目录路径
- 最佳正式 short/long success_rate
- 基线 short/long success_rate
- 状态（`通过`/`未通过`）

## 5）最小自检命令

```bash
for v in v1 v2 v3 v3p1 v3p2 v3p3 v3p4 v3p5 v3p6 v3p7 v3p8 v3p9 v3p10; do
  test -f docs/versions/$v/README.md || echo "missing README: $v"
  test -f docs/versions/$v/CHANGES.md || echo "missing CHANGES: $v"
  test -f docs/versions/$v/RESULTS.md || echo "missing RESULTS: $v"
  test -f docs/versions/$v/runs/README.md || echo "missing runs/README: $v"
done
```
