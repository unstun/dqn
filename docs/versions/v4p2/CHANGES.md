# v4p2 代码/配置改动

## 相对 v4p1 的核心变更
- `forest_vehicle_dqn/agents.py`
  - `AgentConfig` 新增 `aux_admissibility_lambda`。
  - `DQNFamilyAgent` 新增训练期 `aux_adm_head`（仅在 `cnn` 且 `aux_admissibility_lambda>0` 时启用）。
  - `legacy` 与 `dqfd` 两条更新路径新增 `aux_adm_loss`（BCE with logits）。
  - 梯度裁剪改为覆盖 `q + aux_head` 参数。
  - checkpoint 新增 `aux_adm_head_state_dict` 保存/加载；加载后重建优化器参数组。
- `forest_vehicle_dqn/cli/train.py`
  - 新增 CLI 参数 `--aux-admissibility-lambda`。
  - `AgentConfig` 构建时透传 `aux_admissibility_lambda`。
- `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke.json`
  - 新建 v4p2 可复现配置（继承 v4p1，默认 `aux_admissibility_lambda=0.2`）。

## 运行归档更新（2026-02-10）
- 本轮新增 `iter3(aux=0.01)` smoke 训练与推理归档（无新增代码变更）。
- 归档文件更新：
  - `docs/versions/v4p2/README.md`
  - `docs/versions/v4p2/RESULTS.md`
  - `docs/versions/v4p2/runs/README.md`

## 受影响文件清单（代码/配置）
- `forest_vehicle_dqn/agents.py`
- `forest_vehicle_dqn/cli/train.py`
- `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke.json`

## 命名与定义一致性
- DDQN 定义保持不变（online argmax + target eval）。
- DQfD 定义保持不变（PER + 1-step/n-step + margin + L2）。
- strict no-fallback 保持不变（推理期无 mask/top-k/replacement/fallback/heuristic takeover）。
