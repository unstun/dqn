# v7p2 - 变更

## 版本意图
- 以最小改动修复 `v7p1` 中“加速度平滑奖励依赖历史动作、观测未显式给出历史量”的部分马尔可夫性问题。
- 保持训练/推理策略与奖励系数不变，避免把本版混入额外调参影响。

## 相对 v7p1 的代码/配置变更
- 观测修复（核心）：
  - `forest_vehicle_dqn/env.py`
    - `AMRBicycleEnv` 观测维度：`10 + N^2 -> 11 + N^2`
    - `_observe()` 新增 `prev_a_n`（`self._prev_a / a_max_m_s2`，再裁剪到 `[-1,1]`）
    - 标量顺序更新为：
      - `[ax_n, ay_n, gx_n, gy_n, sin_psi, cos_psi, v_n, delta_n, prev_a_n, alpha_n, od_n]`
- 网络布局兼容：
  - `forest_vehicle_dqn/networks.py`
    - `infer_flat_obs_cnn_layout(...)` bicycle 识别从 `10 + N^2` 更新为 `11 + N^2`
    - 错误提示文案同步更新
- 测试新增：
  - `tests/test_v7p2_markov_obs_prev_a.py`
    - 用 `unittest` 验证观测维度与 `prev_a_n` 更新逻辑
    - 验证 `infer_flat_obs_cnn_layout(...)` 可识别 `11 + N^2`

## 配置与文档变更
- 新增主 profile：
  - `configs/v7p2.json`
- 新增可复现配置：
  - `configs/repro_20260220_v7p2_markov_obs_prev_a.json`
- README 同步：
  - `README.md`
  - `README.zh-CN.md`
  - 更新“最新训练/推理命令”为 `v7p2`
  - 记录 `v7p2` 观测维度变更兼容性说明
- 版本索引同步：
  - `README.md`（版本总索引）
  - `docs/versions/README.md`（镜像索引）
- 新增版本四件套：
  - `docs/versions/v7p2/README.md`
  - `docs/versions/v7p2/CHANGES.md`
  - `docs/versions/v7p2/RESULTS.md`
  - `docs/versions/v7p2/runs/README.md`

## 追加实验记录（2026-02-20）
- 本轮无新增代码改动，仅执行与归档以下实验：
  - `v7p2_full300`：默认 `best` checkpoint（检查点）
  - `v7p2_final300`：显式 `--save-ckpt final`
  - 推理均为 `short/mid/long` 各 `runs=20`
- 追加留档文件更新：
  - `docs/versions/v7p2/README.md`
  - `docs/versions/v7p2/RESULTS.md`
  - `docs/versions/v7p2/runs/README.md`
