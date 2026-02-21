# v7p2p4 改动清单（相对 v7p2p3 / v7p1）

## 变更目标
- 保持 `cnn-ddqn` 算法定义不变，仅做网络模块单变量改造：在 `globalcnn_fusion` 上加入空间先验通道，提升地图与位姿对齐能力。

## 代码/配置改动明细

### 1) CNN 网络新增空间先验
- `forest_vehicle_dqn/networks.py`
  - `CNNQNetwork` 新增参数：
    - `globalcnn_spatial_prior`：是否启用空间先验通道。
    - `globalcnn_prior_sigma`：高斯热图核宽度。
  - 在 `globalcnn/globalcnn_fusion` 模式下，若启用先验：
    - 由标量 `ax/ay/gx/gy` 生成 `agent/goal` 两张热力图；
    - 与占据图拼接后送入卷积骨干。
  - 默认保持关闭，旧模型行为不变。

### 2) Agent 参数与 checkpoint 兼容
- `forest_vehicle_dqn/agents.py`
  - `AgentConfig` 增加 `globalcnn_spatial_prior`、`globalcnn_prior_sigma`。
  - `DQNFamilyAgent` 创建网络时透传新参数。
  - `load()` 对旧 checkpoint 使用默认值补齐，保持兼容。

### 3) 训练 CLI 扩展
- `forest_vehicle_dqn/cli/train.py`
  - 新增参数：
    - `--cnn-global-spatial-prior/--no-cnn-global-spatial-prior`
    - `--cnn-global-prior-sigma`
  - 参数写入 `AgentConfig` 并保存到 run 配置。

### 4) 单测扩展
- `tests/test_globalcnn_network.py`
  - 新增 `test_globalcnn_spatial_prior_channel_wiring`，验证先验通道接线维度正确。
  - `globalcnn/globalcnn_fusion` roundtrip 覆盖新参数保存/加载。
  - 旧 checkpoint 兼容加载测试保持通过。

### 5) 新增配置
- `configs/v7p2p4.json`
- `configs/repro_20260221_v7p2p4_globalcnn_spatialprior_smoke.json`

## 受影响文件清单
- `forest_vehicle_dqn/networks.py`
- `forest_vehicle_dqn/agents.py`
- `forest_vehicle_dqn/cli/train.py`
- `tests/test_globalcnn_network.py`
- `configs/v7p2p4.json`
- `configs/repro_20260221_v7p2p4_globalcnn_spatialprior_smoke.json`
- `docs/versions/v7p2p4/README.md`
- `docs/versions/v7p2p4/CHANGES.md`
- `docs/versions/v7p2p4/RESULTS.md`
- `docs/versions/v7p2p4/runs/README.md`
- `docs/versions/README.md`
- `README.md`
- `README.zh-CN.md`
