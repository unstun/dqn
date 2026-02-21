# v7p2p2 改动清单（相对 v7p2p1 / v7p1）

## 变更目标
- 仅实施“GlobalCNN 模块改造”，不改 `cnn-ddqn` 算法定义，不改推理口径。

## 代码/配置改动明细

### 1) CNN 网络支持双骨干
- `legacy`（旧骨干）保持原逻辑不变。
- 新增 `globalcnn`（多尺度全局池化骨干）：
  - `cnn_backbone: legacy -> globalcnn`（可选）
  - `globalcnn_width`（新参数）
  - `globalcnn_dropout`（新参数）

### 2) Agent 配置与加载兼容
- `AgentConfig` 增加：
  - `cnn_backbone`
  - `globalcnn_width`
  - `globalcnn_dropout`
- `DQNFamilyAgent.save/load`：
  - 保存时落盘 `network_kwargs`（网络参数）包含新字段。
  - 加载旧 checkpoint 时，若缺失新字段，自动填默认值并按 `legacy` 兼容重建。

### 3) 训练 CLI 参数扩展
- 新增参数：
  - `--cnn-backbone`
  - `--cnn-global-width`
  - `--cnn-global-dropout`
- 参数透传到 `AgentConfig`，默认仍为 `legacy`，保证旧命令行为不变。

### 4) 新增测试
- 新增 `tests/test_globalcnn_network.py`：
  - 骨干前向维度测试（legacy/globalcnn）。
  - GlobalCNN checkpoint 保存/加载回归。
  - 旧 checkpoint（无新字段）兼容加载回归。

### 5) 新增复现配置
- 新增 `configs/repro_20260221_v7p2p2_globalcnn_smoke.json`：
  - smoke 门固定为 `episodes=150`、`runs=3`。
  - 算法保持 `cnn-ddqn`，仅启用 `cnn_backbone=globalcnn`。

### 6) smoke 运行与版本留档更新
- 远端完成 `episodes=150` 训练与 `runs=3` 推理后，回传 `runs/v7p2p2_globalcnn_smoke/*` 到本地。
- 将 `docs/versions/v7p2p2/` 四件套从 `N/A` 更新为真实 run 路径、KPI 与 `failure_reason` 分布。
- 同步更新 `docs/versions/README.md`、`README.md`、`README.zh-CN.md` 中 `v7p2p2` 状态为失败归档。

## 受影响文件清单
- `forest_vehicle_dqn/networks.py`
- `forest_vehicle_dqn/agents.py`
- `forest_vehicle_dqn/cli/train.py`
- `tests/test_globalcnn_network.py`
- `configs/repro_20260221_v7p2p2_globalcnn_smoke.json`
- `README.md`
- `README.zh-CN.md`
- `docs/versions/README.md`
- `docs/versions/v7p2p2/README.md`
- `docs/versions/v7p2p2/RESULTS.md`
- `docs/versions/v7p2p2/CHANGES.md`
- `docs/versions/v7p2p2/runs/README.md`
