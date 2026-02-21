# v7p2p3 改动清单（相对 v7p2p2 / v7p1）

## 变更目标
- 继续保持 `cnn-ddqn` 算法定义不变，仅做网络模块单变量改造：从 `globalcnn` 升级到 `globalcnn_fusion`（全局+局部双分支融合）。

## 代码/配置改动明细

### 1) CNN 网络新增融合骨干
- `legacy` 与 `globalcnn` 保持原逻辑。
- 新增 `globalcnn_fusion`：
  - 复用 `globalcnn` 的多尺度全局分支；
  - 叠加局部高分辨率分支；
  - 使用门控融合后送入 Q 头部。

### 2) 训练 CLI 扩展骨干枚举
- `--cnn-backbone` 新增可选值：`globalcnn_fusion`。
- `--cnn-global-width`、`--cnn-global-dropout` 说明同步覆盖 `globalcnn_fusion`。

### 3) 单测扩展
- `tests/test_globalcnn_network.py`：
  - 前向维度测试扩到 `legacy/globalcnn/globalcnn_fusion` 三种骨干；
  - checkpoint roundtrip 扩到 `globalcnn/globalcnn_fusion`。

### 4) 新增复现配置
- 新增 `configs/repro_20260221_v7p2p3_globalcnn_fusion_smoke.json`：
  - smoke 门固定：`episodes=150`、`runs=3`；
  - 关键改动：`train.cnn_backbone=globalcnn_fusion`；
  - 其余训练/推理参数与 `v7p2p2` 保持一致，保证单变量对比。

### 5) smoke 运行与留档更新
- 远端完成 `v7p2p3` smoke 训练+推理并回传 `runs/`。
- 新增 `docs/versions/v7p2p3/` 四件套并登记真实 run 路径、KPI 与 `failure_reason`。
- 同步更新 `docs/versions/README.md`、`README.md`、`README.zh-CN.md` 的版本索引与状态。

## 受影响文件清单
- `forest_vehicle_dqn/networks.py`
- `forest_vehicle_dqn/cli/train.py`
- `tests/test_globalcnn_network.py`
- `configs/repro_20260221_v7p2p3_globalcnn_fusion_smoke.json`
- `docs/versions/v7p2p3/README.md`
- `docs/versions/v7p2p3/CHANGES.md`
- `docs/versions/v7p2p3/RESULTS.md`
- `docs/versions/v7p2p3/runs/README.md`
- `docs/versions/README.md`
- `README.md`
- `README.zh-CN.md`
