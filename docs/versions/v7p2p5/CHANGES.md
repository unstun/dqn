# v7p2p5 改动清单（相对 v7p2p4 / v7p1）

## 变更目标
- 不引入“目标走廊先验”，仅在现有 `globalcnn_fusion` 上加入融合归一化，且从 `agents` 配置层直连控制。

## 代码/配置改动明细

### 1) GlobalCNN 融合归一化
- `forest_vehicle_dqn/networks.py`
  - `CNNQNetwork` 新增参数：
    - `globalcnn_fusion_layernorm`
    - `globalcnn_fusion_layernorm_eps`
  - 在 `globalcnn_fusion` 模式下，对融合向量执行可选 `LayerNorm`（门控前）。

### 2) Agent 配置贯通
- `forest_vehicle_dqn/agents.py`
  - `AgentConfig` 新增以上参数。
  - `DQNFamilyAgent` 在建网参数中透传新字段。
  - checkpoint `load()` 对缺失字段使用默认值补齐，保持旧模型兼容。

### 3) 训练 CLI 扩展
- `forest_vehicle_dqn/cli/train.py`
  - 新增：
    - `--cnn-fusion-layernorm/--no-cnn-fusion-layernorm`
    - `--cnn-fusion-layernorm-eps`
  - 参数注入 `AgentConfig` 并写入 run 配置。

### 4) 单测扩展
- `tests/test_globalcnn_network.py`
  - 扩展前向测试，覆盖 `globalcnn_fusion_layernorm` 配置。
  - 扩展 roundtrip，验证保存/加载后新参数不丢失。
  - 旧 checkpoint 兼容测试保持通过。

### 5) 新增配置
- `configs/v7p2p5.json`
- `configs/repro_20260221_v7p2p5_globalcnn_fusionnorm_smoke.json`

## 受影响文件清单
- `forest_vehicle_dqn/networks.py`
- `forest_vehicle_dqn/agents.py`
- `forest_vehicle_dqn/cli/train.py`
- `tests/test_globalcnn_network.py`
- `configs/v7p2p5.json`
- `configs/repro_20260221_v7p2p5_globalcnn_fusionnorm_smoke.json`
- `docs/versions/v7p2p5/README.md`
- `docs/versions/v7p2p5/CHANGES.md`
- `docs/versions/v7p2p5/RESULTS.md`
- `docs/versions/v7p2p5/runs/README.md`
- `docs/versions/README.md`
- `README.md`
- `README.zh-CN.md`
