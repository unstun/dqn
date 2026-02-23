# v8p3 结果对比（smoke）

## 1. 数据来源（待回填）

- smoke（episodes=150, runs=3）：N/A
- 回归（固定 short collision pair，runs=1）：N/A

## 2. 代码级验证结果

### 最小自检
- `conda run -n ros2py310 python train.py --self-check`
- `conda run -n ros2py310 python infer.py --self-check`
- 结果：`PASS`（本地）

### 单元测试
- `conda run -n ros2py310 python -m pytest -q`
- 结果：`PASS`（本地）

## 3. 指标（待回填）

### short/mid/long（runs=3，mean）
- N/A

### `failure_reason` 分布
- N/A

## 4. 门槛检查（smoke，待回填）
- N/A

## 5. 结论（go/no-go，待回填）
- N/A

