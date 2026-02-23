# v8p2 结果对比（待回填）

## 1. 数据来源

### 1) infer-only smoke（固定 v7p1 checkpoint）
- `dijkstra8_nocorner`：`N/A`
- `euclid`（对照）：`N/A`

### 2) train+infer smoke（episodes=150, runs=3）
- train：`N/A`
- infer：`N/A`

## 2. 代码级验证结果

### 最小自检
- `conda run -n ros2py310 python train.py --self-check`
- `conda run -n ros2py310 python infer.py --self-check`
- 结果：`PASS`（本地）

### 单元测试
- `conda run -n ros2py310 python -m pytest -q`
- 结果：`PASS`（本地）

## 3. short/mid/long 指标（infer-only）
- `N/A`

## 4. short/mid/long 指标（train+infer smoke）
- `N/A`

## 5. 门槛检查（smoke）
- `N/A`

## 6. 结论（go/no-go）
- `N/A`

