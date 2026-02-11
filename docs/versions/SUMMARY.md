# 版本线总结

## 当前已观测最佳结果（正式严格集合）
- short 最佳成功率： `v3p1` (`short=0.7`).
- long 最佳成功率：`v3p1` 系列（`long=0.5` 来自 smoke final-ckpt 运行，见 `docs/versions/v3p1/runs/README.md`）。
- 目前尚无版本在 short/long 两套件上同时超过 Hybrid A*-MPC。

## 口径变化说明（strict no-fallback → strict-argmax）

- 自 `v3p12`（`2026-02-09/10`）起，推理侧 `--forest-no-fallback` 被修正为**严格的 strict-argmax**：
  - 推理阶段仅执行 `argmax(Q)`；
  - 不允许使用 admissible-mask/top-k 重选/stop override/replacement/fallback 等“干预”替换动作。
- 因此，`v3p12` 之前的历史结果若未显式记录该开关及其语义，可能与 strict-argmax **不可直接横向可比**。
- 建议做法：
  - 所有对比实验在命令中显式指定 `--forest-no-fallback`（strict-argmax）或 `--no-forest-no-fallback`（hybrid/shielded）。
  - 使用固定随机样本（`--rand-pairs-json`）避免 random pair 漂移造成的“看起来回退/进步”误判。

## 过程提醒
- 保持推理期口径与命名一致（`strict-argmax` 或 `shielded/masked/hybrid`）。
- 保持算法命名与实现定义严格一致。
- 保持 `self-check -> smoke -> full(runs=20)` 流程，并完成按版本目录留档。
