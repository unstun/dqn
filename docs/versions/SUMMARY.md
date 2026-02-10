# 版本线总结

## 当前已观测最佳结果（正式严格集合）
- short 最佳成功率： `v3p1` (`short=0.7`).
- long 最佳成功率：`v3p1` 系列（`long=0.5` 来自 smoke final-ckpt 运行，见 `docs/versions/v3p1/runs/README.md`）。
- 目前尚无版本在 short/long 两套件上同时超过 Hybrid A*-MPC。

## 过程提醒
- 保持推理期口径与命名一致（`strict-argmax` 或 `shielded/masked/hybrid`）。
- 保持算法命名与实现定义严格一致。
- 保持 `self-check -> smoke -> full(runs=20)` 流程，并完成按版本目录留档。
