# v6p2p2 runs 追溯

## 1. 本轮执行命令
- smoke 网格：
  - `conda run -n ros2py310 python scripts/sweep_v6p2p2_reward_grid.py --stage smoke --k-t-values 0.06,0.1,0.14 --k-delta-values 0.8,1.5,2.2`
- 20-run 复核：
  - `conda run -n ros2py310 python infer.py --profile repro_20260219_v6p2p2_reward_kt_kdelta_sweep_smoke --models runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p1_kd0p8_train/train_20260219_120601/models --runs 20 --out repro_20260219_v6p2p2_reward_sweep_kt0p1_kd0p8_infer20 --seed 9101`
  - `conda run -n ros2py310 python infer.py --profile repro_20260219_v6p2p2_reward_kt_kdelta_sweep_smoke --models runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p06_kd1p5_train/train_20260219_115647/models --runs 20 --out repro_20260219_v6p2p2_reward_sweep_kt0p06_kd1p5_infer20 --seed 9102`

## 2. 汇总文件
- smoke 最新汇总：`runs/repro_20260219_v6p2p2_reward_sweep/smoke_summary_latest.csv`
- smoke 时间戳汇总：`runs/repro_20260219_v6p2p2_reward_sweep/smoke_summary_20260219_123345.csv`

## 3. smoke 9 组 run 清单（全部可追溯）

| rank | k_t | k_delta | train_run_dir | train_run_json | infer_run_dir | infer_run_json | kpi_csv | success_rate | avg_path_length | avg_curvature_1_m | path_time_s |
|---:|---:|---:|---|---|---|---|---|---:|---:|---:|---:|
| 1 | 0.06 | 1.5 | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p06_kd1p5_train/train_20260219_115647` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p06_kd1p5_train/train_20260219_115647/configs/run.json` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p06_kd1p5_infer/20260219_120004` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p06_kd1p5_infer/20260219_120004/configs/run.json` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p06_kd1p5_infer/20260219_120004/table2_kpis_mean_raw.csv` | 0.6670 | 38.4131 | 0.166574 | 30.8375 |
| 2 | 0.06 | 2.2 | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p06_kd2p2_train/train_20260219_120103` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p06_kd2p2_train/train_20260219_120103/configs/run.json` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p06_kd2p2_infer/20260219_120521` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p06_kd2p2_infer/20260219_120521/configs/run.json` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p06_kd2p2_infer/20260219_120521/table2_kpis_mean_raw.csv` | 0.6665 | 43.0736 | 0.170700 | 36.6334 |
| 3 | 0.10 | 0.8 | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p1_kd0p8_train/train_20260219_120601` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p1_kd0p8_train/train_20260219_120601/configs/run.json` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p1_kd0p8_infer/20260219_121000` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p1_kd0p8_infer/20260219_121000/configs/run.json` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p1_kd0p8_infer/20260219_121000/table2_kpis_mean_raw.csv` | 0.5000 | 30.0408 | 0.078747 | 18.9625 |
| 4 | 0.14 | 0.8 | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p14_kd0p8_train/train_20260219_122009` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p14_kd0p8_train/train_20260219_122009/configs/run.json` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p14_kd0p8_infer/20260219_122333` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p14_kd0p8_infer/20260219_122333/configs/run.json` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p14_kd0p8_infer/20260219_122333/table2_kpis_mean_raw.csv` | 0.5000 | 40.9767 | 0.220658 | 26.6625 |
| 5 | 0.14 | 2.2 | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p14_kd2p2_train/train_20260219_122917` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p14_kd2p2_train/train_20260219_122917/configs/run.json` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p14_kd2p2_infer/20260219_123304` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p14_kd2p2_infer/20260219_123304/configs/run.json` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p14_kd2p2_infer/20260219_123304/table2_kpis_mean_raw.csv` | 0.3335 | 13.5430 | 0.193443 | 10.5250 |
| 6 | 0.14 | 1.5 | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p14_kd1p5_train/train_20260219_122440` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p14_kd1p5_train/train_20260219_122440/configs/run.json` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p14_kd1p5_infer/20260219_122821` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p14_kd1p5_infer/20260219_122821/configs/run.json` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p14_kd1p5_infer/20260219_122821/table2_kpis_mean_raw.csv` | 0.3335 | 15.3486 | 0.361034 | 12.7500 |
| 7 | 0.10 | 2.2 | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p1_kd2p2_train/train_20260219_121514` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p1_kd2p2_train/train_20260219_121514/configs/run.json` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p1_kd2p2_infer/20260219_121912` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p1_kd2p2_infer/20260219_121912/configs/run.json` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p1_kd2p2_infer/20260219_121912/table2_kpis_mean_raw.csv` | 0.3335 | 16.1448 | 0.161578 | 12.6500 |
| 8 | 0.10 | 1.5 | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p1_kd1p5_train/train_20260219_121042` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p1_kd1p5_train/train_20260219_121042/configs/run.json` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p1_kd1p5_infer/20260219_121356` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p1_kd1p5_infer/20260219_121356/configs/run.json` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p1_kd1p5_infer/20260219_121356/table2_kpis_mean_raw.csv` | 0.3330 | 37.5773 | 0.096391 | 24.3750 |
| 9 | 0.06 | 0.8 | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p06_kd0p8_train/train_20260219_115230` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p06_kd0p8_train/train_20260219_115230/configs/run.json` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p06_kd0p8_infer/20260219_115548` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p06_kd0p8_infer/20260219_115548/configs/run.json` | `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p06_kd0p8_infer/20260219_115548/table2_kpis_mean_raw.csv` | 0.3330 | 49.1152 | 0.265649 | 33.2250 |

## 4. 20-run 复核 run 清单

| k_t | k_delta | infer_run_dir | infer_run_json | kpi_csv | short (SR/len/curv/time) | long (SR/len/curv/time) | short+long mean (SR/len/curv/time) |
|---:|---:|---|---|---|---|---|---|
| 0.10 | 0.8 | `runs/repro_20260219_v6p2p2_reward_sweep_kt0p1_kd0p8_infer20/20260219_123433` | `runs/repro_20260219_v6p2p2_reward_sweep_kt0p1_kd0p8_infer20/20260219_123433/configs/run.json` | `runs/repro_20260219_v6p2p2_reward_sweep_kt0p1_kd0p8_infer20/20260219_123433/table2_kpis_mean_raw.csv` | `0.75 / 20.5956 / 0.232669 / 16.1633` | `0.55 / 59.1761 / 0.246381 / 42.8682` | `0.65 / 39.88585 / 0.239525 / 29.51575` |
| 0.06 | 1.5 | `runs/repro_20260219_v6p2p2_reward_sweep_kt0p06_kd1p5_infer20/20260219_123827` | `runs/repro_20260219_v6p2p2_reward_sweep_kt0p06_kd1p5_infer20/20260219_123827/configs/run.json` | `runs/repro_20260219_v6p2p2_reward_sweep_kt0p06_kd1p5_infer20/20260219_123827/table2_kpis_mean_raw.csv` | `0.45 / 20.8154 / 0.203479 / 17.0889` | `0.30 / 50.8996 / 0.176859 / 33.8083` | `0.375 / 35.8575 / 0.190169 / 25.4486` |

failure_reason 分布（CNN-DDQN）：
- `k_t=0.10, k_delta=0.8`
  - short：`reached=15`, `timeout=4`, `collision=1`
  - long：`reached=11`, `timeout=8`, `collision=1`
- `k_t=0.06, k_delta=1.5`
  - short：`reached=9`, `timeout=10`, `collision=1`
  - long：`reached=6`, `timeout=14`

## 5. 重试/调试运行（不纳入最终统计）
- 由于早期 `latest.txt` 解析使用了错误相对路径，首轮 `k_t=0.06, k_delta=0.8` 产出了一次重试：
  - `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p06_kd0p8_train/train_20260219_114635`
  - `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p06_kd0p8_infer_retry/20260219_115125`
- 修复后统一以 `runs/<out>/latest.txt` 所在目录解析相对路径，后续结果均写入 `smoke_summary_latest.csv`。
