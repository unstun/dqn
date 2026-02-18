# RL planning paper pack

This folder is a curated, **open-access** PDF pack to support writing and debugging RL-based motion planning in this repo (forest bicycle + planner/MPC baselines).

Files:
- `manifest.json` (metadata + tags + 1-line relevance notes; includes abstracts)
- `refs.bib` (BibTeX entries for the PDFs in this folder)

Related in-repo refs:
- `paper/hybrid_a_star_pdf/FULLTEXT01.pdf` (Hybrid A* reference PDF already in repo)
- `paper/mpc_local_replan_refs/2406.15429_improved_Astar_MPC_autoparking.pdf` (A*+MPC recent ref already in repo)
- `paper/dqfd_refs/` (DQfD bib/notes already in repo)

## Classical planning / baselines

- `arxiv_1105.1186_rrt_star_optimal_motion_planning.pdf` (2011) Sampling-based Algorithms for Optimal Motion Planning — RRT* 的原始最优性证明与对比基线口径（采样规划）。

## DQN-family (foundation)

- `arxiv_1312.5602_dqn_atari.pdf` (2013) Playing Atari with Deep Reinforcement Learning — DQN 的核心组件：replay + target network（你当前 CNN-DDQN 的理论起点）。
- `arxiv_1509.06461_double_dqn.pdf` (2015) Deep Reinforcement Learning with Double Q-learning — Double Q-learning 缓解 Q 过估计（对应“DDQN”的学术定义）。
- `arxiv_1511.05952_prioritized_replay.pdf` (2015) Prioritized Experience Replay — Prioritized replay 提升样本效率（对稀疏成功/失败样本更关键）。
- `arxiv_1511.06581_dueling_dqn.pdf` (2015) Dueling Network Architectures for Deep Reinforcement Learning — Dueling 结构将 V/A 分离（在导航/规划这类稠密状态上常更稳）。
- `arxiv_1704.03732_dqfd.pdf` (2017) Deep Q-learning from Demonstrations — DQfD：用专家演示加速并稳定 DQN 学习（非常适合“追强基线”的现实需求）。
- `arxiv_1710.02298_rainbow.pdf` (2017) Rainbow: Combining Improvements in Deep Reinforcement Learning — Rainbow：DQN trick 的组合基线（写 Related Work 时可解释你启用/未启用哪些组件）。

## Shielding / shields (inference-time safety interventions)

- `arxiv_1708.08611_safe_rl_via_shielding.pdf` (2017) Safe Reinforcement Learning via Shielding — Shielding：在策略外加“安全盾”过滤/替换动作（对应你的 `shielded/masked/hybrid` 口径）。
- `arxiv_2112.11490_do_androids_dream_of_electric_fences_safety_aware_reinforcement_learni.pdf` (2021) Do Androids Dream of Electric Fences? Safety-Aware Reinforcement Learning with Latent Shielding — Latent shielding：在隐空间学习安全约束（可作为“学习到的 mask/屏蔽器”参考）。
- `arxiv_2204.00755_shielding_under_partial_observability.pdf` (2022) Safe Reinforcement Learning via Shielding under Partial Observability — 部分可观测下的 shielding：把不确定性也纳入安全干预分析。
- `arxiv_2207.13446_dynamic_shielding_for_reinforcement_learning_in_black_box_environments.pdf` (2022) Dynamic Shielding for Reinforcement Learning in Black-Box Environments — Dynamic shielding：黑箱环境下自适应安全盾（强调无需精确动力学模型）。
- `arxiv_2212.01861_online_shielding_for_reinforcement_learning.pdf` (2022) Online Shielding for Reinforcement Learning — Online shielding：盾可在线更新（适合“训练中/部署中逐步完善规则”）。
- `arxiv_2303.03226_safe_reinforcement_learning_via_probabilistic_logic_shields.pdf` (2023) Safe Reinforcement Learning via Probabilistic Logic Shields — Probabilistic logic shields：用概率逻辑表达/组合安全规则（利于论文写清楚）。
- `arxiv_2308.00707_approximate_model_based_shielding_for_safe_reinforcement_learning.pdf` (2023) Approximate Model-Based Shielding for Safe Reinforcement Learning — 近似模型的 shielding：用近似模型降低盾开销（适合实时推理）。
- `arxiv_2308.14424_shielded_reinforcement_learning_for_hybrid_systems.pdf` (2023) Shielded Reinforcement Learning for Hybrid Systems — Hybrid systems 的 shielded RL：离散-连续混合系统下的安全干预（更贴近车辆控制）。
- `arxiv_2406.06507_verification_guided_shielding_for_deep_reinforcement_learning.pdf` (2024) Verification-Guided Shielding for Deep Reinforcement Learning — Verification-guided shielding：用形式化/验证信息构造盾（适合占据图+离散动作空间）。

## Predictive safety filters (safety layer / minimal intervention)

- `arxiv_1812.05506_a_predictive_safety_filter_for_learning_based_control_of_constrained_n.pdf` (2018) A predictive safety filter for learning-based control of constrained nonlinear dynamical systems — Predictive Safety Filter：把约束满足做成模块化安全过滤层（可视为 MPC/QP 风格的外层）。
- `arxiv_2301.00884_safety_filtering_for_reinforcement_learning_based_adaptive_cruise_cont.pdf` (2023) Safety Filtering for Reinforcement Learning-based Adaptive Cruise Control — ACC 场景的 safety filtering：展示“RL 决策 + 外层安全修正”的工程范式。
- `arxiv_2306.02551_conformal_predictive_safety_filter_for_rl_controllers_in_dynamic_envir.pdf` (2023) Conformal Predictive Safety Filter for RL Controllers in Dynamic Environments — Conformal safety filter：给过滤器引入统计校准/置信度口径（适合“长尾失败”讨论）。
- `arxiv_2410.11671_safety_filtering_while_training_improving_the_performance_and_sample_e.pdf` (2024) Safety Filtering While Training: Improving the Performance and Sample Efficiency of Reinforcement Learning Agents — Training 中引入 safety filter 可提升样本效率与最终性能（“训练期就带盾”）。
- `arxiv_2506.22894_safe_reinforcement_learning_with_a_predictive_safety_filter_for_motion.pdf` (2025) Safe Reinforcement Learning with a Predictive Safety Filter for Motion Planning and Control: A Drifting Vehicle Example — 漂移车辆案例：用预测安全过滤器把 RL 用于规划+控制且保持安全（车辆运动学贴近）。

## CBF/CLF-based safety (barrier functions / stability + safety)

- `arxiv_1903.09885_temporal_logic_guided_safe_reinforcement_learning_using_control_barrie.pdf` (2019) Temporal Logic Guided Safe Reinforcement Learning Using Control Barrier Functions — Temporal-logic + CBF：把安全规格写成可验证的逻辑约束，再用 CBF 落地。
- `arxiv_2004.07584_reinforcement_learning_for_safety_critical_control_under_model_uncerta.pdf` (2020) Reinforcement Learning for Safety-Critical Control under Model Uncertainty, using Control Lyapunov Functions and Control Barrier Functions — CLF/CBF + RL：统一稳定性与安全性（适合解释“到点必须停稳/回正”这类终止条件）。
- `arxiv_2103.01556_model_based_constrained_reinforcement_learning_using_generalized_contr.pdf` (2021) Model-based Constrained Reinforcement Learning using Generalized Control Barrier Function — Generalized CBF + model-based constrained RL：约束 RL 的一条清晰路线（可做方法对照）。
- `arxiv_2110.05415_safe_reinforcement_learning_using_robust_control_barrier_functions.pdf` (2021) Safe Reinforcement Learning Using Robust Control Barrier Functions — Robust CBF：把模型误差/扰动显式纳入安全保证（适合解释“long suite”退化）。
- `arxiv_2404.16879_learning_control_barrier_functions_and_their_application_in_reinforcem.pdf` (2024) Learning Control Barrier Functions and their application in Reinforcement Learning: A Survey — CBF+RL 综述：系统整理 barrier function 如何接入 RL（写 Related Work 最省力）。
- `arxiv_2510.14959_cbf_rl_safety_filtering_reinforcement_learning_in_training_with_contro.pdf` (2025) CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions — CBF-RL：用 Control Barrier Function 做 training-time safety filter（对连续控制很典型）。

## MPC + RL hybrids (safe exploration / constrained execution)

- `arxiv_1906.12189_learning_based_model_predictive_control_for_safe_exploration_and_reinf.pdf` (2019) Learning-based Model Predictive Control for Safe Exploration and Reinforcement Learning — Learning-based MPC：用 MPC 支持安全探索/训练（可视为“训练期的强盾+数据采集器”）。
- `arxiv_1908.00177_learning_when_to_drive_in_intersections_by_combining_reinforcement_lea.pdf` (2019) Learning When to Drive in Intersections by Combining Reinforcement Learning and Model Predictive Control — 交叉口驾驶：RL 决策 + MPC 约束执行/安全（“高层策略+低层安全”范式）。
- `arxiv_2102.11122_reinforcement_learning_of_the_prediction_horizon_in_model_predictive_c.pdf` (2021) Reinforcement Learning of the Prediction Horizon in Model Predictive Control — 用 RL 学 MPC 预测时域（horizon）等超参：把“调 MPC”变成学习问题。
- `arxiv_2112.13941_safe_reinforcement_learning_with_chance_constrained_model_predictive_c.pdf` (2021) Safe Reinforcement Learning with Chance-constrained Model Predictive Control — Chance-constrained MPC + RL：把不确定性用概率约束写进 MPC 安全层。

## Residual RL (controller/planner as nominal)

- `arxiv_2106.08050_residual_reinforcement_learning_from_demonstrations.pdf` (2021) Residual Reinforcement Learning from Demonstrations — Residual RL + demos：以专家/控制器为 nominal，RL 学 residual（很适合“planner/MPC 作为底座”）。

## RL for motion planning under dynamics (kinodynamic / sampling-based)

- `arxiv_1907.04799_rl_rrt_kinodynamic_motion_planning_via_learning_reachability_estimator.pdf` (2019) RL-RRT: Kinodynamic Motion Planning via Learning Reachability Estimators from RL Policies — RL-RRT：用 RL 学可达性估计器来加速 kinodynamic 规划（贴近自行车动力学）。
- `arxiv_2510.10567_reinforcement_learning_based_dynamic_adaptation_for_sampling_based_mot.pdf` (2025) Reinforcement Learning-based Dynamic Adaptation for Sampling-Based Motion Planning in Agile Autonomous Driving — 用 RL 动态调参采样规划器（agile driving 场景；可迁移到“learned sampling/预算分配”）。

## Learning-augmented motion planning (PRM/RRT*/MPNet/MPC)

- `arxiv_1707.03034_learning_heuristic_search_via_imitation.pdf` (2017) Learning Heuristic Search via Imitation — 用模仿学习学搜索策略/启发式（往往比纯 RL 更可复现、也更容易赢时间）。
- `arxiv_1709.05448_learning_sampling_distributions_motion_planning.pdf` (2017) Learning Sampling Distributions for Robot Motion Planning — 学习 RRT 的采样分布：提高窄通道/复杂障碍下的采样效率。
- `arxiv_1710.03937_prm_rl.pdf` (2017) PRM-RL: Long-range Robotic Navigation Tasks by Combining Reinforcement Learning and Sampling-based Planning — PRM-RL：全局 PRM + 局部 RL 控制器（RL 负责动力学可行性/鲁棒性叙事）。
- `arxiv_1806.01968_learning_implicit_sampling_distributions_motion_planning.pdf` (2018) Learning Implicit Sampling Distributions for Motion Planning — 隐式采样分布：把“采样更像在走可行通道里”做成可学习模块。
- `arxiv_1806.05767_mpnet_motion_planning_networks.pdf` (2018) Motion Planning Networks — MPNet：学习到可行路径/连接器，加速传统规划管线。
- `arxiv_2101.06798_mpc_mpnet.pdf` (2021) MPC-MPNet: Model-Predictive Motion Planning Networks for Fast, Near-Optimal Planning under Kinodynamic Constraints — MPC-MPNet：学习规划 + MPC（与你仓库 Hybrid A*/MPC baseline 结构高度相似）。
- `arxiv_2411.17293_sil_rrt_star.pdf` (2024) SIL-RRT*: Learning Sampling Distribution through Self Imitation Learning — SIL-RRT*：自模仿提升 RRT* 连接/采样（learned sampling 的新线）。

## Neural / differentiable planning (VIN/GPPN/Neural A*)

- `arxiv_1602.02867_value_iteration_networks.pdf` (2016) Value Iteration Networks — VIN：把 value iteration 变成可学习模块（占据图→规划 的典型叙事）。
- `arxiv_1806.06408_gated_path_planning_networks.pdf` (2018) Gated Path Planning Networks — GPPN：RNN 形式的可微规划网络（常被用来讨论 VIN 的训练稳定性）。
- `arxiv_2009.07476_neural_a_star_search.pdf` (2020) Path Planning using Neural A* Search — Neural A*：把 A* 融入可微框架（适合“学习启发式/代价图”路线）。
- `arxiv_2105.01480_neural_weighted_a_star.pdf` (2021) Neural Weighted A*: Learning Graph Costs and Heuristics with Differentiable Anytime A* — Neural Weighted A*：学习代价/启发式并用加权 A* 做 anytime 折中。
- `arxiv_2209.05206_diff_loss_learning_heuristics_astar.pdf` (2022) A Differentiable Loss Function for Learning Heuristics in A* — 可微损失学启发式：直接优化搜索轨迹/扩展节点（对齐你想赢的时间指标）。
- `arxiv_2509.22626_learning_admissible_heuristics_astar.pdf` (2025) Learning Admissible Heuristics for A*: Theory and Practice — 学习可采纳（admissible）启发式：把“更快但仍保守”的口径写扎实。

## Safe autonomous driving / offline RL / toolkits

- `arxiv_1902.04118_wisemove_a_framework_for_safe_deep_reinforcement_learning_for_autonomo.pdf` (2019) WiseMove: A Framework for Safe Deep Reinforcement Learning for Autonomous Driving — WiseMove：安全深度 RL 框架（可借鉴安全约束/舒适性指标口径）。
- `arxiv_2110.07067_offline_reinforcement_learning_for_autonomous_driving_with_safety_and.pdf` (2021) Offline Reinforcement Learning for Autonomous Driving with Safety and Exploration Enhancement — 自动驾驶 offline RL：用离线数据训练并强化安全/探索（适合“先用 baseline 造数据”）。
- `arxiv_2206.08528_saferl_kit_evaluating_efficient_reinforcement_learning_methods_for_saf.pdf` (2022) SafeRL-Kit: Evaluating Efficient Reinforcement Learning Methods for Safe Autonomous Driving — SafeRL-Kit：安全自动驾驶 RL 的评测工具/基准（可参考指标与场景划分）。
- `arxiv_2406.08878_cimrl_combining_imitation_and_reinforcement_learning_for_safe_autonomo.pdf` (2024) CIMRL: Combining IMitation and Reinforcement Learning for Safe Autonomous Driving — IM+RL 混合（CIMRL）：把 imitation 与 RL 结合来提升安全性与收敛（贴近你现有 demo 体系）。

## Action masking caveats

- `arxiv_2006.14171_a_closer_look_at_invalid_action_masking_in_policy_gradient_algorithms.pdf` (2020) A Closer Look at Invalid Action Masking in Policy Gradient Algorithms — 无效 action masking 会改变策略梯度并引入偏差（写论文时要说明 mask 的数学口径）。

## RL navigation / collision avoidance (policy-level)

- `arxiv_1609.07845_multiagent_collision_avoidance_drl.pdf` (2016) Decentralized Non-communicating Multiagent Collision Avoidance with Deep Reinforcement Learning — 多智能体避碰 DRL：强调反应式策略与低延迟（可写“实时性/动态障碍”优势来源）。
- `arxiv_1611.03673_learning_to_navigate_complex_environments.pdf` (2016) Learning to Navigate in Complex Environments — 复杂环境导航 DRL：端到端/记忆模块对长时导航的价值（Related Work）。
- `arxiv_1709.10082_decentralized_multirobot_collision_avoidance.pdf` (2017) Towards Optimally Decentralized Multi-Robot Collision Avoidance via Deep Reinforcement Learning — 去中心化多机器人避碰：同样强调实时与泛化（可写“RL vs 规划”优势）。

## Safe RL with explicit constraints (e.g., CPO)

- `arxiv_1705.10528_constrained_policy_optimization.pdf` (2017) Constrained Policy Optimization — CPO：约束强化学习的经典算法（把碰撞/越界写成约束的 baseline 参考）。
