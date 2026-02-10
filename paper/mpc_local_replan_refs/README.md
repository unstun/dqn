# MPC local replanning reference (selected)

- Title: Automatic parking planning control method based on improved A* algorithm
- arXiv: https://arxiv.org/abs/2406.15429
- PDF: `paper/mpc_local_replan_refs/2406.15429_improved_Astar_MPC_autoparking.pdf`
- Why selected: the method follows the target architecture used in this repo update:
  - global A*-family path as topology guidance
  - local MPC-based trajectory control under constraints
  - planning-control coupling for local-map scenarios

## Scope used in this repo

This repository adopts the "global topology + local MPC replanning/tracking" design intent,
implemented as a pure MPC (OSQP QP) local planner/tracker for forest baselines and `astar_mpc` expert.

## Upstream reference + clean-room note

- Upstream structure reference repository:
  - `rst-tu-dortmund/mpc_local_planner`
  - commit: `5b4e465fec718245484e8668e6dd45539ca37983`
- This repo's `mpc2` implementation is a clean-room Python reimplementation aligned to method intent and
  local interfaces (forest bicycle model + two-circle collision), and does **not** copy GPL source text.
