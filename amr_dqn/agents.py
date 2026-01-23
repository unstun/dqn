from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from amr_dqn.networks import DuelingQNetwork, QNetwork
from amr_dqn.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from amr_dqn.schedules import adaptive_epsilon, linear_epsilon


@dataclass(frozen=True)
class AgentConfig:
    # Forest (dt=0.05s, long horizon) needs a higher discount factor; gamma=0.9 makes
    # the terminal reward effectively vanish after a few dozen steps.
    gamma: float = 0.995
    n_step: int = 1
    learning_rate: float = 5e-4
    replay_capacity: int = 100_000
    batch_size: int = 128
    target_update_steps: int = 1000
    target_tau: float = 0.0
    grad_clip_norm: float = 10.0

    eps_start: float = 0.9
    eps_final: float = 0.01
    eps_decay: int = 2000

    hidden_layers: int = 3
    hidden_dim: int = 256

    # Prioritized replay (IDDQN only). Default off for stability on static global-planning maps.
    per_alpha: float = 0.0
    per_beta_start: float = 0.4
    per_beta_steps: int = 50_000

    # Expert margin loss (DQfD-style) for forest stabilization.
    demo_margin: float = 0.8
    demo_lambda: float = 1.0
    demo_ce_lambda: float = 1.0


class DQNFamilyAgent:
    def __init__(
        self,
        algo: Literal["dqn", "ddqn", "iddqn"],
        obs_dim: int,
        n_actions: int,
        *,
        config: AgentConfig,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.algo = algo
        self.config = config
        self.device = torch.device(device)

        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # DQN is the baseline (plain Q-learning).
        # DDQN keeps the same architecture but uses the Double DQN TD target (online argmax + target eval).
        # IDDQN adds extra stabilizers (dueling head + optional PER + soft target updates) for long-horizon forests.
        net_cls = DuelingQNetwork if algo == "iddqn" else QNetwork
        self.q = net_cls(obs_dim, n_actions, hidden_dim=config.hidden_dim, hidden_layers=config.hidden_layers).to(
            self.device
        )
        self.q_target = net_cls(obs_dim, n_actions, hidden_dim=config.hidden_dim, hidden_layers=config.hidden_layers).to(
            self.device
        )
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        self._use_per = bool(algo == "iddqn" and float(config.per_alpha) > 0.0)
        if self._use_per:
            self.replay = PrioritizedReplayBuffer(
                config.replay_capacity,
                obs_dim,
                n_actions,
                rng=self._rng,
                alpha=float(config.per_alpha),
            )
        else:
            self.replay = ReplayBuffer(config.replay_capacity, obs_dim, n_actions, rng=self._rng)

        self._train_steps = 0
        self._n_actions = int(n_actions)
        self._obs_dim = int(obs_dim)
        self._n_step = int(max(1, int(getattr(config, "n_step", 1))))
        # (obs, action, reward, next_obs, done, demo, next_action_mask)
        self._nstep_buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool, bool, np.ndarray | None]] = deque()

    def _rebuild_networks(
        self,
        net_cls: type[nn.Module],
        *,
        hidden_dim: int,
        hidden_layers: int,
    ) -> None:
        self.q = net_cls(self._obs_dim, self._n_actions, hidden_dim=hidden_dim, hidden_layers=hidden_layers).to(self.device)
        self.q_target = net_cls(self._obs_dim, self._n_actions, hidden_dim=hidden_dim, hidden_layers=hidden_layers).to(
            self.device
        )
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.config.learning_rate)

    def epsilon(self, episode: int) -> float:
        if self.algo == "iddqn":
            # The logistic schedule from the paper starts at ~0.5*eps_start (k=0),
            # which is often too little exploration for sparse-success navigation.
            # Use the same linear decay as DQN for stability.
            return linear_epsilon(
                episode,
                eps_start=self.config.eps_start,
                eps_final=self.config.eps_final,
                decay_episodes=self.config.eps_decay,
            )
        return linear_epsilon(
            episode,
            eps_start=self.config.eps_start,
            eps_final=self.config.eps_final,
            decay_episodes=self.config.eps_decay,
        )

    def act(self, obs: np.ndarray, *, episode: int, explore: bool = True) -> int:
        if explore and (self._rng.random() < self.epsilon(episode)):
            return int(self._rng.integers(0, self._n_actions))

        with torch.no_grad():
            x = torch.from_numpy(obs.astype(np.float32, copy=False)).to(self.device)
            q = self.q(x.unsqueeze(0)).squeeze(0)
            return int(torch.argmax(q).item())

    def act_masked(
        self,
        obs: np.ndarray,
        *,
        episode: int,
        explore: bool = True,
        action_mask: np.ndarray | None = None,
    ) -> int:
        """Epsilon-greedy action selection with an optional boolean action mask."""

        mask = None
        if action_mask is not None:
            mask = np.asarray(action_mask, dtype=bool).reshape(-1)
            if mask.size != self._n_actions:
                raise ValueError("action_mask must have shape (n_actions,)")

        if explore and (self._rng.random() < self.epsilon(episode)):
            if mask is None:
                return int(self._rng.integers(0, self._n_actions))
            idxs = np.nonzero(mask)[0]
            if idxs.size == 0:
                return int(self._rng.integers(0, self._n_actions))
            return int(self._rng.choice(idxs))

        with torch.no_grad():
            x = torch.from_numpy(obs.astype(np.float32, copy=False)).to(self.device)
            q = self.q(x.unsqueeze(0)).squeeze(0)
            if mask is not None:
                q = q.clone()
                q[torch.from_numpy(~mask).to(self.device)] = torch.finfo(q.dtype).min
            return int(torch.argmax(q).item())

    def top_actions(self, obs: np.ndarray, *, k: int) -> np.ndarray:
        """Return the top-k action indices by Q-value (descending)."""
        kk = int(max(1, int(k)))
        with torch.no_grad():
            x = torch.from_numpy(obs.astype(np.float32, copy=False)).to(self.device)
            q = self.q(x.unsqueeze(0)).squeeze(0)
            kk = int(min(int(kk), int(q.numel())))
            return torch.topk(q, k=kk, dim=0).indices.detach().cpu().numpy()

    def _add_to_replay(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        *,
        next_action_mask: np.ndarray | None,
        demo: bool,
        n_steps: int,
    ) -> None:
        if self._use_per:
            self.replay.add(
                obs,
                int(action),
                float(reward),
                next_obs,
                bool(done),
                next_action_mask=next_action_mask,
                demo=bool(demo),
                n_steps=int(n_steps),
            )
        else:
            self.replay.add(
                obs,
                int(action),
                float(reward),
                next_obs,
                bool(done),
                next_action_mask=next_action_mask,
                demo=bool(demo),
                n_steps=int(n_steps),
            )

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        *,
        demo: bool = False,
        truncated: bool = False,
        next_action_mask: np.ndarray | None = None,
    ) -> None:
        """Record a transition (supports n-step returns).

        `done` should reflect *true* terminal states (collision/reached). Use `truncated=True`
        for time-limit episode ends so the n-step buffer does not leak across episodes while
        still allowing bootstrapping from the final state.
        """

        if int(self._n_step) <= 1:
            self._add_to_replay(
                obs,
                int(action),
                float(reward),
                next_obs,
                bool(done),
                next_action_mask=next_action_mask,
                demo=bool(demo),
                n_steps=1,
            )
            if bool(done) or bool(truncated):
                self.end_episode()
            return

        self._nstep_buffer.append((obs, int(action), float(reward), next_obs, bool(done), bool(demo), next_action_mask))

        episode_end = bool(done) or bool(truncated)
        if (len(self._nstep_buffer) < int(self._n_step)) and not episode_end:
            return

        self._flush_nstep_buffer(force=episode_end)

    def _flush_nstep_buffer(self, *, force: bool) -> None:
        if int(self._n_step) <= 1:
            return

        gamma = float(self.config.gamma)
        n_target = int(self._n_step)

        # When `force` is True (episode boundary), flush everything with truncated n-step horizons.
        while self._nstep_buffer:
            horizon = min(int(n_target), int(len(self._nstep_buffer)))
            ret = 0.0
            n_used = 0
            next_obs_n = self._nstep_buffer[0][3]
            next_mask_n = self._nstep_buffer[0][6]
            done_n = False
            for i in range(horizon):
                _o, _a, r, no, d, _demo, nm = self._nstep_buffer[i]
                ret += (gamma**i) * float(r)
                n_used = i + 1
                next_obs_n = no
                next_mask_n = nm
                done_n = bool(d)
                if done_n:
                    break

            obs0, a0, _r0, _no0, _d0, demo0, _nm0 = self._nstep_buffer[0]
            self._add_to_replay(
                obs0,
                int(a0),
                float(ret),
                next_obs_n,
                bool(done_n),
                next_action_mask=next_mask_n,
                demo=bool(demo0),
                n_steps=int(n_used),
            )
            self._nstep_buffer.popleft()

            # If we're not at an episode boundary, only emit one transition per step.
            if not bool(force):
                break

    def end_episode(self) -> None:
        """Flush any pending n-step transitions at an episode boundary."""
        if int(self._n_step) <= 1:
            return
        self._flush_nstep_buffer(force=True)

    def pretrain_on_demos(self, *, steps: int) -> int:
        """Supervised warm-start on demonstration transitions (DQfD-style stabilizer)."""

        n_steps = int(steps)
        if n_steps <= 0:
            return 0
        if len(self.replay) < int(self.config.batch_size):
            return 0

        demo_lambda = float(getattr(self.config, "demo_lambda", 0.0))
        demo_margin = float(getattr(self.config, "demo_margin", 0.0))
        demo_ce_lambda = float(getattr(self.config, "demo_ce_lambda", 0.0))

        trained = 0
        for _ in range(n_steps):
            if len(self.replay) < int(self.config.batch_size):
                break

            if self._use_per:
                batch = self.replay.sample(self.config.batch_size, beta=1.0)
                obs = torch.from_numpy(batch.obs).to(self.device)
                actions = torch.from_numpy(batch.actions).to(self.device)
                demos = torch.from_numpy(batch.demos).to(self.device)
            else:
                batch = self.replay.sample(self.config.batch_size)
                obs = torch.from_numpy(batch.obs).to(self.device)
                actions = torch.from_numpy(batch.actions).to(self.device)
                demos = torch.from_numpy(batch.demos).to(self.device)

            demo_mask = demos.float().clamp(0.0, 1.0) > 0.0
            if not bool(torch.any(demo_mask)):
                continue

            q_all = self.q(obs)
            q_demo = q_all[demo_mask]
            a_demo = actions.long()[demo_mask]

            loss = torch.tensor(0.0, device=self.device)
            if demo_ce_lambda > 0.0:
                loss = loss + float(demo_ce_lambda) * F.cross_entropy(q_demo, a_demo, reduction="mean")

            if demo_lambda > 0.0 and demo_margin > 0.0:
                q_a = q_demo.gather(1, a_demo.view(-1, 1)).squeeze(1)
                q_other = q_demo.clone()
                q_other.scatter_(1, a_demo.view(-1, 1), torch.finfo(q_other.dtype).min)
                q_max_other = q_other.max(dim=1).values
                margin = torch.relu(q_max_other + float(demo_margin) - q_a).mean()
                loss = loss + float(demo_lambda) * margin

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=self.config.grad_clip_norm)
            self.optimizer.step()
            trained += 1

        if trained > 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return int(trained)

    def update(self) -> dict[str, float]:
        if len(self.replay) < self.config.batch_size:
            return {}

        if self._use_per:
            beta_steps = max(1, int(self.config.per_beta_steps))
            beta = float(self.config.per_beta_start) + (1.0 - float(self.config.per_beta_start)) * float(
                min(1.0, float(self._train_steps) / float(beta_steps))
            )
            batch = self.replay.sample(self.config.batch_size, beta=beta)
            obs = torch.from_numpy(batch.obs).to(self.device)
            actions = torch.from_numpy(batch.actions).to(self.device)
            rewards = torch.from_numpy(batch.rewards).to(self.device)
            next_obs = torch.from_numpy(batch.next_obs).to(self.device)
            next_action_masks = torch.from_numpy(batch.next_action_masks).to(self.device)
            dones = torch.from_numpy(batch.dones).to(self.device)
            n_steps = torch.from_numpy(batch.n_steps).to(self.device)
            weights = torch.from_numpy(batch.weights).to(self.device)
            demos = torch.from_numpy(batch.demos).to(self.device)
        else:
            batch = self.replay.sample(self.config.batch_size)
            obs = torch.from_numpy(batch.obs).to(self.device)
            actions = torch.from_numpy(batch.actions).to(self.device)
            rewards = torch.from_numpy(batch.rewards).to(self.device)
            next_obs = torch.from_numpy(batch.next_obs).to(self.device)
            next_action_masks = torch.from_numpy(batch.next_action_masks).to(self.device)
            dones = torch.from_numpy(batch.dones).to(self.device)
            n_steps = torch.from_numpy(batch.n_steps).to(self.device)
            weights = None
            demos = torch.from_numpy(batch.demos).to(self.device)

        q_all = self.q(obs)
        q_values = q_all.gather(1, actions.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            mask = next_action_masks.to(torch.bool)
            if self.algo in ("iddqn", "ddqn"):
                # Double DQN target: action selection with online network, evaluation with target network.
                q_next_online = self.q(next_obs)
                q_next_online = q_next_online.masked_fill(~mask, torch.finfo(q_next_online.dtype).min)
                next_actions = torch.argmax(q_next_online, dim=1, keepdim=True)

                q_next_target = self.q_target(next_obs)
                q_next_target = q_next_target.masked_fill(~mask, torch.finfo(q_next_target.dtype).min)
                next_q = q_next_target.gather(1, next_actions).squeeze(1)
            else:
                # Vanilla DQN target: max over target network.
                q_next_target = self.q_target(next_obs)
                q_next_target = q_next_target.masked_fill(~mask, torch.finfo(q_next_target.dtype).min)
                next_q = q_next_target.max(dim=1).values

            # Safety: avoid propagating NaNs/-inf when a mask is malformed.
            next_q = torch.where(torch.isfinite(next_q), next_q, torch.zeros_like(next_q))
            gamma = float(self.config.gamma)
            gamma_n = torch.pow(torch.tensor(gamma, device=self.device, dtype=torch.float32), n_steps.to(torch.float32))
            target = rewards + (1.0 - dones) * (gamma_n * next_q)

        td_error = target - q_values
        losses = self.loss_fn(q_values, target)
        td_loss = (losses * weights).mean() if weights is not None else losses.mean()

        # Expert large-margin loss (DQfD). Only applied to `demo` transitions.
        demo_lambda = float(getattr(self.config, "demo_lambda", 0.0))
        demo_margin = float(getattr(self.config, "demo_margin", 0.0))
        margin_loss = torch.tensor(0.0, device=self.device)
        if demo_lambda > 0.0 and demo_margin > 0.0:
            demo_mask = demos.float().clamp(0.0, 1.0)
            if torch.any(demo_mask > 0.0):
                q_other = q_all.clone()
                q_other.scatter_(1, actions.view(-1, 1), torch.finfo(q_other.dtype).min)
                q_max_other = q_other.max(dim=1).values
                margin = torch.relu(q_max_other + float(demo_margin) - q_values)
                if weights is not None:
                    margin_loss = (margin * demo_mask * weights).mean()
                else:
                    margin_loss = (margin * demo_mask).mean()

        # Expert behavior cloning loss on demo transitions (works as a strong stabilizer on static maps).
        demo_ce_lambda = float(getattr(self.config, "demo_ce_lambda", 0.0))
        ce_loss = torch.tensor(0.0, device=self.device)
        if demo_ce_lambda > 0.0:
            demo_mask = demos.float().clamp(0.0, 1.0)
            if torch.any(demo_mask > 0.0):
                ce = F.cross_entropy(q_all, actions.long(), reduction="none")
                if weights is not None:
                    ce_loss = (ce * demo_mask * weights).mean()
                else:
                    ce_loss = (ce * demo_mask).mean()

        loss = td_loss + float(demo_lambda) * margin_loss + float(demo_ce_lambda) * ce_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=self.config.grad_clip_norm)
        self.optimizer.step()

        if self._use_per:
            self.replay.update_priorities(
                batch.idxs,
                td_errors=td_error.detach().abs().cpu().numpy(),
            )

        self._train_steps += 1
        tau = float(np.clip(float(self.config.target_tau), 0.0, 1.0))
        if self.algo == "iddqn" and tau > 0.0:
            with torch.no_grad():
                for p_t, p in zip(self.q_target.parameters(), self.q.parameters()):
                    p_t.data.mul_(1.0 - tau)
                    p_t.data.add_(tau * p.data)
        else:
            if self._train_steps % self.config.target_update_steps == 0:
                self.q_target.load_state_dict(self.q.state_dict())

        return {
            "loss": float(loss.item()),
            "td_loss": float(td_loss.item()),
            "margin_loss": float(margin_loss.item()),
            "ce_loss": float(ce_loss.item()),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algo": self.algo,
            "network": "dueling" if isinstance(self.q, DuelingQNetwork) else "plain",
            "config": self.config.__dict__,
            "q_state_dict": self.q.state_dict(),
            "q_target_state_dict": self.q_target.state_dict(),
            "train_steps": self._train_steps,
        }
        torch.save(payload, path)

    def load(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device)
        q_sd = payload["q_state_dict"]
        q_target_sd = payload.get("q_target_state_dict", {})

        cfg = payload.get("config") or {}
        hidden_dim = int(cfg.get("hidden_dim", self.config.hidden_dim))
        hidden_layers = int(cfg.get("hidden_layers", self.config.hidden_layers))

        # Backwards-compatible loading:
        # - Old IDDQN checkpoints used `QNetwork` (keys: "net.*").
        # - New IDDQN uses dueling heads (keys: "feature.*", "value.*", "advantage.*").
        if any(str(k).startswith("feature.") or str(k).startswith("value.") or str(k).startswith("advantage.") for k in q_sd):
            net_cls = DuelingQNetwork
        else:
            net_cls = QNetwork

        if not isinstance(self.q, net_cls):
            self._rebuild_networks(net_cls, hidden_dim=hidden_dim, hidden_layers=hidden_layers)
        else:
            # Architecture can change across experiments (hidden_dim/layers). Rebuild when shapes mismatch.
            try:
                self.q.load_state_dict(q_sd, strict=True)
            except RuntimeError:
                self._rebuild_networks(net_cls, hidden_dim=hidden_dim, hidden_layers=hidden_layers)

        self.q.load_state_dict(q_sd, strict=True)
        if q_target_sd:
            try:
                self.q_target.load_state_dict(q_target_sd, strict=True)
            except RuntimeError:
                # Fall back to syncing the target net if the checkpoint predates saving it (or shapes mismatch).
                self.q_target.load_state_dict(self.q.state_dict(), strict=True)
        else:
            self.q_target.load_state_dict(self.q.state_dict(), strict=True)
        self._train_steps = int(payload.get("train_steps", 0))
