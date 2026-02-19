from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


ContinuousAlgo = Literal["ddpg", "sac"]
CONTINUOUS_ALGOS: tuple[ContinuousAlgo, ...] = ("ddpg", "sac")


@dataclass(frozen=True)
class ContinuousAgentConfig:
    gamma: float = 0.995
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    tau: float = 0.005
    replay_capacity: int = 100_000
    batch_size: int = 256
    hidden_layers: int = 3
    hidden_dim: int = 256

    # DDPG exploration noise scale (relative to action half-range)
    ddpg_noise_std: float = 0.10
    ddpg_noise_clip: float = 0.50

    # SAC settings
    sac_alpha: float = 0.2
    sac_auto_alpha: bool = True
    sac_target_entropy: float | None = None
    sac_log_std_min: float = -5.0
    sac_log_std_max: float = 2.0


def is_continuous_algo(algo: str) -> bool:
    return str(algo).lower().strip() in CONTINUOUS_ALGOS


def canonicalize_continuous_algo(algo: str) -> ContinuousAlgo:
    a = str(algo).lower().strip()
    if a in CONTINUOUS_ALGOS:
        return a  # type: ignore[return-value]
    raise ValueError(f"continuous algo must be one of: {' '.join(CONTINUOUS_ALGOS)}")


def infer_continuous_checkpoint_obs_dim(path: str | Path) -> int:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    obs_dim = payload.get("obs_dim")
    if isinstance(obs_dim, (int, float)) and int(obs_dim) > 0:
        return int(obs_dim)
    raise ValueError(f"Could not infer obs_dim from continuous checkpoint: {path}")


def infer_continuous_checkpoint_algo(path: str | Path) -> str:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    algo = str(payload.get("algo", "")).lower().strip()
    if algo in CONTINUOUS_ALGOS:
        return algo
    raise ValueError(f"Unsupported continuous checkpoint algo: {algo!r} ({path})")


class ContinuousReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, *, seed: int = 0) -> None:
        self.capacity = int(max(1, capacity))
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self._rng = np.random.default_rng(int(seed))

        self._obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.float32)

        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return int(self._size)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = int(self._idx)
        self._obs[i] = np.asarray(obs, dtype=np.float32).reshape(self.obs_dim)
        self._actions[i] = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        self._rewards[i] = float(reward)
        self._next_obs[i] = np.asarray(next_obs, dtype=np.float32).reshape(self.obs_dim)
        self._dones[i] = 1.0 if bool(done) else 0.0

        self._idx = (i + 1) % int(self.capacity)
        self._size = min(int(self.capacity), int(self._size) + 1)

    def sample(self, batch_size: int, *, device: torch.device) -> tuple[torch.Tensor, ...]:
        b = int(max(1, batch_size))
        if int(self._size) < int(b):
            raise RuntimeError("replay size is smaller than batch_size")
        idxs = self._rng.integers(0, int(self._size), size=int(b), endpoint=False)

        obs = torch.from_numpy(self._obs[idxs]).to(device)
        actions = torch.from_numpy(self._actions[idxs]).to(device)
        rewards = torch.from_numpy(self._rewards[idxs]).to(device).unsqueeze(1)
        next_obs = torch.from_numpy(self._next_obs[idxs]).to(device)
        dones = torch.from_numpy(self._dones[idxs]).to(device).unsqueeze(1)
        return obs, actions, rewards, next_obs, dones


def _build_mlp(in_dim: int, out_dim: int, *, hidden_dim: int, hidden_layers: int) -> nn.Sequential:
    if int(hidden_layers) < 1:
        raise ValueError("hidden_layers must be >= 1")
    layers: list[nn.Module] = [nn.Linear(int(in_dim), int(hidden_dim)), nn.ReLU()]
    for _ in range(int(hidden_layers) - 1):
        layers.extend([nn.Linear(int(hidden_dim), int(hidden_dim)), nn.ReLU()])
    layers.append(nn.Linear(int(hidden_dim), int(out_dim)))
    return nn.Sequential(*layers)


def _to_tensor(x: np.ndarray, *, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, *, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__()
        self.net = _build_mlp(int(obs_dim), int(action_dim), hidden_dim=int(hidden_dim), hidden_layers=int(hidden_layers))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class SquashedGaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden_dim: int,
        hidden_layers: int,
        log_std_min: float,
        log_std_max: float,
    ) -> None:
        super().__init__()
        self.backbone = _build_mlp(int(obs_dim), int(hidden_dim), hidden_dim=int(hidden_dim), hidden_layers=int(hidden_layers))
        self.mu = nn.Linear(int(hidden_dim), int(action_dim))
        self.log_std = nn.Linear(int(hidden_dim), int(action_dim))
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def _dist(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mu = self.mu(h)
        log_std = self.log_std(h)
        log_std = torch.clamp(log_std, min=float(self.log_std_min), max=float(self.log_std_max))
        std = torch.exp(log_std)
        return Normal(mu, std), mu, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, mu, _ = self._dist(obs)
        z = dist.rsample()
        action_norm = torch.tanh(z)

        # tanh-squash log-prob correction
        log_prob = dist.log_prob(z) - torch.log(1.0 - action_norm.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        mu_tanh = torch.tanh(mu)
        return action_norm, log_prob, mu_tanh


class CriticQ(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, *, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__()
        self.net = _build_mlp(
            int(obs_dim) + int(action_dim),
            1,
            hidden_dim=int(hidden_dim),
            hidden_layers=int(hidden_layers),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=1)
        return self.net(x)


class _ContinuousAgentBase:
    def __init__(
        self,
        *,
        algo: ContinuousAlgo,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        config: ContinuousAgentConfig,
        seed: int,
        device: str | torch.device,
    ) -> None:
        self.algo: ContinuousAlgo = canonicalize_continuous_algo(str(algo))
        self.obs_dim = int(obs_dim)

        low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        if low.shape != high.shape:
            raise ValueError("action_low and action_high must have the same shape")
        if not np.all(high > low):
            raise ValueError("action_high must be strictly greater than action_low elementwise")

        self.action_dim = int(low.size)
        self.action_low = low
        self.action_high = high
        self.action_mid = 0.5 * (low + high)
        self.action_half = 0.5 * (high - low)

        self.config = config
        self.device = torch.device(device)
        self._rng = np.random.default_rng(int(seed))
        torch.manual_seed(int(seed))

        self.replay = ContinuousReplayBuffer(
            capacity=int(config.replay_capacity),
            obs_dim=int(self.obs_dim),
            action_dim=int(self.action_dim),
            seed=int(seed) + 1337,
        )
        self._train_steps = 0

    def _scale_action(self, action_norm: torch.Tensor) -> torch.Tensor:
        mid = torch.as_tensor(self.action_mid, dtype=torch.float32, device=action_norm.device)
        half = torch.as_tensor(self.action_half, dtype=torch.float32, device=action_norm.device)
        return mid + half * action_norm

    def _clip_action_np(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        return np.clip(a, self.action_low, self.action_high).astype(np.float32, copy=False)

    def observe(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.replay.add(obs, action, float(reward), next_obs, bool(done))

    def act(self, obs: np.ndarray, *, explore: bool) -> np.ndarray:
        raise NotImplementedError

    def update(self) -> dict[str, float] | None:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    def load(self, path: str | Path) -> None:
        raise NotImplementedError


class DDPGAgent(_ContinuousAgentBase):
    def __init__(
        self,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        *,
        config: ContinuousAgentConfig,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            algo="ddpg",
            obs_dim=int(obs_dim),
            action_low=action_low,
            action_high=action_high,
            config=config,
            seed=seed,
            device=device,
        )

        self.actor = DeterministicActor(
            self.obs_dim,
            self.action_dim,
            hidden_dim=int(config.hidden_dim),
            hidden_layers=int(config.hidden_layers),
        ).to(self.device)
        self.actor_target = DeterministicActor(
            self.obs_dim,
            self.action_dim,
            hidden_dim=int(config.hidden_dim),
            hidden_layers=int(config.hidden_layers),
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = CriticQ(
            self.obs_dim,
            self.action_dim,
            hidden_dim=int(config.hidden_dim),
            hidden_layers=int(config.hidden_layers),
        ).to(self.device)
        self.critic_target = CriticQ(
            self.obs_dim,
            self.action_dim,
            hidden_dim=int(config.hidden_dim),
            hidden_layers=int(config.hidden_layers),
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(config.actor_lr))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(config.critic_lr))

    def _soft_update(self) -> None:
        tau = float(self.config.tau)
        for p_t, p in zip(self.actor_target.parameters(), self.actor.parameters(), strict=False):
            p_t.data.mul_(1.0 - tau).add_(tau * p.data)
        for p_t, p in zip(self.critic_target.parameters(), self.critic.parameters(), strict=False):
            p_t.data.mul_(1.0 - tau).add_(tau * p.data)

    def act(self, obs: np.ndarray, *, explore: bool) -> np.ndarray:
        with torch.no_grad():
            o = _to_tensor(np.asarray(obs, dtype=np.float32).reshape(1, self.obs_dim), device=self.device)
            a_norm = self.actor(o).cpu().numpy().reshape(self.action_dim)
        a = self.action_mid + self.action_half * a_norm

        if bool(explore):
            noise_std = float(self.config.ddpg_noise_std) * self.action_half
            noise_clip = float(self.config.ddpg_noise_clip) * self.action_half
            noise = self._rng.normal(loc=0.0, scale=noise_std, size=self.action_dim).astype(np.float32)
            noise = np.clip(noise, -noise_clip, noise_clip)
            a = a + noise
        return self._clip_action_np(a)

    def update(self) -> dict[str, float] | None:
        if len(self.replay) < int(self.config.batch_size):
            return None

        obs, action, reward, next_obs, done = self.replay.sample(int(self.config.batch_size), device=self.device)

        with torch.no_grad():
            next_action = self._scale_action(self.actor_target(next_obs))
            q_next = self.critic_target(next_obs, next_action)
            target = reward + (1.0 - done) * float(self.config.gamma) * q_next

        q = self.critic(obs, action)
        critic_loss = F.mse_loss(q, target)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        actor_action = self._scale_action(self.actor(obs))
        actor_loss = -self.critic(obs, actor_action).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update()
        self._train_steps += 1

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algo": "ddpg",
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
            "config": asdict(self.config),
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "train_steps": int(self._train_steps),
        }
        torch.save(payload, p)

    def load(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device)
        if str(payload.get("algo", "")).lower().strip() != "ddpg":
            raise ValueError(f"Not a DDPG checkpoint: {path}")
        self.actor.load_state_dict(payload["actor_state_dict"], strict=True)
        self.actor_target.load_state_dict(payload.get("actor_target_state_dict", payload["actor_state_dict"]), strict=True)
        self.critic.load_state_dict(payload["critic_state_dict"], strict=True)
        self.critic_target.load_state_dict(payload.get("critic_target_state_dict", payload["critic_state_dict"]), strict=True)
        self._train_steps = int(payload.get("train_steps", 0))


class SACAgent(_ContinuousAgentBase):
    def __init__(
        self,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        *,
        config: ContinuousAgentConfig,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            algo="sac",
            obs_dim=int(obs_dim),
            action_low=action_low,
            action_high=action_high,
            config=config,
            seed=seed,
            device=device,
        )

        self.actor = SquashedGaussianActor(
            self.obs_dim,
            self.action_dim,
            hidden_dim=int(config.hidden_dim),
            hidden_layers=int(config.hidden_layers),
            log_std_min=float(config.sac_log_std_min),
            log_std_max=float(config.sac_log_std_max),
        ).to(self.device)

        self.q1 = CriticQ(
            self.obs_dim,
            self.action_dim,
            hidden_dim=int(config.hidden_dim),
            hidden_layers=int(config.hidden_layers),
        ).to(self.device)
        self.q2 = CriticQ(
            self.obs_dim,
            self.action_dim,
            hidden_dim=int(config.hidden_dim),
            hidden_layers=int(config.hidden_layers),
        ).to(self.device)

        self.q1_target = CriticQ(
            self.obs_dim,
            self.action_dim,
            hidden_dim=int(config.hidden_dim),
            hidden_layers=int(config.hidden_layers),
        ).to(self.device)
        self.q2_target = CriticQ(
            self.obs_dim,
            self.action_dim,
            hidden_dim=int(config.hidden_dim),
            hidden_layers=int(config.hidden_layers),
        ).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(config.actor_lr))
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=float(config.critic_lr))
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=float(config.critic_lr))

        self.auto_alpha = bool(config.sac_auto_alpha)
        default_target_entropy = -float(self.action_dim)
        self.target_entropy = (
            float(config.sac_target_entropy)
            if config.sac_target_entropy is not None
            else float(default_target_entropy)
        )

        if self.auto_alpha:
            init_alpha = max(1e-6, float(config.sac_alpha))
            self.log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32, device=self.device, requires_grad=True)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=float(config.critic_lr))
        else:
            self.log_alpha = torch.tensor(np.log(max(1e-6, float(config.sac_alpha))), dtype=torch.float32, device=self.device)
            self.alpha_opt = None

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _soft_update(self) -> None:
        tau = float(self.config.tau)
        for p_t, p in zip(self.q1_target.parameters(), self.q1.parameters(), strict=False):
            p_t.data.mul_(1.0 - tau).add_(tau * p.data)
        for p_t, p in zip(self.q2_target.parameters(), self.q2.parameters(), strict=False):
            p_t.data.mul_(1.0 - tau).add_(tau * p.data)

    def act(self, obs: np.ndarray, *, explore: bool) -> np.ndarray:
        with torch.no_grad():
            o = _to_tensor(np.asarray(obs, dtype=np.float32).reshape(1, self.obs_dim), device=self.device)
            if bool(explore):
                action_norm, _logp, _mu = self.actor.sample(o)
            else:
                _an, _lp, mu = self.actor.sample(o)
                action_norm = mu
            action = self._scale_action(action_norm).cpu().numpy().reshape(self.action_dim)
        return self._clip_action_np(action)

    def update(self) -> dict[str, float] | None:
        if len(self.replay) < int(self.config.batch_size):
            return None

        obs, action, reward, next_obs, done = self.replay.sample(int(self.config.batch_size), device=self.device)

        with torch.no_grad():
            next_action_norm, next_logp, _next_mu = self.actor.sample(next_obs)
            next_action = self._scale_action(next_action_norm)
            q1_t = self.q1_target(next_obs, next_action)
            q2_t = self.q2_target(next_obs, next_action)
            q_t = torch.minimum(q1_t, q2_t) - self.alpha.detach() * next_logp
            target = reward + (1.0 - done) * float(self.config.gamma) * q_t

        q1 = self.q1(obs, action)
        q2 = self.q2(obs, action)
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)

        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.q2_opt.step()

        action_norm, logp, _mu = self.actor.sample(obs)
        action_now = self._scale_action(action_norm)
        q1_pi = self.q1(obs, action_now)
        q2_pi = self.q2(obs, action_now)
        q_pi = torch.minimum(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * logp - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss_value = 0.0
        if bool(self.auto_alpha) and self.alpha_opt is not None:
            alpha_loss = -(self.log_alpha * (logp.detach() + float(self.target_entropy))).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_value = float(alpha_loss.item())

        self._soft_update()
        self._train_steps += 1

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.detach().item()),
            "alpha_loss": float(alpha_loss_value),
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algo": "sac",
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
            "config": asdict(self.config),
            "actor_state_dict": self.actor.state_dict(),
            "q1_state_dict": self.q1.state_dict(),
            "q2_state_dict": self.q2.state_dict(),
            "q1_target_state_dict": self.q1_target.state_dict(),
            "q2_target_state_dict": self.q2_target.state_dict(),
            "log_alpha": float(self.log_alpha.detach().cpu().item()),
            "auto_alpha": bool(self.auto_alpha),
            "train_steps": int(self._train_steps),
        }
        torch.save(payload, p)

    def load(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device)
        if str(payload.get("algo", "")).lower().strip() != "sac":
            raise ValueError(f"Not a SAC checkpoint: {path}")

        self.actor.load_state_dict(payload["actor_state_dict"], strict=True)
        self.q1.load_state_dict(payload["q1_state_dict"], strict=True)
        self.q2.load_state_dict(payload["q2_state_dict"], strict=True)
        self.q1_target.load_state_dict(payload.get("q1_target_state_dict", payload["q1_state_dict"]), strict=True)
        self.q2_target.load_state_dict(payload.get("q2_target_state_dict", payload["q2_state_dict"]), strict=True)

        loaded_auto_alpha = bool(payload.get("auto_alpha", self.auto_alpha))
        loaded_log_alpha = float(payload.get("log_alpha", float(self.log_alpha.detach().cpu().item())))
        if bool(loaded_auto_alpha) and self.alpha_opt is not None:
            self.log_alpha.data.copy_(torch.tensor(loaded_log_alpha, dtype=torch.float32, device=self.device))
        else:
            self.log_alpha = torch.tensor(loaded_log_alpha, dtype=torch.float32, device=self.device)

        self._train_steps = int(payload.get("train_steps", 0))
