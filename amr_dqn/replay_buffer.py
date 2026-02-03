from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FLAG_DEMO = np.uint8(1 << 0)
FLAG_NEAR_GOAL = np.uint8(1 << 1)
FLAG_STUCK = np.uint8(1 << 2)
FLAG_HAZARD = np.uint8(1 << 3)


@dataclass(frozen=True)
class Batch:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    next_action_masks: np.ndarray
    dones: np.ndarray
    demos: np.ndarray
    n_steps: np.ndarray


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        n_actions: int,
        *,
        rng: np.random.Generator,
        stratified: bool = False,
        frac_demo: float = 0.0,
        frac_goal: float = 0.0,
        frac_stuck: float = 0.0,
        frac_hazard: float = 0.0,
    ):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self._rng = rng
        self._stratified = bool(stratified)
        self._frac_demo = float(frac_demo)
        self._frac_goal = float(frac_goal)
        self._frac_stuck = float(frac_stuck)
        self._frac_hazard = float(frac_hazard)

        self._obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity,), dtype=np.int64)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._next_action_masks = np.ones((self.capacity, self.n_actions), dtype=np.bool_)
        self._dones = np.zeros((self.capacity,), dtype=np.float32)
        self._demos = np.zeros((self.capacity,), dtype=np.float32)
        self._n_steps = np.ones((self.capacity,), dtype=np.int64)
        self._flags = np.zeros((self.capacity,), dtype=np.uint8)

        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        *,
        next_action_mask: np.ndarray | None = None,
        demo: bool = False,
        n_steps: int = 1,
        flags: int = 0,
    ) -> None:
        i = self._idx
        # DQfD-style safeguard: preserve demonstration transitions so they
        # remain available for supervised losses throughout long trainings.
        #
        # If the buffer is full and we'd overwrite a demo transition with a
        # non-demo transition, search for the next non-demo slot to overwrite.
        if self._size >= self.capacity and (not bool(demo)) and float(self._demos[i]) > 0.5:
            j = int(i)
            for _ in range(int(self.capacity)):
                if float(self._demos[j]) <= 0.5:
                    i = int(j)
                    break
                j = (int(j) + 1) % int(self.capacity)
        self._obs[i] = obs
        self._actions[i] = int(action)
        self._rewards[i] = float(reward)
        self._next_obs[i] = next_obs
        if next_action_mask is None:
            self._next_action_masks[i] = True
        else:
            m = np.asarray(next_action_mask, dtype=bool).reshape(-1)
            if m.size != int(self.n_actions):
                raise ValueError("next_action_mask must have shape (n_actions,)")
            self._next_action_masks[i] = True if not bool(m.any()) else m
        self._dones[i] = 1.0 if done else 0.0
        self._demos[i] = 1.0 if demo else 0.0
        self._n_steps[i] = int(max(1, int(n_steps)))
        f = np.uint8(int(flags) & 0xFF)
        if bool(demo):
            f = np.uint8(f | FLAG_DEMO)
        self._flags[i] = f

        self._idx = (int(i) + 1) % int(self.capacity)
        self._size = min(self.capacity, self._size + 1)

    def _sample_flag(self, flag: np.uint8, k: int) -> np.ndarray:
        k = int(max(0, int(k)))
        if k == 0:
            return np.zeros((0,), dtype=np.int64)
        if self._size == 0:
            raise ValueError("Cannot sample from an empty buffer")

        picked: list[np.ndarray] = []
        picked_n = 0
        rounds = 0
        # Vectorized rejection sampling: efficient when the flagged subset is sparse.
        m = max(64, int(k) * 4)
        while picked_n < int(k) and rounds < 8:
            cand = self._rng.integers(0, self._size, size=m, dtype=np.int64)
            ok = (self._flags[cand] & flag) != 0
            good = cand[ok]
            if good.size > 0:
                picked.append(good)
                picked_n += int(good.size)
            rounds += 1
            m *= 2

        if picked_n <= 0:
            return self._rng.integers(0, self._size, size=k, dtype=np.int64)

        out = np.concatenate(picked, axis=0)[:k]
        if int(out.size) < int(k):
            extra = self._rng.integers(0, self._size, size=int(k) - int(out.size), dtype=np.int64)
            out = np.concatenate([out, extra], axis=0)
        return out.astype(np.int64, copy=False)

    def sample(self, batch_size: int) -> Batch:
        if self._size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        n = min(int(batch_size), self._size)
        if not bool(self._stratified):
            idxs = self._rng.integers(0, self._size, size=n, dtype=np.int64)
        else:
            frac_demo = float(np.clip(float(self._frac_demo), 0.0, 1.0))
            frac_goal = float(np.clip(float(self._frac_goal), 0.0, 1.0))
            frac_stuck = float(np.clip(float(self._frac_stuck), 0.0, 1.0))
            frac_hazard = float(np.clip(float(self._frac_hazard), 0.0, 1.0))
            frac_sum = float(frac_demo + frac_goal + frac_stuck + frac_hazard)
            if frac_sum <= 0.0:
                idxs = self._rng.integers(0, self._size, size=n, dtype=np.int64)
            else:
                # Scale down if the requested fractions exceed 1.0.
                scale = 1.0 if frac_sum <= 1.0 else (1.0 / float(frac_sum))
                k_demo = int(n * float(frac_demo) * float(scale))
                k_goal = int(n * float(frac_goal) * float(scale))
                k_stuck = int(n * float(frac_stuck) * float(scale))
                k_hazard = int(n * float(frac_hazard) * float(scale))
                k_total = int(k_demo + k_goal + k_stuck + k_hazard)
                k_rest = max(0, int(n) - int(k_total))

                idx_parts = [
                    self._sample_flag(FLAG_DEMO, k_demo),
                    self._sample_flag(FLAG_NEAR_GOAL, k_goal),
                    self._sample_flag(FLAG_STUCK, k_stuck),
                    self._sample_flag(FLAG_HAZARD, k_hazard),
                    self._rng.integers(0, self._size, size=k_rest, dtype=np.int64),
                ]
                idxs = np.concatenate(idx_parts, axis=0)
                if int(idxs.size) > int(n):
                    idxs = idxs[:n]
                elif int(idxs.size) < int(n):
                    extra = self._rng.integers(0, self._size, size=int(n) - int(idxs.size), dtype=np.int64)
                    idxs = np.concatenate([idxs, extra], axis=0)
                # Shuffle to avoid any ordering bias in the returned batch.
                self._rng.shuffle(idxs)

        return Batch(
            obs=self._obs[idxs],
            actions=self._actions[idxs],
            rewards=self._rewards[idxs],
            next_obs=self._next_obs[idxs],
            next_action_masks=self._next_action_masks[idxs],
            dones=self._dones[idxs],
            demos=self._demos[idxs],
            n_steps=self._n_steps[idxs],
        )
