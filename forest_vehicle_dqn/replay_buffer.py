from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FLAG_DEMO = np.uint8(1 << 0)
FLAG_NEAR_GOAL = np.uint8(1 << 1)
FLAG_STUCK = np.uint8(1 << 2)
FLAG_HAZARD = np.uint8(1 << 3)


@dataclass(frozen=True)
class Batch:
    idxs: np.ndarray
    weights: np.ndarray
    obs: np.ndarray
    actions: np.ndarray
    rewards_1: np.ndarray
    next_obs_1: np.ndarray
    next_action_masks_1: np.ndarray
    dones_1: np.ndarray
    rewards_n: np.ndarray
    next_obs_n: np.ndarray
    next_action_masks_n: np.ndarray
    dones_n: np.ndarray
    demos: np.ndarray
    n_steps_n: np.ndarray
    flags: np.ndarray


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        n_actions: int,
        *,
        rng: np.random.Generator,
        stratified: bool = False,
        prioritized: bool = False,
        per_alpha: float = 0.4,
        per_eps_agent: float = 1e-3,
        per_eps_demo: float = 1.0,
        frac_demo: float = 0.0,
        frac_goal: float = 0.0,
        frac_stuck: float = 0.0,
        frac_hazard: float = 0.0,
        per_boost_near_goal: float = 0.0,
        per_boost_stuck: float = 0.0,
        per_boost_hazard: float = 0.0,
    ):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self._rng = rng
        self._stratified = bool(stratified)
        self._prioritized = bool(prioritized)
        self._per_alpha = float(per_alpha)
        self._per_eps_agent = float(per_eps_agent)
        self._per_eps_demo = float(per_eps_demo)
        self._frac_demo = float(frac_demo)
        self._frac_goal = float(frac_goal)
        self._frac_stuck = float(frac_stuck)
        self._frac_hazard = float(frac_hazard)
        self._per_boost_near_goal = float(per_boost_near_goal)
        self._per_boost_stuck = float(per_boost_stuck)
        self._per_boost_hazard = float(per_boost_hazard)

        if self._prioritized and self._stratified:
            raise ValueError("ReplayBuffer cannot enable both prioritized and stratified sampling.")
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        if self.obs_dim <= 0:
            raise ValueError("obs_dim must be > 0")
        if self.n_actions <= 0:
            raise ValueError("n_actions must be > 0")
        if not np.isfinite(self._per_alpha) or self._per_alpha < 0.0:
            raise ValueError("per_alpha must be finite and >= 0")
        if not np.isfinite(self._per_eps_agent) or self._per_eps_agent <= 0.0:
            raise ValueError("per_eps_agent must be finite and > 0")
        if not np.isfinite(self._per_eps_demo) or self._per_eps_demo <= 0.0:
            raise ValueError("per_eps_demo must be finite and > 0")
        if not np.isfinite(self._per_boost_near_goal) or self._per_boost_near_goal < 0.0:
            raise ValueError("per_boost_near_goal must be finite and >= 0")
        if not np.isfinite(self._per_boost_stuck) or self._per_boost_stuck < 0.0:
            raise ValueError("per_boost_stuck must be finite and >= 0")
        if not np.isfinite(self._per_boost_hazard) or self._per_boost_hazard < 0.0:
            raise ValueError("per_boost_hazard must be finite and >= 0")

        self._obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity,), dtype=np.int64)
        self._rewards_1 = np.zeros((self.capacity,), dtype=np.float32)
        self._next_obs_1 = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._next_action_masks_1 = np.ones((self.capacity, self.n_actions), dtype=np.bool_)
        self._dones_1 = np.zeros((self.capacity,), dtype=np.float32)
        self._rewards_n = np.zeros((self.capacity,), dtype=np.float32)
        self._next_obs_n = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._next_action_masks_n = np.ones((self.capacity, self.n_actions), dtype=np.bool_)
        self._dones_n = np.zeros((self.capacity,), dtype=np.float32)
        self._demos = np.zeros((self.capacity,), dtype=np.float32)
        self._n_steps_n = np.ones((self.capacity,), dtype=np.int64)
        self._flags = np.zeros((self.capacity,), dtype=np.uint8)

        self._idx = 0
        self._size = 0

        # Prioritized experience replay (sum-tree for sampling; min-tree for IS-weight normalization).
        self._tree_cap = 1
        while self._tree_cap < int(self.capacity):
            self._tree_cap *= 2
        self._sum_tree = np.zeros((2 * int(self._tree_cap),), dtype=np.float64)
        self._min_tree = np.full((2 * int(self._tree_cap),), float("inf"), dtype=np.float64)
        self._max_priority = 1.0

    def __len__(self) -> int:
        return self._size

    def _tree_set(self, idx: int, p_alpha: float) -> None:
        i = int(idx) + int(self._tree_cap)
        v = float(p_alpha)
        self._sum_tree[i] = v
        self._min_tree[i] = v if v > 0.0 else float("inf")
        i //= 2
        while i >= 1:
            l = 2 * int(i)
            r = l + 1
            self._sum_tree[i] = float(self._sum_tree[l] + self._sum_tree[r])
            self._min_tree[i] = float(min(self._min_tree[l], self._min_tree[r]))
            i //= 2

    def _tree_find_prefixsum(self, mass: float) -> int:
        # Return leaf index for the given prefix-sum mass (assumes 0 <= mass < total_sum).
        idx = 1
        m = float(mass)
        while idx < int(self._tree_cap):
            left = 2 * int(idx)
            if m <= float(self._sum_tree[left]):
                idx = int(left)
            else:
                m -= float(self._sum_tree[left])
                idx = int(left) + 1
        leaf = int(idx) - int(self._tree_cap)
        return int(min(int(leaf), int(self.capacity - 1)))

    def update_priorities(
        self,
        idxs: np.ndarray,
        td_errors: np.ndarray,
        *,
        demos: np.ndarray | None = None,
        flags: np.ndarray | None = None,
    ) -> None:
        """Update PER priorities given TD errors.

        DQfD-style: use different priority constants for demo vs agent transitions:
          p_i = |delta_i| + eps_demo  (demo)
          p_i = |delta_i| + eps_agent (agent)
        The sampling distribution uses p_i ** alpha.
        """

        if not bool(self._prioritized):
            return

        idxs_arr = np.asarray(idxs, dtype=np.int64).reshape(-1)
        td = np.asarray(td_errors, dtype=np.float64).reshape(-1)
        if int(idxs_arr.size) != int(td.size):
            raise ValueError("idxs and td_errors must have the same length")

        if demos is None:
            demo_mask = self._demos[idxs_arr] > 0.5
        else:
            demo_mask = np.asarray(demos, dtype=np.float32).reshape(-1) > 0.5
            if int(demo_mask.size) != int(idxs_arr.size):
                raise ValueError("demos must have the same length as idxs when provided")

        if flags is None:
            flags_arr = self._flags[idxs_arr].astype(np.uint8, copy=False)
        else:
            flags_arr = np.asarray(flags, dtype=np.uint8).reshape(-1)
            if int(flags_arr.size) != int(idxs_arr.size):
                raise ValueError("flags must have the same length as idxs when provided")

        eps = np.where(demo_mask, float(self._per_eps_demo), float(self._per_eps_agent)).astype(np.float64, copy=False)
        p = np.abs(td) + eps

        boosts = np.ones_like(p, dtype=np.float64)
        if float(self._per_boost_near_goal) > 0.0:
            near_goal_mask = (flags_arr & FLAG_NEAR_GOAL) != 0
            boosts = boosts * np.where(near_goal_mask, 1.0 + float(self._per_boost_near_goal), 1.0)
        if float(self._per_boost_stuck) > 0.0:
            stuck_mask = (flags_arr & FLAG_STUCK) != 0
            boosts = boosts * np.where(stuck_mask, 1.0 + float(self._per_boost_stuck), 1.0)
        if float(self._per_boost_hazard) > 0.0:
            hazard_mask = (flags_arr & FLAG_HAZARD) != 0
            boosts = boosts * np.where(hazard_mask, 1.0 + float(self._per_boost_hazard), 1.0)
        p = p * boosts

        # Guard against NaNs/infs.
        p = np.where(np.isfinite(p), p, float(self._max_priority))
        p = np.maximum(p, 1e-12)

        p_max = float(np.max(p)) if int(p.size) > 0 else 1.0
        self._max_priority = float(max(float(self._max_priority), float(p_max)))

        alpha = float(self._per_alpha)
        # alpha==0 => uniform priorities (all 1); still keep the tree consistent.
        p_alpha = np.power(p, alpha) if alpha != 0.0 else np.ones_like(p)
        for i, v in zip(idxs_arr.tolist(), p_alpha.tolist(), strict=False):
            self._tree_set(int(i), float(v))

    def _coerce_next_mask(self, next_action_mask: np.ndarray | None) -> np.ndarray:
        if next_action_mask is None:
            return np.ones((int(self.n_actions),), dtype=np.bool_)
        m = np.asarray(next_action_mask, dtype=bool).reshape(-1)
        if int(m.size) != int(self.n_actions):
            raise ValueError("next_action_mask must have shape (n_actions,)")
        # If the mask is malformed (all False), fall back to "all admissible" to avoid propagating -inf.
        return np.ones((int(self.n_actions),), dtype=np.bool_) if not bool(m.any()) else m

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward_1: float,
        next_obs_1: np.ndarray,
        done_1: bool,
        *,
        next_action_mask_1: np.ndarray | None = None,
        reward_n: float | None = None,
        next_obs_n: np.ndarray | None = None,
        done_n: bool | None = None,
        next_action_mask_n: np.ndarray | None = None,
        demo: bool = False,
        n_steps_n: int = 1,
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
        self._rewards_1[i] = float(reward_1)
        self._next_obs_1[i] = next_obs_1
        self._next_action_masks_1[i] = self._coerce_next_mask(next_action_mask_1)
        self._dones_1[i] = 1.0 if bool(done_1) else 0.0

        if reward_n is None or next_obs_n is None or done_n is None:
            reward_n = float(reward_1)
            next_obs_n = next_obs_1
            done_n = bool(done_1)
            next_action_mask_n = self._next_action_masks_1[i]
            n_steps_n = 1

        self._rewards_n[i] = float(reward_n)
        self._next_obs_n[i] = next_obs_n
        self._next_action_masks_n[i] = self._coerce_next_mask(next_action_mask_n)
        self._dones_n[i] = 1.0 if bool(done_n) else 0.0
        self._demos[i] = 1.0 if demo else 0.0
        self._n_steps_n[i] = int(max(1, int(n_steps_n)))
        f = np.uint8(int(flags) & 0xFF)
        if bool(demo):
            f = np.uint8(f | FLAG_DEMO)
        self._flags[i] = f

        if bool(self._prioritized):
            eps = float(self._per_eps_demo if bool(demo) else self._per_eps_agent)
            p0 = float(max(float(self._max_priority), float(eps)))
            alpha = float(self._per_alpha)
            p_alpha = float(p0**alpha) if alpha != 0.0 else 1.0
            self._tree_set(int(i), float(p_alpha))

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

    def sample(self, batch_size: int, *, beta: float | None = None) -> Batch:
        if self._size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        n = min(int(batch_size), self._size)
        weights = np.ones((n,), dtype=np.float32)

        if bool(self._prioritized):
            total = float(self._sum_tree[1])
            if not (total > 0.0 and np.isfinite(total)):
                idxs = self._rng.integers(0, self._size, size=n, dtype=np.int64)
            else:
                segment = float(total) / float(max(1, int(n)))
                masses = (self._rng.random(int(n)) + np.arange(int(n), dtype=np.float64)) * float(segment)
                # Avoid edge cases where mass==total due to floating point.
                masses = np.minimum(masses, float(total) - 1e-12)
                idxs = np.array([self._tree_find_prefixsum(float(m)) for m in masses], dtype=np.int64)

                b = 1.0 if beta is None else float(beta)
                b = float(np.clip(b, 0.0, 1.0))
                if b > 0.0:
                    leaf = self._sum_tree[int(self._tree_cap) + idxs]
                    p = np.asarray(leaf, dtype=np.float64) / float(total)
                    p = np.maximum(p, 1e-12)
                    n_buf = float(self._size)
                    w = np.power(n_buf * p, -float(b))

                    min_leaf = float(self._min_tree[1])
                    if not np.isfinite(min_leaf) or min_leaf <= 0.0:
                        w_max = float(np.max(w)) if int(w.size) > 0 else 1.0
                    else:
                        p_min = max(float(min_leaf) / float(total), 1e-12)
                        w_max = float(np.power(n_buf * p_min, -float(b)))
                    if not (w_max > 0.0 and np.isfinite(w_max)):
                        w_max = 1.0
                    w = w / float(w_max)
                    weights = w.astype(np.float32, copy=False)

        elif not bool(self._stratified):
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
            idxs=idxs,
            weights=weights,
            obs=self._obs[idxs],
            actions=self._actions[idxs],
            rewards_1=self._rewards_1[idxs],
            next_obs_1=self._next_obs_1[idxs],
            next_action_masks_1=self._next_action_masks_1[idxs],
            dones_1=self._dones_1[idxs],
            rewards_n=self._rewards_n[idxs],
            next_obs_n=self._next_obs_n[idxs],
            next_action_masks_n=self._next_action_masks_n[idxs],
            dones_n=self._dones_n[idxs],
            demos=self._demos[idxs],
            n_steps_n=self._n_steps_n[idxs],
            flags=self._flags[idxs],
        )
