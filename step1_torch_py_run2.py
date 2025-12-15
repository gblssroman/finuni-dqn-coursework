import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import gc
from datetime import datetime
import altair as alt
import joblib
from collections import defaultdict
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces

from tqdm import trange, tqdm

import random

alt.data_transformers.enable("vegafusion")
pd.set_option("display.max_columns", None)

train_df = pl.read_parquet("df_l2_train_FINAL_WIDE.parquet")
test_df = pl.read_parquet("df_l2_test_FINAL_WIDE.parquet")
ohlcv_dict = joblib.load("ohlcv_dfs_FINAL.pkl")

with open("df_l2_cols_order.json", "r") as f:
    cols_order = json.load(f)

eth_ohlcv_tensor = torch.tensor(ohlcv_dict["ETH"].values[:, 2:].astype(np.float32), dtype=torch.float32)
sol_ohlcv_tensor = torch.tensor(ohlcv_dict["SOL"].values[:, 2:].astype(np.float32), dtype=torch.float32)
pepe_ohlcv_tensor = torch.tensor(ohlcv_dict["PEPE"].values[:, 2:].astype(np.float32), dtype=torch.float32)
sui_ohlcv_tensor = torch.tensor(ohlcv_dict["SUI"].values[:, 2:].astype(np.float32), dtype=torch.float32)
xrp_ohlcv_tensor = torch.tensor(ohlcv_dict["XRP"].values[:, 2:].astype(np.float32), dtype=torch.float32)

train_df = train_df[cols_order]
test_df = test_df[cols_order]

train_eth = torch.tensor(train_df.filter(pl.col("trading_pair") == "ETH-USDT").sort("ts")[:, 2:].to_numpy().astype(np.float32), dtype=torch.float32)
train_sol = torch.tensor(train_df.filter(pl.col("trading_pair") == "SOL-USDT").sort("ts")[:, 2:].to_numpy().astype(np.float32), dtype=torch.float32)
train_pepe = torch.tensor(train_df.filter(pl.col("trading_pair") == "PEPE-USDT").sort("ts")[:, 2:].to_numpy().astype(np.float32), dtype=torch.float32)
train_sui = torch.tensor(train_df.filter(pl.col("trading_pair") == "SUI-USDT").sort("ts")[:, 2:].to_numpy().astype(np.float32), dtype=torch.float32)
train_xrp = torch.tensor(train_df.filter(pl.col("trading_pair") == "XRP-USDT").sort("ts")[:, 2:].to_numpy().astype(np.float32), dtype=torch.float32)

test_eth = torch.tensor(test_df.filter(pl.col("trading_pair") == "ETH-USDT").sort("ts")[:, 2:].to_numpy().astype(np.float32), dtype=torch.float32)
test_sol = torch.tensor(test_df.filter(pl.col("trading_pair") == "SOL-USDT").sort("ts")[:, 2:].to_numpy().astype(np.float32), dtype=torch.float32)
test_pepe = torch.tensor(test_df.filter(pl.col("trading_pair") == "PEPE-USDT").sort("ts")[:, 2:].to_numpy().astype(np.float32), dtype=torch.float32)
test_sui = torch.tensor(test_df.filter(pl.col("trading_pair") == "SUI-USDT").sort("ts")[:, 2:].to_numpy().astype(np.float32), dtype=torch.float32)
test_xrp = torch.tensor(test_df.filter(pl.col("trading_pair") == "XRP-USDT").sort("ts")[:, 2:].to_numpy().astype(np.float32), dtype=torch.float32)

import gymnasium as gym
from gymnasium import spaces
import torch


class MultiVenueSpreadEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        l2_tensor: torch.Tensor,
        ohlcv_tensor: torch.Tensor,
        fee_rate: float = 0.0005,
        max_steps: int | None = None,
        invalid_action_penalty: float = 0.0,
        device: str = "cpu",
        notional: float = 100.0,
        random_start: bool = True,
        lambda_time: float = 1e-6,
    ):
        super().__init__()

        self.l2 = l2_tensor.to(device)
        self.ohlcv = ohlcv_tensor.to(device)
        self.N = l2_tensor.shape[0]

        self.fee_rate = fee_rate
        self.max_steps = max_steps or (self.N - 2)
        self.invalid_action_penalty = invalid_action_penalty
        self.device = device
        self.notional = float(notional)
        self.random_start = random_start
        self.lambda_time = lambda_time

        self.venues = [
            "bitget",
            "gate_io",
            "kucoin",
            "gate_io_perpetual",
            "bitget_perpetual",
        ]

        self.venue_offsets = {
            "bitget": 0,
            "gate_io": 12,
            "kucoin": 24,
            "gate_io_perpetual": 36,
            "bitget_perpetual": 48,
        }

        self.products = [
            ("bitget", "bitget_perpetual"),
            ("bitget", "gate_io_perpetual"),
            ("gate_io", "bitget_perpetual"),
            ("gate_io", "gate_io_perpetual"),
            ("kucoin", "bitget_perpetual"),
            ("kucoin", "gate_io_perpetual"),
            ("bitget_perpetual", "gate_io_perpetual"),
        ]
        self.n_products = len(self.products)

        self.action_space = spaces.Discrete(self.n_products + 2)

        self.l2_features_per_venue = 10
        self.l2_state_dim = self.l2_features_per_venue * len(self.venues)
        self.ohlcv_dim = ohlcv_tensor.shape[1]
        self.portfolio_dim = 3
        self.obs_dim = self.l2_state_dim + self.ohlcv_dim + self.portfolio_dim

        self.observation_space = spaces.Box(
            low=-1e12, high=1e12, shape=(self.obs_dim,), dtype=float
        )

        self._t = 0
        self._steps = 0
        self.curr_position = 0
        self.curr_product_idx = None
        self.entry_delta = torch.tensor(0.0, device=device)
        self.time_in_pos = 0
        self.mid0_entry = torch.tensor(0.0, device=device)
        self.mid1_entry = torch.tensor(0.0, device=device)

        self.size_long = torch.tensor(0.0, device=device)
        self.size_short = torch.tensor(0.0, device=device)

    def _get_ask(self, row: torch.Tensor, venue: str):
        off = self.venue_offsets[venue]
        return row[off + 0]

    def _get_bid(self, row: torch.Tensor, venue: str):
        off = self.venue_offsets[venue]
        return row[off + 1]

    def _get_mid(self, row: torch.Tensor, venue: str):
        off = self.venue_offsets[venue]
        ask = row[off + 0]
        bid = row[off + 1]
        return 0.5 * (ask + bid)

    def _total_pnl(self, row: torch.Tensor, product_idx: int):
        v0, v1 = self.products[product_idx]

        mid0 = self._get_mid(row, v0)
        mid1 = self._get_mid(row, v1)

        pnl_long = self.size_long * (mid0 - self.mid0_entry)
        pnl_short = self.size_short * (self.mid1_entry - mid1)

        return pnl_long + pnl_short

    def _build_obs(self):
        row = self.l2[self._t]
        l2_feats = []
        for v in self.venues:
            off = self.venue_offsets[v]
            l2_feats.append(row[off + 2 : off + 12])
        l2_feats = torch.cat(l2_feats).float()

        idx_feat = int(row[-1].item())
        ohlcv_feats = self.ohlcv[idx_feat].float()

        portfolio = torch.tensor(
            [
                float(self.curr_position),
                float(self.entry_delta.item()) if self.curr_position == 1 else 0.0,
                float(torch.log1p(torch.tensor(self.time_in_pos)).item())
                if self.curr_position == 1
                else 0.0,
            ],
            device=self.device,
            dtype=torch.float32,
        )

        return torch.cat([l2_feats, ohlcv_feats, portfolio]).float()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.random_start:
            max_start = self.N - self.max_steps - 2
            max_start = max(max_start, 1)
            start_t = int(torch.randint(0, max_start, (1,)).item())
            self._t = start_t
        else:
            self._t = 0

        self._steps = 0
        self.curr_position = 0
        self.curr_product_idx = None
        self.entry_delta = torch.tensor(0.0, device=self.device)
        self.time_in_pos = 0

        self.mid0_entry = torch.tensor(0.0, device=self.device)
        self.mid1_entry = torch.tensor(0.0, device=self.device)
        self.size_long = torch.tensor(0.0, device=self.device)
        self.size_short = torch.tensor(0.0, device=self.device)

        obs = self._build_obs()
        return obs, {}

    def step(self, action: int):
        done = False
        reward = torch.tensor(0.0, device=self.device)

        row_t = self.l2[self._t]
        row_tp1 = self.l2[self._t + 1]

        if 0 <= action < self.n_products:
            if self.curr_position == 0:
                v0, v1 = self.products[action]

                mid0 = self._get_mid(row_t, v0)
                mid1 = self._get_mid(row_t, v1)

                # prevent zeros/nans
                if (mid0 <= 0) or (mid1 <= 0) \
                   or (not torch.isfinite(mid0)) \
                   or (not torch.isfinite(mid1)):
                    reward -= self.invalid_action_penalty
                else:
                    self.curr_position = 1
                    self.curr_product_idx = action
                    self.time_in_pos = 0

                    self.mid0_entry = mid0
                    self.mid1_entry = mid1

                    self.size_long = torch.tensor(self.notional, device=self.device) / mid0
                    self.size_short = torch.tensor(self.notional, device=self.device) / mid1

                    self.entry_delta = torch.log(mid0 / mid1)

                    fee_entry = self.fee_rate * (
                        self.size_long * mid0 + self.size_short * mid1
                    )
                    reward -= fee_entry
            else:
                reward -= self.invalid_action_penalty

        elif action == self.n_products:
            pass

        elif action == self.n_products + 1:
            if self.curr_position == 1 and self.curr_product_idx is not None:
                pnl_t = self._total_pnl(row_t, self.curr_product_idx)
                pnl_tp1 = self._total_pnl(row_tp1, self.curr_product_idx)
                reward += pnl_tp1 - pnl_t

                v0, v1 = self.products[self.curr_product_idx]
                mid0_close = self._get_mid(row_tp1, v0)
                mid1_close = self._get_mid(row_tp1, v1)

                fee_exit = self.fee_rate * (
                    self.size_long * mid0_close + self.size_short * mid1_close
                )
                reward -= fee_exit

                self.curr_position = 0
                self.curr_product_idx = None
                self.entry_delta = torch.tensor(0.0, device=self.device)
                self.time_in_pos = 0
                self.mid0_entry = torch.tensor(0.0, device=self.device)
                self.mid1_entry = torch.tensor(0.0, device=self.device)
                self.size_long = torch.tensor(0.0, device=self.device)
                self.size_short = torch.tensor(0.0, device=self.device)
            else:
                reward -= self.invalid_action_penalty

        if (
            self.curr_position == 1
            and self.curr_product_idx is not None
            and action != self.n_products + 1
        ):
            pnl_t = self._total_pnl(row_t, self.curr_product_idx)
            pnl_tp1 = self._total_pnl(row_tp1, self.curr_product_idx)
            reward += pnl_tp1 - pnl_t

            time_penalty = self.lambda_time * torch.log1p(
                torch.tensor(self.time_in_pos, device=self.device, dtype=torch.float32)
            )
            reward -= time_penalty

            self.time_in_pos += 1

        self._t += 1
        self._steps += 1

        if self._t >= self.N - 2:
            done = True
        if self._steps >= self.max_steps:
            done = True

        obs = self._build_obs()
        return obs, float(reward.item()), done, False, {}

class DQN(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device

        self.states = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity,), dtype=torch.long, device=device)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity,), dtype=torch.float32, device=device)

        self.size = 0
        self.pos = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self.size


def train_dqn_multi_env(
    envs,
    num_steps: int = 500_000,
    warmup_steps: int = 20_000,
    batch_size: int = 512,
    gamma: float = 0.99,
    lr: float = 1e-4,
    buffer_capacity: int = 500_000,
    target_update_freq: int = 1_000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 400_000,
    device: str = "cuda",
    save_path_q: str | None = None,
    save_path_target: str | None = None,
):
    current_env = random.choice(envs)
    obs, info = current_env.reset()

    if isinstance(obs, torch.Tensor):
        obs = obs.to(device)
    else:
        obs = torch.tensor(obs, dtype=torch.float32, device=device)

    obs_dim = obs.shape[0]
    n_actions = current_env.action_space.n

    q_net = DQN(obs_dim, n_actions).to(device)
    target_net = DQN(obs_dim, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity, obs_dim, device=device)

    global_step = 0
    episode_reward = 0.0
    episode = 0

    pbar = trange(num_steps, desc="training", smoothing=0.1)

    for _ in pbar:
        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * (global_step / epsilon_decay_steps),
        )

        if torch.rand(1).item() < epsilon:
            action = current_env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_net(obs.unsqueeze(0))
                action = int(torch.argmax(q_values, dim=1).item())

        next_obs, reward, done, truncated, info = current_env.step(action)

        if isinstance(next_obs, torch.Tensor):
            next_obs_t = next_obs.to(device)
        else:
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

        r_t = torch.tensor(reward, dtype=torch.float32, device=device)
        terminal = done or truncated

        buffer.add(obs, action, r_t, next_obs_t, terminal)

        obs = next_obs_t
        episode_reward += reward
        global_step += 1

        if len(buffer) >= max(batch_size, warmup_steps):
            states_b, actions_b, rewards_b, next_states_b, dones_b = buffer.sample(batch_size)

            with torch.no_grad():
                next_q = target_net(next_states_b)
                next_q_max, _ = next_q.max(dim=1)
                target = rewards_b + gamma * (1.0 - dones_b) * next_q_max

            q_values = q_net(states_b)
            q_selected = q_values.gather(1, actions_b.unsqueeze(1)).squeeze(1)

            loss = nn.SmoothL1Loss()(q_selected, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
            optimizer.step()

            if global_step % target_update_freq == 0:
                target_net.load_state_dict(q_net.state_dict())

        if terminal:
            episode += 1
            pbar.set_postfix(
                eps=f"{epsilon:.3f}",
                ep_reward=f"{episode_reward:.2f}",
                env_id=f"{envs.index(current_env)}",
            )
            episode_reward = 0.0
            current_env = random.choice(envs)
            obs, info = current_env.reset()
            if isinstance(obs, torch.Tensor):
                obs = obs.to(device)
            else:
                obs = torch.tensor(obs, dtype=torch.float32, device=device)

        if global_step >= num_steps:
            break

    if save_path_q is not None:
        torch.save(q_net.state_dict(), save_path_q)
    if save_path_target is not None:
        torch.save(target_net.state_dict(), save_path_target)

    return q_net, target_net

device = "cuda"

envs = []

FEE = 0.0001
MAX_STEPS = 20_000 # 33.3min

# envs.append(MultiVenueSpreadEnv(
#     l2_tensor=train_eth,
#     ohlcv_tensor=eth_ohlcv_tensor,
#     fee_rate=FEE,
#     max_steps=MAX_STEPS,
#     invalid_action_penalty=0.0,
#     device=device,
# ))

envs.append(MultiVenueSpreadEnv(
    l2_tensor=train_sol,
    ohlcv_tensor=sol_ohlcv_tensor,
    fee_rate=FEE,
    max_steps=MAX_STEPS,
    invalid_action_penalty=0.0,
    device=device,
))

envs.append(MultiVenueSpreadEnv(
    l2_tensor=train_pepe,
    ohlcv_tensor=pepe_ohlcv_tensor,
    fee_rate=FEE,
    max_steps=MAX_STEPS,
    invalid_action_penalty=0.0,
    device=device,
))

envs.append(MultiVenueSpreadEnv(
    l2_tensor=train_sui,
    ohlcv_tensor=sui_ohlcv_tensor,
    fee_rate=FEE,
    max_steps=MAX_STEPS,
    invalid_action_penalty=0.0,
    device=device,
))

envs.append(MultiVenueSpreadEnv(
    l2_tensor=train_xrp,
    ohlcv_tensor=xrp_ohlcv_tensor,
    fee_rate=FEE,
    max_steps=MAX_STEPS,
    invalid_action_penalty=0.0,
    device=device,
))

q_net, target_net = train_dqn_multi_env(
    envs=envs,
    num_steps=1_000_000,
    warmup_steps=50_000,
    batch_size=512,
    gamma=0.99,
    lr=1e-4,
    buffer_capacity=1_000_000,
    target_update_freq=5_000,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay_steps=800_000,
    device=device,
    save_path_q="q_net_run2.pth",
    save_path_target="target_net_run2.pth",
)


def eval_dqn_multi_env(
    envs,
    q_net: torch.nn.Module,
    device: str = "cuda",
    episodes_per_env: int = 5,
    max_steps_per_episode: int | None = None,
):
    q_net.eval()
    results = {}

    for env_idx, env in enumerate(envs):
        env_name = f"env_{env_idx}"
        ep_rewards = []
        ep_num_trades = []
        ep_avg_trade_len = []
        ep_frac_in_pos = []

        for ep in tqdm(range(episodes_per_env)):
            obs, info = env.reset()
            if isinstance(obs, torch.Tensor):
                state = obs.to(device)
            else:
                state = torch.tensor(obs, dtype=torch.float32, device=device)

            done = False
            truncated = False
            ep_reward = 0.0
            steps = 0

            # metrics
            num_trades = 0
            time_in_pos_steps = 0 
            curr_trade_len = 0
            trade_lens = []

            while not (done or truncated):
                pos_before = env.curr_position

                with torch.no_grad():
                    q_values = q_net(state.unsqueeze(0))
                    action = int(torch.argmax(q_values, dim=1).item())

                next_obs, reward, done, truncated, info = env.step(action)

                ep_reward += reward
                steps += 1

                if isinstance(next_obs, torch.Tensor):
                    state = next_obs.to(device)
                else:
                    state = torch.tensor(next_obs, dtype=torch.float32, device=device)

                pos_after = env.curr_position

                if pos_after == 1:
                    time_in_pos_steps += 1
                    curr_trade_len += 1

                if pos_before == 0 and pos_after == 1:
                    num_trades += 1
                    curr_trade_len = 1  # первый шаг в трейде

                if pos_before == 1 and pos_after == 0:
                    if curr_trade_len > 0:
                        trade_lens.append(curr_trade_len)
                        curr_trade_len = 0

                if max_steps_per_episode is not None and steps >= max_steps_per_episode:
                    break

            if env.curr_position == 1 and curr_trade_len > 0:
                trade_lens.append(curr_trade_len)

            ep_rewards.append(ep_reward)
            ep_num_trades.append(num_trades)
            if len(trade_lens) > 0:
                ep_avg_trade_len.append(float(np.mean(trade_lens)))
            else:
                ep_avg_trade_len.append(0.0)

            frac_in_pos = time_in_pos_steps / max(steps, 1)
            ep_frac_in_pos.append(frac_in_pos)

        ep_rewards = np.array(ep_rewards, dtype=float)
        ep_num_trades = np.array(ep_num_trades, dtype=float)
        ep_avg_trade_len = np.array(ep_avg_trade_len, dtype=float)
        ep_frac_in_pos = np.array(ep_frac_in_pos, dtype=float)

        results[env_name] = {
            "episodes": episodes_per_env,
            "mean_ep_reward": float(ep_rewards.mean()),
            "std_ep_reward": float(ep_rewards.std()),
            "all_ep_rewards": ep_rewards.tolist(),
            "mean_num_trades": float(ep_num_trades.mean()),
            "std_num_trades": float(ep_num_trades.std()),
            "mean_trade_len": float(ep_avg_trade_len.mean()),
            "std_trade_len": float(ep_avg_trade_len.std()),
            "mean_frac_in_pos": float(ep_frac_in_pos.mean()),
            "std_frac_in_pos": float(ep_frac_in_pos.std()),
        }

    return results

device = "cuda" if torch.cuda.is_available() else "cpu"
q_net.to(device)

eval_results = eval_dqn_multi_env(
    envs=envs,
    q_net=q_net,
    device=device,
    episodes_per_env=16,
)

for k, v in eval_results.items():
    flat_frac = 1.0 - v["mean_frac_in_pos"]
    print(
        k,
        f"mean_ep_reward={v['mean_ep_reward']:.3f}",
        f"trades/ep={v['mean_num_trades']:.2f}",
        f"avg_trade_len={v['mean_trade_len']:.2f} steps",
        f"%time_in_pos={100*v['mean_frac_in_pos']:.1f}%",
        f"%time_flat={100*flat_frac:.1f}%",
    )

envs_eval = []

envs_eval.append(MultiVenueSpreadEnv(
    l2_tensor=test_eth,
    ohlcv_tensor=eth_ohlcv_tensor,
    fee_rate=FEE,
    max_steps=MAX_STEPS,
    invalid_action_penalty=0.0,
    device=device,
))

envs_eval.append(MultiVenueSpreadEnv(
    l2_tensor=test_sol,
    ohlcv_tensor=sol_ohlcv_tensor,
    fee_rate=FEE,
    max_steps=MAX_STEPS,
    invalid_action_penalty=0.0,
    device=device,
))

envs_eval.append(MultiVenueSpreadEnv(
    l2_tensor=test_pepe,
    ohlcv_tensor=pepe_ohlcv_tensor,
    fee_rate=FEE,
    max_steps=MAX_STEPS,
    invalid_action_penalty=0.0,
    device=device,
))

envs_eval.append(MultiVenueSpreadEnv(
    l2_tensor=test_sui,
    ohlcv_tensor=sui_ohlcv_tensor,
    fee_rate=FEE,
    max_steps=MAX_STEPS,
    invalid_action_penalty=0.0,
    device=device,
))

envs_eval.append(MultiVenueSpreadEnv(
    l2_tensor=test_xrp,
    ohlcv_tensor=xrp_ohlcv_tensor,
    fee_rate=FEE,
    max_steps=MAX_STEPS,
    invalid_action_penalty=0.0,
    device=device,
))

device = "cuda" if torch.cuda.is_available() else "cpu"
q_net.to(device)

eval_results = eval_dqn_multi_env(
    envs=envs_eval,
    q_net=q_net,
    device=device,
    episodes_per_env=16,
)

for k, v in eval_results.items():
    flat_frac = 1.0 - v["mean_frac_in_pos"]
    print(
        k,
        f"mean_ep_reward={v['mean_ep_reward']:.3f}",
        f"trades/ep={v['mean_num_trades']:.2f}",
        f"avg_trade_len={v['mean_trade_len']:.2f} steps",
        f"%time_in_pos={100*v['mean_frac_in_pos']:.1f}%",
        f"%time_flat={100*flat_frac:.1f}%",
    )

