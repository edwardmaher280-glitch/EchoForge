"""
EchoForge Trading Environment v2
Gymnasium-compatible env for agentic RL research
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

class EchoForgeTradingEnv(gym.Env):
    """
    A multi-asset, transaction-cost aware trading env.
    Observation: [balance_norm, position, unrealized_pnl, 5x OHLCV returns + RSI, MACD, vol]
    Action: Discrete(3) -> 0: HOLD, 1: BUY (25% capital), 2: SELL (25% position)
             or Box continuous for portfolio weights (see Continuous variant below)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        initial_balance: float = 10000.0,
        commission: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,
        window_size: int = 50,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.window_size = window_size
        self.render_mode = render_mode

        # Synthetic data if none provided (for quick testing)
        if df is None:
            np.random.seed(42)
            n = 5000
            prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
            df = pd.DataFrame({
                'close': prices,
                'open': prices * (1 + np.random.randn(n)*0.002),
                'high': prices * (1 + np.abs(np.random.randn(n))*0.005),
                'low': prices * (1 - np.abs(np.random.randn(n))*0.005),
                'volume': np.random.lognormal(10, 0.5, n)
            })

        self.df = df.reset_index(drop=True)
        self._compute_features()

        # Action: 0 Hold, 1 Buy, 2 Sell
        self.action_space = spaces.Discrete(3)

        # Obs: balance(1) + position(1) + pnl(1) + market_features(12) * window
        obs_dim = 3 + 12
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.reset()

    def _compute_features(self):
        df = self.df
        df['returns'] = df['close'].pct_change().fillna(0)
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50) / 100.0
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = (ema12 - ema26) / df['close']
        df['macd'] = df['macd'].fillna(0)
        # Volatility
        df['vol'] = df['returns'].rolling(20).std().fillna(0)
        # Volume norm
        df['vol_norm'] = (df['volume'] / df['volume'].rolling(50).mean()).fillna(1).clip(0, 5)
        # OHLC relative
        df['open_ret'] = (df['open'] / df['close'] - 1).fillna(0)
        df['high_ret'] = (df['high'] / df['close'] - 1).fillna(0)
        df['low_ret'] = (df['low'] / df['close'] - 1).fillna(0)

        self.df = df.fillna(0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0  # shares held
        self.avg_entry = 0.0
        self.total_trades = 0
        self.max_balance = self.initial_balance
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        # Normalized balance
        balance_norm = np.log(self.balance / self.initial_balance)
        pos_norm = self.position * row['close'] / self.initial_balance
        unrealized = 0.0
        if self.position > 0:
            unrealized = (row['close'] - self.avg_entry) / self.avg_entry

        market = np.array([
            row['returns'], row['log_ret'], row['rsi'], row['macd'],
            row['vol'], row['vol_norm'], row['open_ret'], row['high_ret'],
            row['low_ret'],
            self.df.iloc[self.current_step-5:self.current_step]['returns'].mean(),
            self.df.iloc[self.current_step-20:self.current_step]['returns'].mean(),
            self.df.iloc[self.current_step-50:self.current_step]['returns'].std() if self.current_step >= 50 else 0
        ], dtype=np.float32)

        obs = np.concatenate([
            np.array([balance_norm, pos_norm, unrealized], dtype=np.float32),
            market
        ])
        return obs.astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        prev_val = self.balance + self.position * self.df.iloc[self.current_step]['close']
        price = self.df.iloc[self.current_step]['close']
        executed = False

        if action == 1:  # BUY 25% of balance
            cost = self.balance * 0.25
            if cost > 10:
                shares = cost / (price * (1 + self.slippage))
                fee = cost * self.commission
                self.position += shares
                self.balance -= (cost + fee)
                self.avg_entry = (self.avg_entry * (self.position - shares) + price * shares) / self.position if self.position > 0 else price
                executed = True
                self.total_trades += 1

        elif action == 2:  # SELL 25% of position
            if self.position > 1e-6:
                shares = self.position * 0.25
                revenue = shares * price * (1 - self.slippage)
                fee = revenue * self.commission
                self.balance += (revenue - fee)
                self.position -= shares
                executed = True
                self.total_trades += 1

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        curr_price = self.df.iloc[self.current_step]['close'] if not done else price
        curr_val = self.balance + self.position * curr_price
        reward = np.log(curr_val / prev_val) if prev_val > 0 else 0.0

        # Penalize excessive trading
        if executed:
            reward -= 0.0002

        # Drawdown penalty
        self.max_balance = max(self.max_balance, curr_val)
        drawdown = (self.max_balance - curr_val) / self.max_balance
        reward -= drawdown * 0.1

        terminated = done
        truncated = False

        info = {
            "portfolio_value": curr_val,
            "balance": self.balance,
            "position": self.position,
            "drawdown": drawdown,
            "trades": self.total_trades
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"Step {self.current_step} | Value: {self.balance + self.position * self.df.iloc[self.current_step]['close']:.2f} | Pos: {self.position:.4f}")

# Continuous weight variant for PPO/SAC
class EchoForgeContinuousEnv(EchoForgeTradingEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def step(self, action):
        # Map -1..1 to -0.25..0.25 portfolio delta
        target_delta = float(action[0]) * 0.25
        discrete = 0
        if target_delta > 0.05:
            discrete = 1
        elif target_delta < -0.05:
            discrete = 2
        return super().step(discrete)
