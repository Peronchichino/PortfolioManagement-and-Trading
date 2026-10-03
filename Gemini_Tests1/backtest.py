"""
backtest.py - Event-driven, leakage-free backtesting engine for daily equity strategies.

Execution Protocol:
1. Signal Generation: At close of day t, signals are generated strictly using historical data
   up to and including close of day t. No future data is accessible.
2. Trade Execution: Orders are executed at the Open of day t+1.
   - Long Entry: buy at Adj_Open_{t+1} * (1 + slippage_bps / 10000)
   - Exit: sell at Adj_Open_{t+1} * (1 - slippage_bps / 10000)
3. Transaction Costs: Commission of $0.005/share (min $1.00 per order) + execution slippage.
4. Valuation: At the close of each day t, the portfolio is marked to market using Adj_Close_t.
"""

import numpy as np
import pandas as pd
from typing import Callable, Optional


class BacktestEngine:
    def __init__(
        self,
        starting_capital: float = 100000.0,
        slippage_bps: float = 2.0,          # 2 bps default for SPY (tight ETF)
        commission_per_share: float = 0.005, # $0.005 per share
        min_commission: float = 1.0,         # $1.00 minimum commission
        allow_fractional_shares: bool = False
    ):
        self.starting_capital = starting_capital
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.allow_fractional_shares = allow_fractional_shares

    def calculate_commission(self, shares: float) -> float:
        """Calculate commission with minimum threshold."""
        return max(self.min_commission, shares * self.commission_per_share)

    def run(
        self,
        df: pd.DataFrame,
        signals_df: pd.DataFrame,
        eval_start_date: Optional[str] = None,
        eval_end_date: Optional[str] = None
    ) -> tuple[pd.DataFrame, list]:
        """
        Run leakage-free backtest.
        
        Parameters:
        - df: DataFrame with columns ['Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Volume']
        - signals_df: DataFrame with 'target_position' column (0.0 to 1.0), indexed by Date.
                      target_position generated at close of day t.
        - eval_start_date: First date to evaluate performance.
        - eval_end_date: Last date to evaluate performance.
        
        Returns:
        - history_df: Daily portfolio tracking (Date, Cash, Shares, Portfolio Value, etc.)
        - trades: List of completed round-trip trades.
        """
        # Ensure chronological alignment
        common_idx = df.index.intersection(signals_df.index)
        data = df.loc[common_idx].copy()
        sig = signals_df.loc[common_idx].copy()

        cash = self.starting_capital
        shares = 0
        current_trade = None
        trades = []
        history = []

        n_bars = len(data)
        dates = data.index

        for i in range(n_bars):
            curr_date = dates[i]
            open_p = data["Adj_Open"].iloc[i]
            high_p = data["Adj_High"].iloc[i]
            low_p = data["Adj_Low"].iloc[i]
            close_p = data["Adj_Close"].iloc[i]

            # 1. TRADE EXECUTION AT TODAY'S OPEN
            # Decision was determined at close of yesterday (i - 1)
            if i > 0:
                target_pos = sig["target_position"].iloc[i - 1]
                
                # Check for Entry: Currently flat (shares == 0) and target_pos > 0
                if shares == 0 and target_pos > 0:
                    fill_price = open_p * (1.0 + self.slippage_bps / 10000.0)
                    alloc_capital = cash * target_pos
                    
                    if self.allow_fractional_shares:
                        target_shares = alloc_capital / fill_price
                    else:
                        # Allow room for commission
                        target_shares = int((alloc_capital - self.min_commission) / fill_price)
                        
                    if target_shares > 0:
                        comm = self.calculate_commission(target_shares)
                        total_cost = (target_shares * fill_price) + comm
                        if total_cost <= cash:
                            shares = target_shares
                            cash -= total_cost
                            current_trade = {
                                "entry_date": curr_date,
                                "entry_price": fill_price,
                                "shares": shares,
                                "commission_paid": comm,
                                "slippage_paid": shares * open_p * (self.slippage_bps / 10000.0)
                            }

                # Check for Exit: Currently long (shares > 0) and target_pos == 0
                elif shares > 0 and target_pos == 0:
                    fill_price = open_p * (1.0 - self.slippage_bps / 10000.0)
                    comm = self.calculate_commission(shares)
                    proceeds = (shares * fill_price) - comm
                    slippage = shares * open_p * (self.slippage_bps / 10000.0)
                    
                    # Record completed trade
                    if current_trade is not None:
                        gross_pnl = (fill_price - current_trade["entry_price"]) * shares
                        total_comm = current_trade["commission_paid"] + comm
                        total_slip = current_trade["slippage_paid"] + slippage
                        net_pnl = gross_pnl - comm  # entry comm already paid from cash
                        holding_days = (curr_date - current_trade["entry_date"]).days
                        
                        trade_record = {
                            "entry_date": current_trade["entry_date"],
                            "exit_date": curr_date,
                            "entry_price": current_trade["entry_price"],
                            "exit_price": fill_price,
                            "shares": shares,
                            "gross_pnl": gross_pnl,
                            "net_pnl": net_pnl,
                            "pnl": net_pnl,
                            "pnl_pct": (fill_price - current_trade["entry_price"]) / current_trade["entry_price"],
                            "total_comm": total_comm,
                            "total_slip": total_slip,
                            "holding_days": holding_days
                        }
                        trades.append(trade_record)
                        current_trade = None
                        
                    cash += proceeds
                    shares = 0

            # 2. MARK TO MARKET AT TODAY'S CLOSE
            port_val = cash + (shares * close_p)
            history.append({
                "Date": curr_date,
                "cash": cash,
                "shares": shares,
                "position": 1 if shares > 0 else 0,
                "portfolio_value": port_val,
                "close_price": close_p
            })

        # Close out any open trade at the final bar close
        if shares > 0 and current_trade is not None:
            final_p = data["Adj_Close"].iloc[-1]
            comm = self.calculate_commission(shares)
            proceeds = (shares * final_p) - comm
            gross_pnl = (final_p - current_trade["entry_price"]) * shares
            net_pnl = gross_pnl - comm
            trades.append({
                "entry_date": current_trade["entry_date"],
                "exit_date": dates[-1],
                "entry_price": current_trade["entry_price"],
                "exit_price": final_p,
                "shares": shares,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "pnl": net_pnl,
                "pnl_pct": (final_p - current_trade["entry_price"]) / current_trade["entry_price"],
                "total_comm": current_trade["commission_paid"] + comm,
                "total_slip": current_trade["slippage_paid"],
                "holding_days": (dates[-1] - current_trade["entry_date"]).days
            })
            cash += proceeds
            shares = 0

        hist_df = pd.DataFrame(history).set_index("Date")
        hist_df["daily_return"] = hist_df["portfolio_value"].pct_change().fillna(0.0)
        running_max = hist_df["portfolio_value"].cummax()
        hist_df["drawdown"] = (hist_df["portfolio_value"] - running_max) / running_max

        # Filter by evaluation date window if requested
        if eval_start_date:
            hist_df = hist_df.loc[hist_df.index >= eval_start_date]
            trades = [t for t in trades if pd.to_datetime(t["entry_date"]) >= pd.to_datetime(eval_start_date)]
        if eval_end_date:
            hist_df = hist_df.loc[hist_df.index <= eval_end_date]
            trades = [t for t in trades if pd.to_datetime(t["exit_date"]) <= pd.to_datetime(eval_end_date)]

        return hist_df, trades
