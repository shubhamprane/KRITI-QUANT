import cvxpy as cp
import pandas as pd
import numpy as np
import os
import cvxportfolio as cvx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

INITIAL_CAPITAL      = 5_000_000
TXN_COST_PCT         = 0.00268
MAX_TOTAL_POSITIONS  = 100
OUT_DIR              = "strategy_output"
os.makedirs(OUT_DIR, exist_ok=True)

IPO_MIN_SLOTS        = 5
IPO_MAX_SLOTS_CAP    = 20
IPO_TRAILING_STOP    = 0.20
IPO_DAY_CHECK        = 5
IPO_MAX_PRICE        = 40.0
IPO_MAX_TRADES       = 10
IPO_COOLDOWN         = 5
IPO_REENTRY_RISE     = 0.10
IPO_MAX_POS_PCT      = 0.10

DEFAULT_IPO_PCT      = 0.30
DEFAULT_PF_PCT       = 0.70
LARGE_UNIV_IPO_PCT   = 0.10
LARGE_UNIV_PF_PCT    = 0.90
LARGE_UNIV_THRESHOLD = 1200

IPO_DD_HALF          = 0.3
IPO_DD_FULL          = 0.4

print("=" * 70)
print("  KRITI 2026 COMBINED STRATEGY")
print("=" * 70)

# ── Data Loading ──
print("\n[1/7] Loading data...")
DATA_FILE = "nse_prices_complete 1.parquet"
if os.path.exists(DATA_FILE):
    df_raw = pd.read_parquet(DATA_FILE)
elif os.path.exists("nse_prices_complete.parquet"):
    df_raw = pd.read_parquet("nse_prices_complete.parquet")
elif os.path.exists("nse_prices_complete.csv"):
    df_raw = pd.read_csv("nse_prices_complete.csv")
else:
    raise FileNotFoundError("No data file found.")


IS_DEV = 'tradedate' in df_raw.columns or 'fid' in df_raw.columns
IS_COMP = 'date' in df_raw.columns and 'symbol' in df_raw.columns

if IS_DEV:
    df_dev = df_raw.copy()
    cm = {}
    if 'tradedate' in df_dev.columns:       cm['tradedate'] = 'date'
    if 'fid' in df_dev.columns:             cm['fid'] = 'symbol'
    if 'traded_value' in df_dev.columns:    cm['traded_value'] = 'value'
    if 'traded_volume' in df_dev.columns:   cm['traded_volume'] = 'volume'
    if 'gics_sector' in df_dev.columns:     cm['gics_sector'] = 'sector'
    if 'cap_classification' in df_dev.columns: cm['cap_classification'] = 'lms'
    if 'in_NSE500' in df_dev.columns:       cm['in_NSE500'] = 'in_nse500'
    df_std = df_raw.rename(columns=cm).copy()
elif IS_COMP:
    df_std = df_raw.copy()
    cm_r = {}
    if 'date' in df_std.columns:      cm_r['date'] = 'tradedate'
    if 'symbol' in df_std.columns:    cm_r['symbol'] = 'fid'
    if 'value' in df_std.columns:     cm_r['value'] = 'traded_value'
    if 'volume' in df_std.columns:    cm_r['volume'] = 'traded_volume'
    if 'sector' in df_std.columns:    cm_r['sector'] = 'gics_sector'
    if 'lms' in df_std.columns:       cm_r['lms'] = 'cap_classification'
    if 'in_nse500' in df_std.columns: cm_r['in_nse500'] = 'in_NSE500'
    df_dev = df_raw.rename(columns=cm_r).copy()
else:
    raise ValueError("Cannot detect data format.")

df_dev['tradedate'] = pd.to_datetime(df_dev['tradedate'])
df_dev = df_dev.sort_values(['fid', 'tradedate']).reset_index(drop=True)
df_std['date'] = pd.to_datetime(df_std['date'])
df_std = df_std.sort_values(['symbol', 'date']).reset_index(drop=True)
df_std['exec_price'] = (df_std['open'] + df_std['high'] + df_std['low'] + df_std['close']) / 4.0
if 'in_nse500' in df_std.columns:
    df_std['in_nse500'] = df_std['in_nse500'].astype(bool)

all_dates = sorted(df_std['date'].unique())
date2idx  = {d: i for i, d in enumerate(all_dates)}
N         = len(all_dates)
dates_idx = pd.DatetimeIndex(all_dates)
first_date, last_date = all_dates[0], all_dates[-1]
nyrs = (pd.Timestamp(last_date) - pd.Timestamp(first_date)).days / 365.25
day1_count = df_std[df_std['date'] == first_date]['symbol'].nunique()

print(f"  {str(first_date)[:10]} -> {str(last_date)[:10]} ({nyrs:.1f} yrs), {N} days")
print(f"  Stocks: {df_std['symbol'].nunique()} | Day-1: {day1_count}")

if day1_count > LARGE_UNIV_THRESHOLD:
    IPO_ALLOC, PF_ALLOC = LARGE_UNIV_IPO_PCT, LARGE_UNIV_PF_PCT
else:
    IPO_ALLOC, PF_ALLOC = DEFAULT_IPO_PCT, DEFAULT_PF_PCT
print(f"  Allocation: {IPO_ALLOC*100:.0f}% IPO / {PF_ALLOC*100:.0f}% Portfolio")
ipo_capital = INITIAL_CAPITAL * IPO_ALLOC
pf_capital  = INITIAL_CAPITAL * PF_ALLOC


# ═══════════════════════════════════════════════════════════════════
# [2/7] IPO STRATEGY WITH DD-BASED LIQUIDATION
# ═══════════════════════════════════════════════════════════════════
print("\n[2/7] Running IPO strategy...")
df = df_std
IPO_MAX_SLOTS = int(np.clip(np.round(5 + nyrs), IPO_MIN_SLOTS, IPO_MAX_SLOTS_CAP))
first_app = df.groupby('symbol')['date'].min().reset_index()
first_app.columns = ['symbol', 'first_date']
ipo_symbols = set(first_app.loc[first_app['first_date'] > first_date, 'symbol'])

df_ipo = df[df['symbol'].isin(ipo_symbols)].copy()
close_piv_ipo = df_ipo.pivot_table('close', 'date', 'symbol', 'first').reindex(all_dates).ffill()
exec_piv_ipo  = df_ipo.pivot_table('exec_price', 'date', 'symbol', 'first').reindex(all_dates).ffill()

sym_first = {}
for _, r in first_app[first_app['symbol'].isin(ipo_symbols)].iterrows():
    if r['first_date'] in date2idx:
        sym_first[r['symbol']] = date2idx[r['first_date']]

ipo_valid = {}
for sym, fi in sym_first.items():
    d5 = fi + IPO_DAY_CHECK - 1
    if d5 < N and sym in close_piv_ipo.columns:
        c5 = close_piv_ipo.iat[d5, close_piv_ipo.columns.get_loc(sym)]
        if pd.notna(c5) and 0 < c5 < IPO_MAX_PRICE:
            ipo_valid[sym] = d5

print(f"  IPOs: {len(ipo_symbols)} total, {len(ipo_valid)} valid, slots: {IPO_MAX_SLOTS}")
ipo_col_idx = {sym: close_piv_ipo.columns.get_loc(sym) for sym in close_piv_ipo.columns}

def _ic(sym, t):
    if sym not in ipo_col_idx: return None
    v = close_piv_ipo.iat[t, ipo_col_idx[sym]]
    return v if pd.notna(v) and v > 0 else None

def _ie(sym, t):
    if sym not in ipo_col_idx: return None
    v = exec_piv_ipo.iat[t, ipo_col_idx[sym]]
    return v if pd.notna(v) and v > 0 else None


def run_ipo_strategy(initial_cash):
    """
    Returns: equity, cash_arr, npos_arr, turn_arr, trade_log, transfers
    transfers: list of (exec_day_index, cash_amount) for PF injection
    """
    cash = initial_cash
    positions = {}; primary_wl = {}; secondary_wl = {}; trade_counts = {}
    pending_sells = []; pending_buys = []
    equity = np.zeros(N); cash_arr = np.zeros(N)
    npos_arr = np.zeros(N, dtype=int); turn_arr = np.zeros(N); trade_log = []

    ipo_hwm = initial_cash
    half_liquidated = False
    full_liquidated = False
    pending_liq = None
    transfers = []
    flagIPO = True

    for t in range(N):
        today = all_dates[t]; dv = 0.0

        sold_syms = set()
        for sym in pending_sells:
            if sym not in positions: continue
            pos = positions[sym]
            ep = _ie(sym, t)
            if ep is None: ep = pos.get('last_close', pos['cost_basis'])
            net = ep * (1 - TXN_COST_PCT); proceeds = pos['qty'] * net
            cash += proceeds; dv += pos['qty'] * ep
            trade_log.append(dict(symbol=sym, action='SELL', date=str(today)[:10],
                                  price=round(ep,2), qty=pos['qty'], value=round(proceeds,2)))
            tc = trade_counts.get(sym, 0)
            if tc < IPO_MAX_TRADES:
                primary_wl[sym] = {'sell_px': ep, 't_exit': t, 'tc': tc}
            sold_syms.add(sym); del positions[sym]
        pending_sells.clear()

        buys = [b for b in pending_buys if b[0] not in positions and b[0] not in sold_syms]
        pending_buys.clear()
        valid_buys = []
        for sym, ts, src in buys:
            if _ie(sym, t) is not None: valid_buys.append((sym, ts, src))
            elif src == 'new_ipo': secondary_wl.setdefault(sym, {'t_signal': ts})

        avail = IPO_MAX_SLOTS - len(positions)
        if len(valid_buys) > avail:
            spill = valid_buys[avail:]; valid_buys = valid_buys[:avail]
            for sym, ts, src in spill:
                if src == 'new_ipo': secondary_wl.setdefault(sym, {'t_signal': ts})

        if valid_buys and cash > 0:
            flagIPO = False
            pf = cash
            for f, p in positions.items():
                c = _ic(f, t)
                pf += p['qty'] * (c if c else p.get('last_close', p['cost_basis']))
            cap = IPO_MAX_POS_PCT * pf
            empty = IPO_MAX_SLOTS - len(positions)
            per_slot = cash / empty if empty > 0 else 0
            overflow = 0.0; allocs = {}
            for sym, ts, src in valid_buys:
                if per_slot > cap: allocs[sym] = cap; overflow += (per_slot - cap)
                else: allocs[sym] = per_slot
            existing_filled = list(positions.keys())
            topup = 0.0
            if overflow > 0:
                reinv = overflow * 0.90
                if existing_filled: topup = reinv / len(existing_filled)

            for sym, ts, src in valid_buys:
                ep = _ie(sym, t)
                if ep is None: continue
                bp = ep * (1 + TXN_COST_PCT); a = allocs.get(sym, 0); qty = int(a / bp)
                if qty <= 0:
                    if src == 'new_ipo': secondary_wl.setdefault(sym, {'t_signal': ts})
                    continue
                cost = qty * bp
                if cost > cash: qty = int(cash / bp); cost = qty * bp
                if qty <= 0: continue
                cash -= cost; dv += cost
                positions[sym] = {'qty': qty, 'cost_basis': bp, 'peak': ep, 't_entry': t, 'last_close': ep}
                trade_counts[sym] = trade_counts.get(sym, 0) + 1
                primary_wl.pop(sym, None); secondary_wl.pop(sym, None)
                trade_log.append(dict(symbol=sym, action='BUY', date=str(today)[:10],
                                      price=round(ep,2), qty=qty, value=round(-cost,2)))

            if topup > 0:
                for fr in existing_filled:
                    ep_r = _ie(fr, t)
                    if ep_r is None: continue
                    bp_r = ep_r * (1 + TXN_COST_PCT); aq = int(topup / bp_r)
                    if aq <= 0: continue
                    ac = aq * bp_r
                    if ac > cash: aq = int(cash / bp_r); ac = aq * bp_r
                    if aq <= 0: continue
                    old_q = positions[fr]['qty']; old_cb = positions[fr]['cost_basis']
                    new_q = old_q + aq
                    positions[fr]['cost_basis'] = (old_q * old_cb + aq * bp_r) / new_q
                    positions[fr]['qty'] = new_q; cash -= ac; dv += ac
                    trade_log.append(dict(symbol=fr, action='TOPUP', date=str(today)[:10],
                                          price=round(ep_r,2), qty=aq, value=round(-ac,2)))

        if pending_liq is not None:
            sell_pct, sig_day = pending_liq
            if sig_day + 1 == t:
                liq_proceeds = 0.0
                to_del = []
                for sym, pos in positions.items():
                    sell_qty = int(np.floor(pos['qty'] * sell_pct))
                    if sell_qty <= 0: continue
                    ep = _ie(sym, t)
                    if ep is None: ep = pos.get('last_close', pos['cost_basis'])
                    net_per = ep * (1 - TXN_COST_PCT)
                    liq_proceeds += sell_qty * net_per
                    dv += sell_qty * ep
                    pos['qty'] -= sell_qty
                    trade_log.append(dict(symbol=sym, action='LIQ_SELL', date=str(today)[:10],
                                          price=round(ep,2), qty=sell_qty,
                                          value=round(sell_qty*net_per,2)))
                    if pos['qty'] <= 0: to_del.append(sym)
                for sym in to_del: del positions[sym]
                cash_xfer = cash * sell_pct
                cash -= cash_xfer
                total_xfer = liq_proceeds + cash_xfer
                transfers.append((t, total_xfer))
                pending_liq = None

        for sym, pos in positions.items():
            cc = _ic(sym, t)
            if cc is None: continue
            pos['last_close'] = cc
            if cc > pos['peak']: pos['peak'] = cc
            dd = (pos['peak'] - cc) / pos['peak']
            if dd >= IPO_TRAILING_STOP: pending_sells.append(sym)
        pending_sells = list(set(pending_sells))
        slots_tomorrow = IPO_MAX_SLOTS - (len(positions) - len(pending_sells))

        pri_sigs = []; dead = []
        for sym, pw in primary_wl.items():
            if sym in positions: continue
            if trade_counts.get(sym, 0) >= IPO_MAX_TRADES: dead.append(sym); continue
            if (t - pw['t_exit']) < IPO_COOLDOWN: continue
            cc = _ic(sym, t)
            if cc is None: dead.append(sym); continue
            if cc >= pw['sell_px'] * (1 + IPO_REENTRY_RISE): pri_sigs.append((sym, t, 'primary'))
        for f in dead: del primary_wl[f]

        ipo_sigs = [(sym, t, 'new_ipo') for sym, d5 in ipo_valid.items()
                    if d5 == t and sym not in positions]

        sec_sigs = []; dead2 = []
        for sym, sw in secondary_wl.items():
            if sym in positions: dead2.append(sym); continue
            if _ic(sym, t) is None: dead2.append(sym); continue
            sec_sigs.append((sym, sw['t_signal'], 'secondary'))
        for f in dead2: del secondary_wl[f]

        pri_sigs.sort(key=lambda x: -x[1]); sec_sigs.sort(key=lambda x: -x[1])
        all_sigs = pri_sigs + ipo_sigs + sec_sigs
        chosen = []; used = set(pending_sells) | {b[0] for b in pending_buys}
        for sym, ts, src in all_sigs:
            if len(chosen) >= slots_tomorrow: break
            if sym in used or sym in positions: continue
            chosen.append((sym, ts, src)); used.add(sym)
        pending_buys.extend(chosen)
        for sym, ts, src in ipo_sigs:
            if sym not in used and sym not in secondary_wl and sym not in positions:
                secondary_wl[sym] = {'t_signal': ts}

        pf = cash
        for sym, pos in positions.items():
            cc = _ic(sym, t)
            if cc: pf += pos['qty'] * cc
            else: pf += pos['qty'] * pos.get('last_close', pos['cost_basis'])
        equity[t] = pf; cash_arr[t] = cash; npos_arr[t] = len(positions); turn_arr[t] = dv

        if pf > ipo_hwm: ipo_hwm = pf
        ipo_dd = (ipo_hwm - pf) / ipo_hwm if ipo_hwm > 0 else 0

        if pending_liq is None and not full_liquidated:
            if ipo_dd >= IPO_DD_FULL and not full_liquidated:

                pending_liq = (1.0, t)
                full_liquidated = True
            elif ipo_dd >= IPO_DD_HALF and not half_liquidated:
                pending_liq = (0.5, t)
                half_liquidated = True

        if pending_liq is None and not full_liquidated:
            if t == 126 and flagIPO:
                pending_liq = (1.0, t)
                full_liquidated = True

    return equity, cash_arr, npos_arr, turn_arr, trade_log, transfers


ipo_equity, ipo_cash_arr, ipo_npos, ipo_turn, ipo_trades, ipo_transfers = run_ipo_strategy(ipo_capital)
print(f"  IPO Final: {ipo_equity[-1]:,.0f}")
if ipo_transfers:
    for td, amt in ipo_transfers:
        print(f"  Transfer on day {td} ({str(all_dates[td])[:10]}): {amt:,.0f}")
else:
    print("  No DD liquidation triggered.")


# ═══════════════════════════════════════════════════════════════════
# [3/7] PORTFOLIO STRATEGY (FEATURE SELECTOR + CVXPORTFOLIO + BACKTESTER)
# ═══════════════════════════════════════════════════════════════════
print("\n[3/7] Running portfolio strategy...")

def feature_selector(df_in, enter_n=65, exit_n=80):
    df_in = df_in.copy()
    df_in['tradedate'] = pd.to_datetime(df_in['tradedate'])
    df_in = df_in.drop_duplicates(subset=['tradedate', 'fid'])
    closes = df_in.pivot(index='tradedate', columns='fid', values='close').ffill()
    highs  = df_in.pivot(index='tradedate', columns='fid', values='high').ffill()
    lows   = df_in.pivot(index='tradedate', columns='fid', values='low').ffill()
    values = df_in.pivot(index='tradedate', columns='fid', values='traded_volume').fillna(0)
    ret = np.log(closes / closes.shift(1))
    mom_long = (closes / closes.shift(60)) - 1
    mom_med  = (closes / closes.shift(20)) - 1
    volatility = ret.rolling(60).std() * np.sqrt(252)
    liquidity  = values.rolling(20).mean()
    prev_close = closes.shift(1)
    tr = np.maximum(highs - lows, np.maximum((highs - prev_close).abs(), (lows - prev_close).abs()))
    natr = (tr.rolling(14).mean() / closes) * 100
    market_ret = ret.mean(axis=1)
    beta = ret.rolling(60).cov(market_ret).div(market_ret.rolling(60).var(), axis=0).fillna(0)
    def get_zscore(factor_df):
        mean = factor_df.mean(axis=1)
        std  = factor_df.std(axis=1).replace(0, np.nan)
        return factor_df.sub(mean, axis=0).div(std, axis=0).clip(-3, 3)
    total_score = (
        (get_zscore(mom_long)   *  0.30) + (get_zscore(mom_med)    *  0.20) +
        (get_zscore(volatility) * -0.30) + (get_zscore(liquidity)  *  0.10) +
        (get_zscore(natr)       *  0.05) + (get_zscore(beta)       *  0.05)
    )
    valid = total_score.notnull() & (total_score != 0)
    total_score = total_score.where(valid)
    ranks = total_score.rank(axis=1, ascending=False)
    enter_mask = ranks <= enter_n; exit_mask = ranks >= exit_n
    universe = pd.DataFrame(False, index=ranks.index, columns=ranks.columns)
    universe.iloc[0] = enter_mask.iloc[0]
    for t in range(1, len(universe)):
        universe.iloc[t] = (enter_mask.iloc[t] | (universe.iloc[t-1] & ~exit_mask.iloc[t]))
    return universe.fillna(False)


class Backtester:
    def __init__(self, price_df, allocation_df, transaction_cost=0.00268,
                 initial_capital=5000000, cash_injections=None):
        """
        cash_injections: dict {pd.Timestamp: float} — extra cash from IPO DD liquidation.
        Accumulated and deployed proportionally on next rebalance.
        """
        self.price_df = price_df.sort_index().copy()
        self.allocation_df = allocation_df.sort_index().copy()
        self.price_df.index = pd.to_datetime(self.price_df.index)
        self.allocation_df.index = pd.to_datetime(self.allocation_df.index)
        self.allocation_df = self.allocation_df.fillna(0)
        self.asset_cols = [c for c in self.allocation_df.columns if c != 'cash']
        missing = set(self.asset_cols) - set(self.price_df.columns)
        if missing: raise ValueError(f"Missing price data for: {missing}")
        self.tc = transaction_cost
        self.initial_capital = initial_capital
        self.cash_injections = cash_injections or {}

    def enforce_no_leverage(self, allocation_df):
        allocation_df = allocation_df.copy()
        stock_cols = [c for c in allocation_df.columns if c != "cash"]
        for date in allocation_df.index:
            cash_val = allocation_df.at[date, "cash"]
            if cash_val >= 0: continue
            deficit = -cash_val
            stock_total = allocation_df.loc[date, stock_cols].sum()
            if stock_total <= 0: allocation_df.at[date, "cash"] = 0; continue
            scale = max((stock_total - deficit) / stock_total, 0)
            allocation_df.loc[date, stock_cols] *= scale
            allocation_df.at[date, "cash"] = 0.0
        return allocation_df

    def preprocess_allocations(self):
        processed = self.allocation_df.copy()
        stock_cols = [c for c in processed.columns if c != 'cash']
        processed[stock_cols] = processed[stock_cols].clip(lower=0)
        return processed

    def enforce_max_positions(self, allocation_df, max_positions=80):
        processed = allocation_df.copy()
        stock_cols = [c for c in processed.columns if c != 'cash']
        for date in processed.index:
            row = processed.loc[date, stock_cols]
            non_zero = row[row > 0]
            if len(non_zero) <= max_positions: continue
            top_assets = non_zero.nlargest(max_positions).index
            drop_assets = non_zero.index.difference(top_assets)
            dropped_amount = processed.loc[date, drop_assets].sum()
            processed.loc[date, drop_assets] = 0
            processed.loc[date, 'cash'] += dropped_amount
        return processed

    def initialize_with_minimum_position(self, allocation_df):
        allocation_df = allocation_df.copy()
        first_price_date = self.price_df.index.min()
        first_alloc_date = first_price_date + pd.Timedelta(days=100)
        if first_alloc_date <= first_price_date: return allocation_df
        first_prices = self.price_df.loc[first_price_date, self.asset_cols]
        cheapest_asset = first_prices.idxmin()
        cheapest_price = first_prices[cheapest_asset]
        pre_alloc_dates = self.price_df.index[self.price_df.index < first_alloc_date]
        for date in pre_alloc_dates:
            if date in allocation_df.index: continue
            new_row = pd.Series(0.0, index=allocation_df.columns)
            new_row[cheapest_asset] = cheapest_price
            new_row["cash"] = self.initial_capital - cheapest_price
            allocation_df.loc[date] = new_row
        allocation_df = allocation_df.sort_index()
        return allocation_df

    def run(self):
        allocation_df = self.preprocess_allocations()
        allocation_df = self.enforce_no_leverage(allocation_df)
        allocation_df = self.enforce_max_positions(allocation_df, max_positions=79)
        allocation_df = self.initialize_with_minimum_position(allocation_df)
        allocation_df=allocation_df[:-1]
        self.allocation_df = allocation_df

        start_date = allocation_df.index.min()
        prices_bt = self.price_df.loc[self.price_df.index >= start_date].copy()
        price_index = prices_bt.index

        execution_map = {}
        for signal_date in allocation_df.index:
            if signal_date not in price_index: continue
            loc = price_index.get_loc(signal_date)
            if loc + 1 >= len(price_index): continue
            execution_map[price_index[loc + 1]] = signal_date

        shares = pd.DataFrame(0, index=price_index, columns=self.asset_cols, dtype=float)
        cash_series = pd.Series(0.0, index=price_index)
        turnover_series = pd.Series(0.0, index=price_index)

        prev_shares = pd.Series(0.0, index=self.asset_cols)
        prev_cash = 0.0
        extra_cash = 0.0

        for date in price_index:
            current_shares = prev_shares.copy()
            current_cash = prev_cash

            if date in self.cash_injections:
                inj = self.cash_injections[date]
                current_cash+=inj
                idx = allocation_df.index.searchsorted(date, side='left')
                if idx < len(allocation_df.index):
                    actual_date = allocation_df.index[idx]
                    curr_port=allocation_df.loc[actual_date].sum()
                    factor=(curr_port+inj)/curr_port
                    pos = allocation_df.index.get_loc(actual_date)
                    allocation_df.iloc[pos-2:] *= factor
                    self.allocation_df=allocation_df


            if date in execution_map:
                signal_date = execution_map[date]
                alloc_row = allocation_df.loc[signal_date, self.asset_cols].copy()
                base_cash = float(allocation_df.loc[signal_date, "cash"])
                execution_prices = prices_bt.loc[date, self.asset_cols]

            

                target_shares = np.floor(alloc_row / execution_prices)

                zero_mask = (target_shares == 0)
                if zero_mask.any():
                    candidate_prices = execution_prices[zero_mask]
                    cheapest_asset = candidate_prices.idxmin()
                else:
                    cheapest_asset = execution_prices.idxmin()
                target_shares[cheapest_asset] = 1

                trade_shares = target_shares - prev_shares
                trade_value = trade_shares * execution_prices
                cost = np.abs(trade_value).sum() * self.tc
                current_cash = base_cash - cost

                if current_cash < 0:
                    held = target_shares[target_shares > 0].copy()
                    held_vals = (held * execution_prices[held.index]).sort_values()
                    for asset in held_vals.index:
                        if current_cash >= 0:
                            break
                        base_cash += target_shares[asset] * execution_prices[asset]
                        target_shares[asset] = 0
                        trade_shares = target_shares - prev_shares
                        trade_value = trade_shares * execution_prices
                        cost = np.abs(trade_value).sum() * self.tc
                        current_cash = base_cash - cost

                turnover_series.loc[date] = np.abs((target_shares - prev_shares) * execution_prices).sum()
                current_shares = target_shares

            shares.loc[date] = current_shares
            cash_series.loc[date] = current_cash
            prev_shares = current_shares
            prev_cash = current_cash

        asset_value = shares * prices_bt[self.asset_cols]
        portfolio_value = asset_value.sum(axis=1) + cash_series
        daily_returns = portfolio_value.pct_change().fillna(0)

        self.portfolio_value = portfolio_value[1:]
        self.daily_returns = daily_returns[1:]
        self.shares = shares
        self.cash_series = cash_series[1:]
        self.turnover_series = turnover_series[1:]
        return portfolio_value, daily_returns


# ── Feature selection + cvxportfolio ──
print("  Feature selection...")
df_dev['tradedate'] = pd.to_datetime(df_dev['tradedate'])
df_dev['price'] = (df_dev['open'] + df_dev['high'] + df_dev['low'] + df_dev['close']) / 4

prices_shifted = df_dev.pivot(index='tradedate', columns='fid', values='price').ffill().bfill().shift(-1)
prices_pf      = df_dev.pivot(index='tradedate', columns='fid', values='price')
volumes_pf     = df_dev.pivot(index='tradedate', columns='fid', values='traded_volume')
volumes_pf     = volumes_pf[prices_pf.columns]

returns_pf = prices_pf.pct_change()
returns_pf['cash'] = 0.0
is_alive = volumes_pf > 0

bitsmask = feature_selector(df_dev.copy(), enter_n=65, exit_n=79) & is_alive
print(f"  Universe: {bitsmask.sum(axis=1).min()}-{bitsmask.sum(axis=1).max()}")

print("  cvxportfolio optimization...")
objective = cvx.ReturnsForecast() - 0.05*cvx.ReturnsForecastError() - 15 * (
    cvx.WorstCaseRisk([cvx.DiagonalCovariance(), cvx.FactorModelCovariance()])
    + 0.01 * cvx.RiskForecastError()
)
constraints = [
    cvx.LeverageLimit(1), cvx.LongOnly(applies_to_cash=True), cvx.MaxWeights(bitsmask),
]
policy = cvx.MultiPeriodOptimization(objective, constraints,planning_horizon=2)
simulator = cvx.MarketSimulator(
    returns=returns_pf, volumes=volumes_pf, prices=prices_shifted,
    round_trades=True, cash_key='cash', trading_frequency='weekly',
    costs=[cvx.TransactionCost(a=0.00268)], min_history=pd.Timedelta('5 days')
)
print("  Running cvxportfolio backtest...")
results = simulator.backtest(policy, initial_value=pf_capital)

cash_injections = {}
for day_idx, amount in ipo_transfers:
    ts = pd.Timestamp(all_dates[day_idx])
    cash_injections[ts] = cash_injections.get(ts, 0) + amount
if cash_injections:
    print(f"  PF will receive {len(cash_injections)} cash injection(s)")

print("  Backtester execution...")
bt = Backtester(
    price_df=prices_pf, allocation_df=results.h_plus,
    transaction_cost=TXN_COST_PCT, initial_capital=pf_capital,
    cash_injections=cash_injections
)
bt.run()
pf_portfolio_value = bt.portfolio_value

pf_equity_series = pf_portfolio_value.reindex(dates_idx, method='ffill').fillna(pf_capital)
pf_equity = pf_equity_series.values
pf_npos_series = (bt.shares > 0).sum(axis=1).reindex(dates_idx, method='ffill').fillna(0)
pf_npos = pf_npos_series.values.astype(int)
pf_turn_aligned = bt.turnover_series.reindex(dates_idx).fillna(0).values

print(f"  PF Final: {pf_equity[-1]:,.0f}")


# ═══════════════════════════════════════════════════════════════════
# [4/7] COMBINE & METRICS
# ═══════════════════════════════════════════════════════════════════
print("\n[4/7] Combining sleeves...")

combined_equity = ipo_equity + pf_equity
total_npos = ipo_npos + pf_npos
total_turn = (ipo_turn + pf_turn_aligned)

print(f"  Combined Final: {combined_equity[-1]:,.0f}")
print(f"  Max positions: {total_npos.max()}")


# ═══════════════════════════════════════════════════════════════════
# [5/7] BENCHMARK
# ═══════════════════════════════════════════════════════════════════
print("\n[5/7] Benchmark...")
bench_cum = None
try:
    import yfinance as yf
    s_str = (pd.Timestamp(first_date) - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
    e_str = (pd.Timestamp(last_date) + pd.Timedelta(days=10)).strftime('%Y-%m-%d')
    for ticker in ['^CRSLDX', 'NIFTY500.NS', '^CNX500']:
        try:
            tmp = yf.download(ticker, start=s_str, end=e_str, progress=False)
            if tmp is not None and len(tmp) > 100:
                if isinstance(tmp.columns, pd.MultiIndex): tmp.columns = tmp.columns.get_level_values(0)
                bc = tmp['Close'].copy()
                bc.index = pd.to_datetime(bc.index).tz_localize(None)
                bc = bc.sort_index().reindex(dates_idx, method='ffill')
                bench_ret = bc.pct_change().fillna(0)
                bench_cum = (bc / bc.iloc[0]) * INITIAL_CAPITAL
                bench_src = f"Nifty 500 ({ticker})"; break
        except: continue
except ImportError: pass

if bench_cum is None:
    if 'in_nse500' in df_std.columns:
        nse500 = df_std[df_std['in_nse500'] == True].sort_values(['symbol', 'date'])
    else:
        nse500 = df_std.sort_values(['symbol', 'date'])
    nse500['ret'] = nse500.groupby('symbol')['close'].pct_change()
    bench_ret = nse500.groupby('date')['ret'].mean().reindex(dates_idx).fillna(0)
    bench_cum = (1 + bench_ret).cumprod() * INITIAL_CAPITAL
    bench_src = "Nifty 500 Equal-Weight Proxy"
bench_ret = bench_ret.reindex(dates_idx).fillna(0)
print(f"  {bench_src}")


# ═══════════════════════════════════════════════════════════════════
# [6/7] METRICS
# ═══════════════════════════════════════════════════════════════════
print("\n[6/7] Metrics...")
eq_s = pd.Series(combined_equity, index=dates_idx)
dr   = eq_s.pct_change().dropna()
bd   = bench_ret.reindex(dr.index).fillna(0)

cagr       = (combined_equity[-1] / INITIAL_CAPITAL) ** (1/nyrs) - 1
bench_cagr = (bench_cum.iloc[-1] / INITIAL_CAPITAL) ** (1/nyrs) - 1
ann_vol    = dr.std() * np.sqrt(252)
bench_vol  = bd.std() * np.sqrt(252)
rmx = eq_s.cummax(); dd_s = (eq_s - rmx) / rmx; max_dd = dd_s.min()
brmx = bench_cum.cummax(); bdd_s = (bench_cum - brmx) / brmx; bench_max_dd = bdd_s.min()
sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
bench_sharpe = bd.mean() / bd.std() * np.sqrt(252) if bd.std() > 0 else 0
dn_std = dr[dr < 0].std() * np.sqrt(252)
sortino = dr.mean() * 252 / dn_std if dn_std > 0 else 0
act = dr - bd; te = act.std() * np.sqrt(252)
ir = act.mean() * 252 / te if te > 0 else 0
bu = bd[bd > 0]; bdn = bd[bd < 0]
up_cap = dr.reindex(bu.index).mean() / bu.mean() * 100 if len(bu) > 0 and bu.mean() != 0 else 0
dn_cap = dr.reindex(bdn.index).mean() / bdn.mean() * 100 if len(bdn) > 0 and bdn.mean() != 0 else 0
ann_turn = total_turn.sum() / combined_equity.mean() / nyrs if combined_equity.mean() > 0 else 0

sc = eq_s / eq_s.iloc[0]
bc_n = pd.Series(bench_cum.values, index=dates_idx) / INITIAL_CAPITAL
roll = {}
for wy, lab in [(1,'1Y'),(3,'3Y'),(5,'5Y')]:
    wd = int(wy * 252)
    if len(sc) < wd: roll[lab] = (np.nan, np.nan); continue
    sr = sc.pct_change(wd).dropna(); br2 = bc_n.pct_change(wd).dropna()
    ci = sr.index.intersection(br2.index); op = sr.loc[ci] - br2.loc[ci]
    roll[lab] = (op.mean()*100, op.min()*100)


# ═══════════════════════════════════════════════════════════════════
# [7/7] REPORT & OUTPUTS
# ═══════════════════════════════════════════════════════════════════
print("\n[7/7] Outputs...")
lines = []
def rp(s=""): print(s); lines.append(s)

rp("=" * 72)
rp("  PERFORMANCE REPORT")
rp("=" * 72)
rp(f"  Period           {str(dates_idx[0].date())} -> {str(dates_idx[-1].date())} ({nyrs:.1f} yrs)")
rp(f"  Benchmark        {bench_src}")
rp(f"  Initial Capital  Rs.{INITIAL_CAPITAL:>14,}")
rp(f"  Final NAV        Rs.{combined_equity[-1]:>14,.0f}")
rp(f"  " + "-"*60)
rp(f"  {'Metric':<30} {'Strategy':>12} {'Nifty 500':>12}")
rp(f"  " + "-"*58)
rp(f"  {'CAGR':<30} {cagr*100:>11.2f}% {bench_cagr*100:>11.2f}%")
rp(f"  {'Annualized Std Dev':<30} {ann_vol*100:>11.2f}% {bench_vol*100:>11.2f}%")
rp(f"  {'Maximum Drawdown':<30} {max_dd*100:>11.2f}% {bench_max_dd*100:>11.2f}%")
rp(f"  {'Sharpe Ratio':<30} {sharpe:>12.3f} {bench_sharpe:>12.3f}")
rp(f"  {'Sortino Ratio':<30} {sortino:>12.3f}")
rp(f"  {'Information Ratio vs Nifty500':<30} {ir:>12.3f}")
rp(f"  {'Up-Capture vs Nifty500':<30} {up_cap:>11.1f}%")
rp(f"  {'Down-Capture vs Nifty500':<30} {dn_cap:>11.1f}%")
rp(f"  {'Annualized Turnover':<30} {ann_turn:>11.2f}x")
rp(f"  " + "-"*60)
rp(f"  Rolling Outperformance vs Nifty 500")
rp(f"  {'Window':<8} {'Avg Outperformance':>20} {'Worst Underperformance':>24}")
for lab, (a, w) in roll.items():
    af = f"{a:.2f}%" if not np.isnan(a) else "N/A"
    wf = f"{w:.2f}%" if not np.isnan(w) else "N/A"
    rp(f"  {lab:<8} {af:>20} {wf:>24}")
rp("=" * 72)
with open(os.path.join(OUT_DIR, 'performance_report.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(dates_idx, combined_equity/1e5, label=f'Strategy (CAGR {cagr*100:.1f}%)', color='#1565C0', lw=2)
ax.plot(dates_idx, bench_cum.values/1e5, label=f'Nifty 500 (CAGR {bench_cagr*100:.1f}%)', color='#FF8F00', lw=1.5)
ax.set_ylabel('NAV (Lakhs)'); ax.set_yscale('log')
ax.set_title('Equity Curve'); ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.savefig(os.path.join(OUT_DIR, 'equity_curve.png'), dpi=150, bbox_inches='tight'); plt.close()

fig, ax = plt.subplots(figsize=(16, 6))
ax.fill_between(dates_idx, dd_s.values*100, 0, color='#1565C0', alpha=0.35, label='Strategy')
ax.fill_between(dates_idx, bdd_s.values*100, 0, color='#FF8F00', alpha=0.2, label='Nifty 500')
ax.set_ylabel('Drawdown (%)'); ax.set_title('Drawdown Curve'); ax.legend(); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.savefig(os.path.join(OUT_DIR, 'drawdown_curve.png'), dpi=150, bbox_inches='tight'); plt.close()

fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(dates_idx, total_npos, color='#2E7D32', lw=1)
ax.axhline(MAX_TOTAL_POSITIONS, color='red', ls='--', alpha=0.5, label='Max (100)')
ax.set_ylabel('Positions'); ax.set_title('Position Count Over Time'); ax.legend(); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.savefig(os.path.join(OUT_DIR, 'position_count.png'), dpi=150, bbox_inches='tight'); plt.close()

fig, ax = plt.subplots(figsize=(16, 6))
daily_turn_pct = pd.Series(total_turn, index=dates_idx) / eq_s * 100
ax.bar(dates_idx, daily_turn_pct.values, width=1, color='#7B1FA2', alpha=0.6)
ax.set_ylabel('Daily Turnover (% of NAV)'); ax.set_title('Portfolio Turnover'); ax.grid(True, alpha=0.3, axis='y')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.savefig(os.path.join(OUT_DIR, 'turnover.png'), dpi=150, bbox_inches='tight'); plt.close()

fig, ax = plt.subplots(figsize=(16, 6))
if len(sc) > 252:
    sr_1y = sc.pct_change(252).dropna()*100; br_1y = bc_n.pct_change(252).dropna()*100
    ci_1y = sr_1y.index.intersection(br_1y.index); outperf = sr_1y.loc[ci_1y] - br_1y.loc[ci_1y]
    ax.fill_between(ci_1y, outperf.values, 0, where=outperf.values>=0, color='#43A047', alpha=0.4)
    ax.fill_between(ci_1y, outperf.values, 0, where=outperf.values<0, color='#E53935', alpha=0.4)
    ax.axhline(0, color='black', lw=0.8)
ax.set_ylabel('Rolling 1Y Outperformance (%)'); ax.set_title('Rolling 1-Year Outperformance vs Nifty 500')
ax.grid(True, alpha=0.3); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.savefig(os.path.join(OUT_DIR, 'rolling_outperformance.png'), dpi=150, bbox_inches='tight'); plt.close()

mret = dr.resample('ME').apply(lambda x: (1+x).prod()-1)
mdf = mret.to_frame('r'); mdf['yr'] = mdf.index.year; mdf['mo'] = mdf.index.month
mp = mdf.pivot_table('r', 'yr', 'mo', 'first') * 100
fig, ax = plt.subplots(figsize=(14, max(4, len(mp)*0.55)))
vm = max(15, np.nanmax(np.abs(mp.values)))
im = ax.imshow(mp.values, cmap='RdYlGn', aspect='auto', vmin=-vm, vmax=vm)
ax.set_xticks(range(12)); ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.set_yticks(range(len(mp))); ax.set_yticklabels(mp.index.astype(int))
ax.set_title('Monthly Returns (%)'); plt.colorbar(im, label='%', shrink=0.8)
for i in range(mp.shape[0]):
    for j in range(mp.shape[1]):
        v = mp.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=7,
                    color='black' if abs(v)<vm*0.6 else 'white')
plt.savefig(os.path.join(OUT_DIR, 'monthly_heatmap.png'), dpi=150, bbox_inches='tight'); plt.close()

pd.DataFrame(ipo_trades).to_csv(os.path.join(OUT_DIR, 'trade_log.csv'), index=False)
pd.DataFrame({
    'date': dates_idx, 'nav': combined_equity, 'positions': total_npos,
    'benchmark': bench_cum.values,
    'drawdown_pct': dd_s.values*100, 'benchmark_drawdown_pct': bdd_s.values*100,
}).to_csv(os.path.join(OUT_DIR, 'daily_nav.csv'), index=False)
roll_rows = []
for lab, (a, w) in roll.items():
    roll_rows.append({'Window': lab,
        'Avg Outperformance (%)': round(a,2) if not np.isnan(a) else None,
        'Worst Underperformance (%)': round(w,2) if not np.isnan(w) else None})
pd.DataFrame(roll_rows).to_csv(os.path.join(OUT_DIR, 'rolling_outperformance.csv'), index=False)
pd.DataFrame({
    'Metric': ['CAGR (%)', 'Annualized Std Dev (%)', 'Maximum Drawdown (%)',
               'Sharpe Ratio', 'Sortino Ratio', 'Information Ratio vs Nifty 500',
               'Up-Capture (%)', 'Down-Capture (%)', 'Annualized Turnover (x)', 'Final NAV'],
    'Strategy': [round(cagr*100,2), round(ann_vol*100,2), round(max_dd*100,2),
                 round(sharpe,3), round(sortino,3), round(ir,3),
                 round(up_cap,1), round(dn_cap,1), round(ann_turn,2), round(combined_equity[-1],0)],
    'Nifty 500': [round(bench_cagr*100,2), round(bench_vol*100,2), round(bench_max_dd*100,2),
                  round(bench_sharpe,3), '', '', '', '', '', round(bench_cum.iloc[-1],0)]
}).to_csv(os.path.join(OUT_DIR, 'metrics_summary.csv'), index=False)

print(f"\n{'='*70}")
print(f"  DONE - Outputs in '{OUT_DIR}/'")
print(f"  Final NAV: Rs.{combined_equity[-1]:,.0f}")
print(f"  CAGR: {cagr*100:.2f}% | Sharpe: {sharpe:.3f} | Max DD: {max_dd*100:.2f}%")
print(f"  Info Ratio: {ir:.3f} | Up-Cap: {up_cap:.1f}% | Dn-Cap: {dn_cap:.1f}%")
print(f"{'='*70}")