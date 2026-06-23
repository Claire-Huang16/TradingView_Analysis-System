"""
================================================================================
【Code Gym】TradingView 策略分析系統
本檔案實作兩個可切換的策略：
  1. Volatility Forecast [EXCAVO]
     來源：https://tw.tradingview.com/script/eYkvXmII-Volatility-Forecast-EXCAVO/
  2. CUSUM Volatility Breakout
     來源：https://tw.tradingview.com/script/s8f1jeTo-CUSUM-Volatility-Breakout/

================================================================================
【策略一：Volatility Forecast [EXCAVO]】核心概念

策略以布林通道（Bollinger Bands）為基礎，計算價格的中軌（移動平均）與
上下軌（中軌 ± N 倍標準差），用以衡量目前的波動範圍。

當價格觸碰到上軌或下軌後又被「拉回」通道內（reclaim），且收盤價落在
通道對應的那一半，便視為一次「波動帶回測」事件：
  - 觸碰上軌後拉回 → 偏空訊號（Bear Reclaim）
  - 觸碰下軌後拉回 → 偏多訊號（Bull Reclaim）

可選擇加上趨勢均線濾網（SMA/EMA/WMA/HMA），只有當收盤價方向與均線
方向一致時才視為有效訊號，過濾與大趨勢相反的假訊號。

【技術指標清單】
- 布林通道：SMA(20) ± 2.0 × 標準差(20)
- ATR(14)：作為波動參考
- 趨勢均線：預設 SMA(100)，可選 EMA / WMA / HMA

【買入條件（BUY）】
1. 前一根 K 棒最低點 ≤ 下軌；本根最低點拉回至下軌之上
2. 本根收盤價 < 中軌（落在通道下半部）
3.（可選）收盤價 > 趨勢均線
4. 與上一次買入訊號間隔 ≥ 布林通道長度（避免訊號過於密集）

【賣出條件（SELL）】
1. 前一根 K 棒最高點 ≥ 上軌；本根最高點拉回至上軌之下
2. 本根收盤價 > 中軌（落在通道上半部）
3.（可選）收盤價 < 趨勢均線
4. 與上一次賣出訊號間隔 ≥ 布林通道長度

【適用場景】
- 區間震盪、波動帶清晰的市場
- 中長線波段操作，作為「過熱拉回」的提示工具

【風險提示】
- 本策略屬於通道回測邏輯，趨勢延續時可能持續發出反向假訊號
- 原始 Pine Script 為「指標」非「策略」，本轉換僅將回測標記轉為買賣信號，
  不包含停損停利機制

================================================================================
【策略二：CUSUM Volatility Breakout】核心概念

CUSUM（累積和控制圖, Cumulative Sum Control Chart）源自統計流程管制（SPC），
用於偵測一個時間序列「均值」是否發生持續性偏移。本策略將其應用於價格的
差分序列（去除趨勢後的價格變化），藉由累積正/負偏離量，偵測價格趨勢的
真正轉折，而非單根 K 棒的雜訊波動。

核心流程：
1. 將收盤價做一階差分（去除趨勢，使序列平穩）
2. 以「移動範圍法」估計差分序列的標準差 σ
3. 設定決策門檻 h（以 σ 為單位，並依市場波動率自適應調整：
   波動放大時降低門檻以加快反應，波動縮小時提高門檻以減少假警報）
4. 計算正向 CUSUM（Cu）與負向 CUSUM（Cl），任一方超過門檻（±h·σ）
   即視為偵測到一次顯著的趨勢偏移

【技術指標清單】
- 一階差分價格序列（Length = 30）
- 移動範圍標準差估計（MR / 1.128）
- ATR(10) / ATR(40) 波動比率，用於動態調整決策門檻 h
- CUSUM 正向(Cu)、負向(Cl) 累積量

【買入條件（BUY）】
1. 正向 CUSUM（Cu）累積超過上控制限 UCL = h_sigma × σ
2. 若同時觸發買賣兩個方向（衝突），預設策略會將兩者都忽略
3. 訊號狀態由「空/平」翻轉為「多」時才記錄一次 BUY

【賣出條件（SELL）】
1. 負向 CUSUM（Cl）累積低於下控制限 LCL = −h_sigma × σ
2. 衝突處理同上
3. 訊號狀態由「多/平」翻轉為「空」時才記錄一次 SELL

【適用場景】
- 偵測趨勢的「真正轉折點」，適合中長線波段，避免對單日雜訊反應過度
- 波動性穩定、有明確趨勢輪動的市場

【風險提示】
- CUSUM 對參數（Length、k、h）相當敏感，不同市場需重新校準
- 本轉換為簡化版本：未實作成交量確認、ATR 確認、布林通道輔助訊號、
  與多種重置/衝突處理模式，僅保留核心 CUSUM 偵測邏輯（衝突採「兩者皆忽略」）
- 過去績效不代表未來表現，本系統僅供學術研究，不構成投資建議
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Optional, Tuple

# ================================================================================
# SECTION 1: 系統設定與常數
# ================================================================================

STRATEGIES = {
    "Volatility Forecast [EXCAVO]": {
        "key": "VOLFORECAST",
        "short_name": "EXCAVO-VF",
        "source_url": "https://tw.tradingview.com/script/eYkvXmII-Volatility-Forecast-EXCAVO/",
        "author": "EXCAVO",
        "version": "@version=6",
    },
    "CUSUM Volatility Breakout": {
        "key": "CUSUM",
        "short_name": "CUSUM-VB",
        "source_url": "https://tw.tradingview.com/script/s8f1jeTo-CUSUM-Volatility-Breakout/",
        "author": "CoinOperator",
        "version": "@version=6 (v1.0)",
    },
}

# FinMind 市場對照
FINMIND_TW_MARKET = "TaiwanStockPrice"


# ================================================================================
# SECTION 2: 資料來源 — FMP + FinMind
# ================================================================================

class FMPClient:
    """Financial Modeling Prep API 客戶端"""
    BASE_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"

    def __init__(self, api_key: str):
        self.api_key = api_key.strip()

    def validate_api_key(self) -> bool:
        if not self.api_key:
            return False
        try:
            params = {"symbol": "AAPL", "apikey": self.api_key, "from": "2024-01-01", "to": "2024-01-05"}
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            data = resp.json()
            if isinstance(data, dict) and "Error Message" in data:
                return False
            return True
        except Exception:
            return False

    def get_historical_data(self, symbol: str, from_date: str, to_date: str) -> Optional[pd.DataFrame]:
        params = {
            "symbol": symbol.upper(),
            "apikey": self.api_key,
            "from": from_date,
            "to": to_date
        }
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=30)
            data = resp.json()
            if not data or isinstance(data, dict):
                return None
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df[['open', 'high', 'low', 'close', 'volume']].dropna(subset=['close'])
            return df
        except Exception as e:
            raise RuntimeError(f"FMP API 錯誤: {e}")


class FinMindClient:
    """FinMind API 客戶端 — 台股資料"""
    BASE_URL = "https://api.finmindtrade.com/api/v4/data"

    def __init__(self, token: str = ""):
        self.token = token.strip()

    def get_historical_data(self, stock_id: str, from_date: str, to_date: str) -> Optional[pd.DataFrame]:
        params = {
            "dataset": FINMIND_TW_MARKET,
            "data_id": stock_id,
            "start_date": from_date,
            "end_date": to_date,
        }
        if self.token:
            params["token"] = self.token

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=30)
            result = resp.json()

            if result.get("status") != 200:
                msg = result.get("msg", "未知錯誤")
                raise RuntimeError(f"FinMind 錯誤: {msg}")

            records = result.get("data", [])
            if not records:
                return None

            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()

            rename_map = {
                'open': 'open',
                'max': 'high',
                'min': 'low',
                'close': 'close',
                'Trading_Volume': 'volume'
            }
            df = df.rename(columns=rename_map)

            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    df[col] = np.nan

            df = df[['open', 'high', 'low', 'close', 'volume']].dropna(subset=['close'])
            return df
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"FinMind 連線失敗: {e}")

    def get_stock_info(self, stock_id: str) -> dict:
        """嘗試取得股票基本資訊（股票名稱）"""
        try:
            params = {"dataset": "TaiwanStockInfo", "data_id": stock_id}
            if self.token:
                params["token"] = self.token
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            result = resp.json()
            if result.get("status") == 200 and result.get("data"):
                rec = result["data"][0]
                return {"name": rec.get("stock_name", stock_id), "id": stock_id}
        except Exception:
            pass
        return {"name": stock_id, "id": stock_id}


# ================================================================================
# SECTION 3: 技術指標計算（純 pandas/numpy，不依賴 ta-lib）
# ================================================================================

class Indicators:

    @staticmethod
    def sma(series: pd.Series, length: int) -> pd.Series:
        return series.rolling(window=length).mean()

    @staticmethod
    def ema(series: pd.Series, length: int) -> pd.Series:
        return series.ewm(span=length, adjust=False).mean()

    @staticmethod
    def wma(series: pd.Series, length: int) -> pd.Series:
        weights = np.arange(1, length + 1)
        return series.rolling(window=length).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

    @staticmethod
    def hma(series: pd.Series, length: int) -> pd.Series:
        half_len = max(int(length / 2), 1)
        sqrt_len = max(int(round(np.sqrt(length))), 1)
        wma_half = Indicators.wma(series, half_len)
        wma_full = Indicators.wma(series, length)
        diff = 2 * wma_half - wma_full
        return Indicators.wma(diff, sqrt_len)

    @staticmethod
    def pick_ma(series: pd.Series, length: int, kind: str) -> pd.Series:
        if kind == "EMA":
            return Indicators.ema(series, length)
        if kind == "WMA":
            return Indicators.wma(series, length)
        if kind == "HMA":
            return Indicators.hma(series, length)
        return Indicators.sma(series, length)

    @staticmethod
    def stdev(series: pd.Series, length: int) -> pd.Series:
        """population standard deviation，對應 Pine ta.stdev"""
        return series.rolling(window=length).std(ddof=0)

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range（以 RMA 近似 Pine ta.atr）"""
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / period, adjust=False).mean()


# ================================================================================
# SECTION 4-A: 策略一 — Volatility Forecast [EXCAVO]（波動帶回測訊號）
# ================================================================================

class VolatilityForecastStrategy:
    """
    Volatility Forecast [EXCAVO] — Python 實現
    僅轉換「布林通道 + 帶回測標記（Reclaim Marker）」邏輯為買賣信號；
    原腳本的前向投影（Forecast Envelope）僅供圖表視覺化，不影響訊號判定，
    故本轉換不實作投影部分。
    """

    def __init__(self, df: pd.DataFrame,
                 bb_length: int = 20,
                 bb_mult: float = 2.0,
                 filter_by_ma: bool = True,
                 ma_type: str = "SMA",
                 ma_len: int = 100):
        self.df = df.copy()
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.filter_by_ma = filter_by_ma
        self.ma_type = ma_type
        self.ma_len = ma_len

    def _compute_indicators(self):
        df = self.df
        ind = Indicators

        df['bb_basis'] = ind.sma(df['close'], self.bb_length)
        df['bb_dev']   = ind.stdev(df['close'], self.bb_length)
        df['bb_width'] = self.bb_mult * df['bb_dev']
        df['bb_upper'] = df['bb_basis'] + df['bb_width']
        df['bb_lower'] = df['bb_basis'] - df['bb_width']
        df['atr14']    = ind.atr(df['high'], df['low'], df['close'], 14)
        df['trend_ma'] = ind.pick_ma(df['close'], self.ma_len, self.ma_type)

    def _generate_signals(self) -> pd.DataFrame:
        df = self.df
        n = len(df)

        high     = df['high'].values
        low      = df['low'].values
        close    = df['close'].values
        bb_upper = df['bb_upper'].values
        bb_lower = df['bb_lower'].values
        bb_basis = df['bb_basis'].values
        trend_ma = df['trend_ma'].values
        atr14    = df['atr14'].values

        rows = []
        last_bull_bar = -10 ** 9
        last_bear_bar = -10 ** 9

        for i in range(1, n):
            if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
                    np.isnan(bb_upper[i - 1]) or np.isnan(bb_lower[i - 1])):
                continue

            bear_touch = (high[i - 1] >= bb_upper[i - 1]) and (high[i] < bb_upper[i]) and (close[i] > bb_basis[i])
            bull_touch = (low[i - 1]  <= bb_lower[i - 1]) and (low[i]  > bb_lower[i]) and (close[i] < bb_basis[i])

            if self.filter_by_ma and not np.isnan(trend_ma[i]):
                bear_touch = bear_touch and (close[i] < trend_ma[i])
                bull_touch = bull_touch and (close[i] > trend_ma[i])

            bull_ok = bool(bull_touch) and (i - last_bull_bar) >= self.bb_length
            bear_ok = bool(bear_touch) and (i - last_bear_bar) >= self.bb_length

            if bull_ok:
                last_bull_bar = i
            if bear_ok:
                last_bear_bar = i

            if bull_ok or bear_ok:
                sig_type = "BUY" if bull_ok else "SELL"
                rows.append({
                    '日期':     df.index[i],
                    '信號類型': sig_type,
                    '價格':     round(float(close[i]), 2),
                    'BB上軌':   round(float(bb_upper[i]), 2),
                    'BB中軌':   round(float(bb_basis[i]), 2),
                    'BB下軌':   round(float(bb_lower[i]), 2),
                    '趨勢MA':   round(float(trend_ma[i]), 2) if not np.isnan(trend_ma[i]) else np.nan,
                    'ATR':      round(float(atr14[i]), 2) if not np.isnan(atr14[i]) else np.nan,
                })

        return pd.DataFrame(rows)

    def run(self) -> pd.DataFrame:
        self._compute_indicators()
        return self._generate_signals()


# ================================================================================
# SECTION 4-B: 策略二 — CUSUM Volatility Breakout（核心 CUSUM 偵測邏輯）
# ================================================================================

class CUSUMVolatilityBreakoutStrategy:
    """
    CUSUM Volatility Breakout — Python 實現（簡化核心版）

    僅實作：差分價格序列 → 移動範圍 σ 估計 → 波動率自適應決策門檻 h
    → 正/負向 CUSUM 累積 → 突破偵測。
    未實作：成交量加權確認、ATR 突破確認、布林通道輔助訊號、
    多種重置/衝突處理模式（衝突一律採用原腳本預設「Ignore Both」處理）。
    """

    def __init__(self, df: pd.DataFrame,
                 price_len: int = 30,
                 order: int = 1,
                 lag: int = 1,
                 k_const: float = 0.5,
                 h_min: float = 2.0,
                 h_max: float = 8.0,
                 h_adapt_pct: float = 3.0,
                 smooth_vol_ratio: bool = True,
                 arl_target: float = 4.0):
        self.df = df.copy()
        self.price_len = price_len
        self.order = order
        self.lag = lag
        self.k_const = k_const
        self.h_min = h_min
        self.h_max = h_max
        self.h_adapt_pct = h_adapt_pct / 100.0
        self.smooth_vol_ratio = smooth_vol_ratio
        self.arl_target = arl_target

    def _diff_series(self) -> pd.Series:
        c = self.df['close']
        lag = self.lag
        if self.order == 0:
            return c.copy()
        elif self.order == 1:
            return c - c.shift(lag)
        elif self.order == 2:
            return c - 2 * c.shift(lag) + c.shift(2 * lag)
        else:
            return c - 3 * c.shift(lag) + 3 * c.shift(2 * lag) - c.shift(3 * lag)

    def _compute(self):
        df = self.df
        ind = Indicators

        df['atr10'] = ind.atr(df['high'], df['low'], df['close'], 10)
        df['atr40'] = ind.atr(df['high'], df['low'], df['close'], 40)
        df['price_diff'] = self._diff_series()

        mr = df['price_diff'].diff().abs().rolling(self.price_len).mean()
        df['sd_price'] = mr / 1.128

        vol_ratio_raw = df['atr10'] / df['atr40'].replace(0, np.nan)
        df['vol_ratio'] = vol_ratio_raw.rolling(5).mean() if self.smooth_vol_ratio else vol_ratio_raw

        n = len(df)
        price_diff = df['price_diff'].values
        sd_price   = df['sd_price'].values
        vol_ratio  = df['vol_ratio'].values

        h_sigma_arr = np.full(n, np.nan)
        cu_arr  = np.full(n, 0.0)
        cl_arr  = np.full(n, 0.0)
        ucl_arr = np.full(n, np.nan)
        lcl_arr = np.full(n, np.nan)
        long_sig  = np.zeros(n, dtype=bool)
        short_sig = np.zeros(n, dtype=bool)

        h_sigma = (self.h_min + self.h_max) / 2.0
        cu = 0.0
        cl = 0.0

        for i in range(n):
            vr = vol_ratio[i]
            if not np.isnan(vr):
                if vr > 1.10:
                    h_sigma = max(self.h_min, h_sigma * (1 - self.h_adapt_pct))
                elif vr < 0.90:
                    h_sigma = min(self.h_max, h_sigma * (1 + self.h_adapt_pct))

            # ARL 估計與門檻微調
            arl0 = np.exp(h_sigma * self.k_const) / (self.k_const ** 2)
            delta = 2.0 * self.k_const
            denom = (delta / self.k_const - 1.0)
            if denom != 0:
                arl1 = (h_sigma / self.k_const) / denom
                if arl1 != 0 and not np.isnan(arl1):
                    arl_ratio = arl0 / arl1
                    if arl_ratio < self.arl_target:
                        h_sigma = min(self.h_max, h_sigma * 1.02)

            h_sigma_arr[i] = h_sigma

            is_ready = i >= self.price_len
            sdp = sd_price[i]
            pdv = price_diff[i] if not np.isnan(price_diff[i]) else 0.0

            if is_ready and not np.isnan(sdp) and sdp > 0:
                k_price = self.k_const * sdp
                ucl = h_sigma * sdp
                lcl = -h_sigma * sdp
                cu = max(0.0, cu + (pdv - k_price))
                cl = min(0.0, cl + (pdv + k_price))
                ucl_arr[i] = ucl
                lcl_arr[i] = lcl
            # 尚未 ready 時，Cu/Cl 維持 0（與原腳本初始狀態一致）

            cu_arr[i] = cu
            cl_arr[i] = cl

            if is_ready and not np.isnan(ucl_arr[i]):
                ls = cu > ucl_arr[i]
                ss = cl < lcl_arr[i]
                if ls and ss:          # 衝突：兩者皆忽略（Ignore Both，原腳本預設）
                    ls = False
                    ss = False
                long_sig[i] = ls
                short_sig[i] = ss

        df['h_sigma']   = h_sigma_arr
        df['cu_price']  = cu_arr
        df['cl_price']  = cl_arr
        df['ucl_price'] = ucl_arr
        df['lcl_price'] = lcl_arr
        df['long_signal']  = long_sig
        df['short_signal'] = short_sig

    def _generate_signals(self) -> pd.DataFrame:
        df = self.df
        n = len(df)

        long_sig  = df['long_signal'].values
        short_sig = df['short_signal'].values
        close     = df['close'].values
        cu        = df['cu_price'].values
        cl        = df['cl_price'].values
        ucl       = df['ucl_price'].values
        lcl       = df['lcl_price'].values
        h_sigma   = df['h_sigma'].values

        rows = []
        in_long = False
        in_short = False

        for i in range(n):
            entry_long  = bool(long_sig[i])
            entry_short = bool(short_sig[i])
            exit_long   = entry_short
            exit_short  = entry_long

            prev_in_long, prev_in_short = in_long, in_short

            if entry_long:
                in_long = True
            elif exit_long:
                in_long = False

            if entry_short:
                in_short = True
            elif exit_short:
                in_short = False

            if entry_long and not prev_in_long:
                rows.append({
                    '日期':      df.index[i],
                    '信號類型':  'BUY',
                    '價格':      round(float(close[i]), 2),
                    'Cu(CUSUM+)': round(float(cu[i]), 4),
                    'UCL':        round(float(ucl[i]), 4) if not np.isnan(ucl[i]) else np.nan,
                    'Cl(CUSUM-)': np.nan,
                    'LCL':        np.nan,
                    'H(sigma)':   round(float(h_sigma[i]), 2),
                })
            elif entry_short and not prev_in_short:
                rows.append({
                    '日期':      df.index[i],
                    '信號類型':  'SELL',
                    '價格':      round(float(close[i]), 2),
                    'Cu(CUSUM+)': np.nan,
                    'UCL':        np.nan,
                    'Cl(CUSUM-)': round(float(cl[i]), 4),
                    'LCL':        round(float(lcl[i]), 4) if not np.isnan(lcl[i]) else np.nan,
                    'H(sigma)':   round(float(h_sigma[i]), 2),
                })

        return pd.DataFrame(rows)

    def run(self) -> pd.DataFrame:
        self._compute()
        return self._generate_signals()


# ================================================================================
# SECTION 5: 輔助函數
# ================================================================================

def get_date_range(period_str: str) -> Tuple[str, str]:
    today = datetime.today()
    period_map = {
        "最近 1 年": 365,
        "最近 2 年": 730,
        "最近 3 年": 1095,
        "最近 5 年": 1825,
    }
    days = period_map.get(period_str, 365)
    from_dt = today - timedelta(days=days)
    return from_dt.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")


def is_tw_stock(symbol: str) -> bool:
    """判斷是否為台股代碼（純數字或數字+英文字母）"""
    s = symbol.strip().upper()
    if s.isdigit() and len(s) >= 4:
        return True
    if len(s) == 5 and s[:4].isdigit():
        return True
    return False


# ================================================================================
# SECTION 6: Streamlit 主程式
# ================================================================================

def main():
    st.set_page_config(
        page_title="TradingView 策略分析系統",
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 TradingView 策略分析系統")
    st.divider()

    # ── 側邊欄 ──
    with st.sidebar:
        st.markdown("# 🎯 策略分析器")
        st.divider()

        st.markdown("### 🧩 策略選擇")
        strategy_name = st.selectbox("選擇策略", list(STRATEGIES.keys()), index=0)
        strategy_info = STRATEGIES[strategy_name]

        st.markdown("---")
        st.markdown("### 📡 資料來源設定")

        data_source = st.radio(
            "選擇資料來源",
            ["🇹🇼 台股 (FinMind)", "🌐 美股 (FMP API)"],
            index=0
        )
        use_tw = "台股" in data_source

        if use_tw:
            finmind_token = st.text_input(
                "FinMind Token（可選，不填亦可取得基本資料）",
                type="password",
                help="免費使用者不需填入，但有流量限制。申請 Token：https://finmindtrade.com/"
            )
            fmp_api_key = ""
            symbol_placeholder = "2330"
            symbol_help = "輸入台股代號，如 2330（台積電）、0050（元大台灣50）"
        else:
            fmp_api_key = st.text_input(
                "FMP API Key *",
                type="password",
                help="申請免費 API Key：https://financialmodelingprep.com/"
            )
            finmind_token = ""
            symbol_placeholder = "AAPL"
            symbol_help = "輸入美股代號，如 AAPL、TSLA、MSFT"

        symbol = st.text_input(
            "股票代碼 *",
            value=symbol_placeholder,
            help=symbol_help
        ).strip().upper()

        period_opts = ["最近 1 年", "最近 2 年", "最近 3 年", "最近 5 年", "自訂區間"]
        period = st.selectbox("分析期間", period_opts, index=1)

        if period == "自訂區間":
            col1, col2 = st.columns(2)
            with col1:
                custom_from = st.date_input("起始日期", value=datetime(2022, 1, 1))
            with col2:
                custom_to = st.date_input("結束日期", value=datetime.today())
            from_date_str = custom_from.strftime("%Y-%m-%d")
            to_date_str = custom_to.strftime("%Y-%m-%d")
        else:
            from_date_str, to_date_str = get_date_range(period)

        st.markdown("---")
        st.markdown("### ⚙️ 策略參數")

        if strategy_info["key"] == "VOLFORECAST":
            bb_length = st.slider("布林通道長度 (Length)", 5, 100, 20)
            bb_mult = st.slider("標準差倍數 (Multiplier)", 0.5, 5.0, 2.0, 0.1)
            filter_by_ma = st.checkbox("使用趨勢均線濾網", value=True)
            ma_type = st.selectbox("趨勢均線類型", ["SMA", "EMA", "WMA", "HMA"], index=0,
                                    disabled=not filter_by_ma)
            ma_len = st.slider("趨勢均線長度", 5, 500, 100, disabled=not filter_by_ma)
        else:  # CUSUM
            price_len = st.slider("差分序列長度 (Length)", 10, 100, 30, 5)
            order = st.selectbox("差分階數 (Order)", [0, 1, 2, 3], index=1)
            lag = st.slider("差分間隔 (Lag)", 1, 10, 1)
            k_const = st.slider("k（敏感度，σ 單位）", 0.1, 1.9, 0.5, 0.05)
            h_min = st.slider("h（最小決策門檻，σ 單位）", 1.0, 7.5, 2.0, 0.5)
            h_max = st.slider("h（最大決策門檻，σ 單位）", h_min + 0.5, 8.0, 8.0, 0.5)
            h_adapt_pct = st.slider("h 自適應調整幅度 (%)", 0.5, 100.0, 3.0, 0.5)
            smooth_vol_ratio = st.checkbox("平滑波動率比率", value=True)
            arl_target = st.slider("最低目標 ARL 比率", 1.0, 10.0, 4.0, 0.5)

        st.markdown("---")
        run_btn = st.button("🚀 開始分析", use_container_width=True, type="primary")

        st.markdown("---")
        st.markdown("""
### ⚠️ 免責聲明
本系統僅供學術研究用途，所提供的數據與分析結果**僅供參考，不構成投資建議**。

請使用者自行判斷決策，並承擔相關風險。本系統作者不對任何投資行為負責，
亦不承擔任何損失責任。

**風險提示**:
- 過去的績效不代表未來的表現
- 技術分析有其局限性
- 請謹慎評估自身風險承受能力
        """)

    # ── 初始化 session_state ──
    if 'results' not in st.session_state:
        st.session_state.results = None
        st.session_state.raw_data = None
        st.session_state.symbol = ""
        st.session_state.date_range = ""
        st.session_state.stock_name = ""
        st.session_state.strategy_name = ""

    # ── 執行分析 ──
    if run_btn:
        if not symbol:
            st.error("❌ 請輸入股票代碼")
            st.stop()

        if not use_tw and not fmp_api_key:
            st.error("❌ 請輸入 FMP API Key")
            st.stop()

        with st.spinner("正在分析中..."):
            try:
                status = st.empty()

                if use_tw:
                    status.info("🔍 正在連線 FinMind 台股資料庫...")
                    client = FinMindClient(token=finmind_token if finmind_token else "")
                    status.info(f"📊 正在獲取 {symbol} 的歷史數據...")
                    df = client.get_historical_data(symbol, from_date_str, to_date_str)
                    info = client.get_stock_info(symbol)
                    stock_name = info.get("name", symbol)
                else:
                    status.info("🔍 正在驗證 FMP API Key...")
                    client = FMPClient(fmp_api_key)
                    if not client.validate_api_key():
                        st.error("❌ FMP API Key 無效，請檢查後重試")
                        st.stop()
                    status.info(f"📊 正在獲取 {symbol} 的歷史數據...")
                    df = client.get_historical_data(symbol, from_date_str, to_date_str)
                    stock_name = symbol

                if df is None or len(df) == 0:
                    st.error(f"❌ 無法獲取 {symbol} 的數據，請確認代碼是否正確或調整日期範圍")
                    st.stop()

                if len(df) < 50:
                    st.warning(f"⚠️ 數據量不足（{len(df)} 筆），分析準確性可能受影響")

                status.info("⚙️ 正在計算技術指標...")

                if strategy_info["key"] == "VOLFORECAST":
                    strategy = VolatilityForecastStrategy(
                        df,
                        bb_length=bb_length,
                        bb_mult=bb_mult,
                        filter_by_ma=filter_by_ma,
                        ma_type=ma_type,
                        ma_len=ma_len,
                    )
                else:
                    strategy = CUSUMVolatilityBreakoutStrategy(
                        df,
                        price_len=price_len,
                        order=order,
                        lag=lag,
                        k_const=k_const,
                        h_min=h_min,
                        h_max=h_max,
                        h_adapt_pct=h_adapt_pct,
                        smooth_vol_ratio=smooth_vol_ratio,
                        arl_target=arl_target,
                    )

                status.info("🎯 正在產生交易信號...")
                signal_table = strategy.run()

                st.session_state.results = signal_table
                st.session_state.raw_data = strategy.df
                st.session_state.symbol = symbol
                st.session_state.date_range = f"{from_date_str} ~ {to_date_str}"
                st.session_state.stock_name = stock_name
                st.session_state.strategy_name = strategy_name

                status.empty()
                st.success(f"✅ 分析完成！共找到 {len(signal_table)} 個交易信號")

            except Exception as e:
                st.error(f"❌ 分析過程發生錯誤: {str(e)}")
                st.exception(e)

    # ===== 結果展示區域 =====
    if st.session_state.results is not None:
        sym = st.session_state.symbol
        name = st.session_state.stock_name
        raw = st.session_state.raw_data
        sigs = st.session_state.results
        cur_strategy_name = st.session_state.strategy_name
        cur_info = STRATEGIES.get(cur_strategy_name, list(STRATEGIES.values())[0])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            label = f"{sym}" if name == sym else f"{sym} {name}"
            st.metric("股票代碼", label)
        with col2:
            st.metric("分析期間", st.session_state.date_range)
        with col3:
            st.metric("數據筆數", f"{len(raw):,} 筆")
        with col4:
            st.metric("策略", cur_info["short_name"])

        st.divider()

        tab1, tab2, tab3 = st.tabs(["📊 策略說明", "📋 交易信號", "📈 原始數據"])

        # ────── Tab 1: 策略說明 ──────
        with tab1:
            st.markdown(f"## {cur_strategy_name}")
            st.markdown(f"**來源**: [{cur_info['source_url']}]({cur_info['source_url']})")
            st.markdown(f"**作者**: {cur_info['author']}　｜　**版本**: {cur_info['version']}")

            if cur_info["key"] == "VOLFORECAST":
                st.markdown("### 📖 核心概念")
                st.markdown("""
策略以布林通道（Bollinger Bands）為基礎，計算價格的中軌（移動平均）與
上下軌（中軌 ± N 倍標準差），用以衡量目前的波動範圍。

當價格觸碰到上軌或下軌後又被「拉回」通道內，且收盤價落在通道對應的
那一半，便視為一次「波動帶回測」事件，並可選擇加上趨勢均線濾網，
過濾與大趨勢相反的假訊號。
                """)
                st.markdown("### 🔧 技術指標")
                st.markdown("""
- 布林通道：SMA(20) ± 2.0 × 標準差(20)
- ATR(14)：作為波動參考
- 趨勢均線：可選 SMA / EMA / WMA / HMA
                """)
                col_l, col_r = st.columns(2)
                with col_l:
                    st.markdown("### 📈 買入條件")
                    st.info("""
1. 前一根 K 棒最低點 ≤ 下軌；本根最低點拉回至下軌之上
2. 本根收盤價 < 中軌
3.（可選）收盤價 > 趨勢均線
4. 與上次買入訊號間隔 ≥ 通道長度
                    """)
                with col_r:
                    st.markdown("### 📉 賣出條件")
                    st.warning("""
1. 前一根 K 棒最高點 ≥ 上軌；本根最高點拉回至上軌之下
2. 本根收盤價 > 中軌
3.（可選）收盤價 < 趨勢均線
4. 與上次賣出訊號間隔 ≥ 通道長度
                    """)
                st.markdown("### 🎯 適用場景")
                st.markdown("""
- 區間震盪、波動帶清晰的市場
- 中長線波段操作，作為「過熱拉回」的提示工具
                """)
                st.markdown("### ⚠️ 風險提示")
                st.warning("""
- 趨勢延續時可能持續發出反向假訊號
- 原始腳本為「指標」非「策略」，本轉換僅將回測標記轉為買賣信號，不含停損停利
                """)
            else:
                st.markdown("### 📖 核心概念")
                st.markdown("""
CUSUM（累積和控制圖）源自統計流程管制，用於偵測時間序列均值是否發生
持續性偏移。本策略將其應用於價格的一階差分序列，藉由累積正/負偏離量
偵測價格趨勢的真正轉折，而非單根 K 棒的雜訊波動。

決策門檻 h 會依市場波動率自適應調整：波動放大時降低門檻以加快反應，
波動縮小時提高門檻以減少假警報。
                """)
                st.markdown("### 🔧 技術指標")
                st.markdown("""
- 一階差分價格序列（Length = 30）
- 移動範圍標準差估計（MR / 1.128）
- ATR(10) / ATR(40) 波動比率，用於動態調整 h
- CUSUM 正向(Cu)、負向(Cl) 累積量
                """)
                col_l, col_r = st.columns(2)
                with col_l:
                    st.markdown("### 📈 買入條件")
                    st.info("""
1. 正向 CUSUM（Cu）超過上控制限 UCL = h_sigma × σ
2. 衝突時兩者皆忽略
3. 狀態由空/平翻轉為多時記錄一次 BUY
                    """)
                with col_r:
                    st.markdown("### 📉 賣出條件")
                    st.warning("""
1. 負向 CUSUM（Cl）低於下控制限 LCL = −h_sigma × σ
2. 衝突時兩者皆忽略
3. 狀態由多/平翻轉為空時記錄一次 SELL
                    """)
                st.markdown("### 🎯 適用場景")
                st.markdown("""
- 偵測趨勢真正轉折點，適合中長線波段
- 波動性穩定、有明確趨勢輪動的市場
                """)
                st.markdown("### ⚠️ 風險提示")
                st.warning("""
- 對參數（Length、k、h）相當敏感，不同市場需重新校準
- 本轉換為簡化版本：未實作成交量確認、ATR 確認、布林通道輔助訊號等進階選項
                """)

        # ────── Tab 2: 交易信號 ──────
        with tab2:
            st.markdown("### 📊 信號統計")
            if len(sigs) == 0:
                st.warning("⚠️ 在此參數設定下沒有產生任何交易信號，請嘗試調整策略參數")
            else:
                buy_cnt = len(sigs[sigs['信號類型'] == 'BUY'])
                sell_cnt = len(sigs[sigs['信號類型'] == 'SELL'])

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("總信號數", len(sigs))
                with col2:
                    st.metric("🟢 買入信號", buy_cnt)
                with col3:
                    st.metric("🔴 賣出信號", sell_cnt)

                st.divider()

                latest = sigs.iloc[-1]
                sig_type = latest['信號類型']
                date_str = str(latest['日期'])[:10]
                if sig_type == 'BUY':
                    st.success(f"### 🟢 最新信號: **{sig_type}**　｜　日期: {date_str}　｜　價格: {latest['價格']}")
                else:
                    st.error(f"### 🔴 最新信號: **{sig_type}**　｜　日期: {date_str}　｜　價格: {latest['價格']}")

                st.divider()
                st.markdown("### 📋 完整交易信號")

                display_sigs = sigs.copy()
                display_sigs['日期'] = display_sigs['日期'].astype(str).str[:10]

                def highlight_signals(row):
                    if row['信號類型'] == 'BUY':
                        return ['background-color: #d4edda; color: #155724'] * len(row)
                    else:
                        return ['background-color: #f8d7da; color: #721c24'] * len(row)

                styled = display_sigs.style.apply(highlight_signals, axis=1)
                st.dataframe(styled, use_container_width=True, height=420)

                csv = sigs.to_csv(index=False, encoding='utf-8-sig')
                fname = f"{sym}_{cur_info['short_name']}_signals_{datetime.now().strftime('%Y%m%d')}.csv"
                st.download_button("📥 下載信號表格 (CSV)", csv, fname, "text/csv")

        # ────── Tab 3: 原始數據 ──────
        with tab3:
            st.markdown("### 📊 數據摘要")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("數據筆數", f"{len(raw):,} 筆")
            with c2:
                st.metric("最高收盤價", f"{raw['close'].max():.2f}")
            with c3:
                st.metric("最低收盤價", f"{raw['close'].min():.2f}")
            with c4:
                date_rng = f"{raw.index[0].strftime('%Y-%m-%d')} ~ {raw.index[-1].strftime('%Y-%m-%d')}"
                st.metric("日期範圍", date_rng)

            st.divider()
            st.markdown("### 📈 計算完整數據（含技術指標，顯示最近 200 筆）")

            if cur_info["key"] == "VOLFORECAST":
                show_cols = ['open', 'high', 'low', 'close', 'volume',
                             'bb_basis', 'bb_upper', 'bb_lower', 'trend_ma', 'atr14']
            else:
                show_cols = ['open', 'high', 'low', 'close', 'volume',
                             'price_diff', 'sd_price', 'h_sigma',
                             'cu_price', 'cl_price', 'ucl_price', 'lcl_price']

            available_cols = [c for c in show_cols if c in raw.columns]
            display_raw = raw[available_cols].tail(200).copy()
            display_raw.index = display_raw.index.strftime('%Y-%m-%d')

            st.dataframe(display_raw, use_container_width=True, height=420)

            csv_data = raw.to_csv(encoding='utf-8-sig')
            fname_data = f"{sym}_data_{datetime.now().strftime('%Y%m%d')}.csv"
            st.download_button("📥 下載完整數據 (CSV)", csv_data, fname_data, "text/csv")


if __name__ == "__main__":
    main()
