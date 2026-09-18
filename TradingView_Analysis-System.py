"""
================================================================================
【Code Gym】TradingView 策略分析系統
本檔案實作策略：
  Activity and Volume Orderflow Profile [AlgoAlpha]
  來源：https://tw.tradingview.com/script/r9x4Zp25-Activity-and-Volume-Orderflow-Profile-AlgoAlpha/
  作者：AlgoAlpha
  Pine Script 版本：@version=5

================================================================================
策略說明：Activity and Volume Orderflow Profile [AlgoAlpha]

【核心概念】
原始 Pine Script 是一個「成交量分佈輪廓（Volume Profile）」視覺化指標。
每根 K 棒都會重新取「過去 N 根 K 棒」（Profile Lookback）的最高／最低價
區間，並切分成若干水平價格區塊（bin，數量由 Profile Resolution 決定）。
對每個區塊，分別加總落在該價位區間內的『買方成交量』（收盤價 > 開盤價
的當根成交量）與『賣方成交量』（開盤價 > 收盤價的當根成交量），並以箱體
（box）視覺化畫在圖表右側，協助交易者觀察近期成交量集中在哪個價位、
以及該價位是買方還是賣方主導。

**重要說明**：原始腳本本身是「指標」而非「策略」，且**沒有**明確定義
「買入 / 賣出」訊號 —— 它單純呈現量能輪廓供人工判讀。為了符合本系統
「產生可交易信號」的需求，本轉換在保留其核心「量分佈 + 買賣方占比」
邏輯的前提下，額外定義一組訊號規則：找出當前量能最集中的價格區塊
（POC, Point of Control），計算該區塊內買方量占比，當價格落在 POC
區間內、且買方（或賣方）占比由未過門檻翻轉為過門檻時，記錄一次
BUY（或 SELL）訊號。這組訊號規則屬於本轉換的延伸詮釋，並非原始腳本
內建邏輯，使用前請務必理解此簡化假設。

【技術指標清單】
- 量能輪廓（Volume Profile）：Profile Lookback = 70（回看根數）、
  Profile Resolution = 20（價格區塊數）
- 買方成交量：收盤價 > 開盤價的 K 棒成交量
- 賣方成交量：開盤價 > 收盤價的 K 棒成交量
- POC（Point of Control）：買賣量總和最大的價格區塊
- POC 買方占比 = POC 區塊買方量 / (POC 區塊買方量 + 賣方量)

【買入條件（BUY，本轉換延伸定義）】
1. 收盤價落在當前 POC 區塊的價格區間內
2. POC 買方占比 ≥ 主導門檻（預設 60%），且前一根尚未達門檻（狀態剛翻轉為買方主導）
3. 與上一次買入訊號間隔 ≥ 最小訊號間隔（避免訊號過於密集）

【賣出條件（SELL，本轉換延伸定義）】
1. 收盤價落在當前 POC 區塊的價格區間內
2. POC 賣方占比 ≥ 主導門檻（即買方占比 ≤ 1 − 門檻），且前一根尚未達門檻
3. 與上一次賣出訊號間隔 ≥ 最小訊號間隔

【適用場景】
- 想觀察『價量集中區』由何方主導、藉此判斷短中期價格是否可能延續或反轉的場景
- 成交量資訊充足、流動性佳的標的（成交量稀疏時分佈輪廓參考性較低）

【風險提示】
- 本策略的買賣訊號規則為本轉換延伸定義，並非原始 Pine Script 內建邏輯，
  與原圖表上的箱體視覺化不能直接對應
- 「買方量 / 賣方量」僅以「收盤 > 開盤」或「開盤 > 收盤」概略判定，
  未使用逐筆委託單資料，與真實委買委賣力道可能有落差
- 高／低點重疊區塊採「重複計入」方式加總量能（與原腳本邏輯一致），
  區塊間的量能加總並非彼此互斥，解讀時請留意
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

STRATEGY_INFO = {
    "name": "Activity and Volume Orderflow Profile [AlgoAlpha]",
    "key": "AVOP",
    "short_name": "AlgoAlpha-AVOP",
    "source_url": "https://tw.tradingview.com/script/r9x4Zp25-Activity-and-Volume-Orderflow-Profile-AlgoAlpha/",
    "author": "AlgoAlpha",
    "version": "@version=5",
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
# SECTION 3: 技術指標函數庫（純 pandas/numpy，不依賴 ta-lib）
# ================================================================================

class Indicators:
    """通用技術指標函數庫，供未來擴充策略時使用"""

    @staticmethod
    def sma(series: pd.Series, length: int) -> pd.Series:
        return series.rolling(window=length).mean()

    @staticmethod
    def ema(series: pd.Series, length: int) -> pd.Series:
        return series.ewm(span=length, adjust=False).mean()

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
# SECTION 4: 策略實現 — Activity and Volume Orderflow Profile [AlgoAlpha]
# ================================================================================

class ActivityAndVolumeOrderflowProfileStrategy:
    """
    Activity and Volume Orderflow Profile [AlgoAlpha] — Python 實現（訊號簡化版）

    原始腳本為「成交量分佈輪廓」視覺化指標，逐根 K 棒重算過去 plook 根
    K 棒的高低價區間，切成 res 個水平價格區塊，統計各區塊的買方／賣方
    成交量並以箱體呈現，本身不輸出買賣訊號。

    本轉換額外定義：找出量能最集中的價格區塊（POC），計算 POC 區塊的
    買方量占比；當收盤價落在 POC 區間內、且買方（或賣方）占比由未達門檻
    翻轉為達門檻時，記錄一次 BUY（或 SELL）訊號（詳見檔頭策略說明）。
    """

    def __init__(self, df: pd.DataFrame,
                 plook: int = 70,
                 res: int = 20,
                 ptype: str = "Net Order Flow",
                 threshold: float = 60.0,
                 min_gap: int = 10):
        self.df = df.copy()
        self.plook = plook
        self.res = res
        self.ptype = ptype
        self.threshold = threshold
        self.min_gap = min_gap

    def _compute_profile(self):
        """
        逐根 K 棒計算量能輪廓，找出 POC（Point of Control，量能最大的
        價格區塊）與該區塊的買方量占比，並判斷買賣方主導狀態。
        """
        df = self.df
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        close = df['close'].values.astype(float)
        open_ = df['open'].values.astype(float)
        volume = df['volume'].values.astype(float)
        n = len(df)

        # 買方成交量：收盤 > 開盤；賣方成交量：開盤 > 收盤（與原腳本一致）
        buy_vol = np.where(close > open_, volume, 0.0)
        sell_vol = np.where(open_ > close, volume, 0.0)

        res = max(int(self.res), 1)
        plook = max(int(self.plook), 2)
        threshold = self.threshold / 100.0

        poc_ratio = np.full(n, np.nan)
        poc_top = np.full(n, np.nan)
        poc_bottom = np.full(n, np.nan)
        dominant_state = np.zeros(n, dtype=int)  # 1=買方主導, -1=賣方主導, 0=中性

        for i in range(plook - 1, n):
            w_start = i - plook + 1
            w_high = high[w_start:i + 1]
            w_low = low[w_start:i + 1]
            w_buy = buy_vol[w_start:i + 1]
            w_sell = sell_vol[w_start:i + 1]

            maxx = w_high.max()
            minn = w_low.min()
            if not (maxx > minn):
                continue

            step = (maxx - minn) / res
            bottoms = minn + step * np.arange(res)
            tops = bottoms + step

            # 每個價格區塊與每根K棒的高低範圍是否重疊（與原腳本邏輯一致，
            # 高低範圍跨區塊時會被重複計入多個區塊）
            overlap = ~((w_low[None, :] > tops[:, None]) | (w_high[None, :] < bottoms[:, None]))
            bin_buy = (overlap * w_buy[None, :]).sum(axis=1)
            bin_sell = (overlap * w_sell[None, :]).sum(axis=1)
            bin_total = bin_buy + bin_sell

            poc_idx = int(np.argmax(bin_total))
            total_at_poc = bin_total[poc_idx]
            if total_at_poc <= 0:
                continue

            ratio = bin_buy[poc_idx] / total_at_poc
            poc_ratio[i] = ratio
            poc_bottom[i] = bottoms[poc_idx]
            poc_top[i] = tops[poc_idx]

            if ratio >= threshold:
                dominant_state[i] = 1
            elif ratio <= (1 - threshold):
                dominant_state[i] = -1
            else:
                dominant_state[i] = 0

        df['poc_buy_ratio'] = poc_ratio
        df['poc_top'] = poc_top
        df['poc_bottom'] = poc_bottom
        df['dominant_state'] = dominant_state

    def _generate_signals(self) -> pd.DataFrame:
        df = self.df
        n = len(df)

        close = df['close'].values
        ratio = df['poc_buy_ratio'].values
        top = df['poc_top'].values
        bottom = df['poc_bottom'].values
        state = df['dominant_state'].values

        rows = []
        last_buy_bar = -10 ** 9
        last_sell_bar = -10 ** 9
        prev_state = 0

        for i in range(n):
            if np.isnan(ratio[i]):
                prev_state = 0
                continue

            price_in_poc = bottom[i] <= close[i] <= top[i]
            cur_state = int(state[i])

            if (price_in_poc and cur_state == 1 and prev_state != 1
                    and (i - last_buy_bar) >= self.min_gap):
                rows.append({
                    '日期':          df.index[i],
                    '信號類型':      'BUY',
                    '價格':          round(float(close[i]), 2),
                    'POC買方占比(%)': round(float(ratio[i]) * 100, 1),
                    'POC區間上緣':    round(float(top[i]), 2),
                    'POC區間下緣':    round(float(bottom[i]), 2),
                })
                last_buy_bar = i
            elif (price_in_poc and cur_state == -1 and prev_state != -1
                    and (i - last_sell_bar) >= self.min_gap):
                rows.append({
                    '日期':          df.index[i],
                    '信號類型':      'SELL',
                    '價格':          round(float(close[i]), 2),
                    'POC買方占比(%)': round(float(ratio[i]) * 100, 1),
                    'POC區間上緣':    round(float(top[i]), 2),
                    'POC區間下緣':    round(float(bottom[i]), 2),
                })
                last_sell_bar = i

            prev_state = cur_state

        return pd.DataFrame(rows)

    def run(self) -> pd.DataFrame:
        self._compute_profile()
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

        st.markdown(f"### 🧩 策略：{STRATEGY_INFO['name']}")
        st.caption(f"來源：{STRATEGY_INFO['source_url']}")

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

        plook = st.slider("量能輪廓回看根數 (Profile Lookback)", 20, 200, 70, 5)
        res = st.slider("量能輪廓價格區塊數 (Profile Resolution)", 5, 50, 20)
        ptype = st.selectbox("輪廓顯示類型 (Profile Type，僅影響說明文字)",
                              ["Net Order Flow", "Comparison"], index=0)
        threshold = st.slider("買/賣方主導門檻 (%)", 50.0, 95.0, 60.0, 1.0,
                               help="POC 區塊買方（或賣方）占比達此門檻才視為該方主導")
        min_gap = st.slider("最小訊號間隔（根）", 1, 60, 10,
                             help="避免同方向訊號過於密集出現")

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

                status.info("⚙️ 正在計算量能輪廓...")

                strategy = ActivityAndVolumeOrderflowProfileStrategy(
                    df,
                    plook=plook,
                    res=res,
                    ptype=ptype,
                    threshold=threshold,
                    min_gap=min_gap,
                )

                status.info("🎯 正在產生交易信號...")
                signal_table = strategy.run()

                st.session_state.results = signal_table
                st.session_state.raw_data = strategy.df
                st.session_state.symbol = symbol
                st.session_state.date_range = f"{from_date_str} ~ {to_date_str}"
                st.session_state.stock_name = stock_name

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

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            label = f"{sym}" if name == sym else f"{sym} {name}"
            st.metric("股票代碼", label)
        with col2:
            st.metric("分析期間", st.session_state.date_range)
        with col3:
            st.metric("數據筆數", f"{len(raw):,} 筆")
        with col4:
            st.metric("策略", STRATEGY_INFO["short_name"])

        st.divider()

        tab1, tab2, tab3 = st.tabs(["📊 策略說明", "📋 交易信號", "📈 原始數據"])

        # ────── Tab 1: 策略說明 ──────
        with tab1:
            st.markdown(f"## {STRATEGY_INFO['name']}")
            st.markdown(f"**來源**: [{STRATEGY_INFO['source_url']}]({STRATEGY_INFO['source_url']})")
            st.markdown(f"**作者**: {STRATEGY_INFO['author']}　｜　**版本**: {STRATEGY_INFO['version']}")

            st.markdown("### 📖 核心概念")
            st.markdown("""
原始腳本是一個「成交量分佈輪廓」視覺化指標，逐根 K 棒重新計算過去 N 根
K 棒（Profile Lookback）的高低價區間，切成數個水平價格區塊（Profile
Resolution），分別統計各區塊的買方／賣方成交量並以箱體視覺化呈現，
用來觀察近期成交量集中在哪個價位、以及該價位由買方或賣方主導。

原始腳本本身**沒有**定義買賣訊號，僅供人工判讀量能分佈。本轉換額外
定義：找出量能最集中的價格區塊（POC, Point of Control），計算該區塊
買方量占比，當收盤價落在 POC 區間內、且買方（或賣方）占比翻轉為越過
主導門檻時，記錄一次 BUY（或 SELL）訊號，這是本轉換的延伸詮釋。
            """)
            st.markdown("### 🔧 技術指標")
            st.markdown("""
- 量能輪廓：Profile Lookback（回看根數）、Profile Resolution（價格區塊數）
- 買方成交量：收盤 > 開盤的 K 棒成交量；賣方成交量：開盤 > 收盤的 K 棒成交量
- POC（Point of Control）：買賣量總和最大的價格區塊
- POC 買方占比 = POC 區塊買方量 / (買方量 + 賣方量)
            """)
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("### 📈 買入條件（本轉換延伸定義）")
                st.info("""
1. 收盤價落在當前 POC 區塊的價格區間內
2. POC 買方占比 ≥ 主導門檻，且前一根尚未達門檻（剛翻轉為買方主導）
3. 與上次買入訊號間隔 ≥ 最小訊號間隔
                """)
            with col_r:
                st.markdown("### 📉 賣出條件（本轉換延伸定義）")
                st.warning("""
1. 收盤價落在當前 POC 區塊的價格區間內
2. POC 賣方占比 ≥ 主導門檻，且前一根尚未達門檻（剛翻轉為賣方主導）
3. 與上次賣出訊號間隔 ≥ 最小訊號間隔
                """)
            st.markdown("### 🎯 適用場景")
            st.markdown("""
- 想觀察「價量集中區」由何方主導、藉此判斷短中期價格延續或反轉的場景
- 成交量資訊充足、流動性佳的標的
            """)
            st.markdown("### ⚠️ 風險提示")
            st.warning("""
- 買賣訊號規則為本轉換延伸定義，非原始腳本內建邏輯，與圖表箱體視覺化無法直接對應
- 買／賣方量僅以「收盤 vs 開盤」概略判定，未使用逐筆委託單資料
- 高低點重疊區塊採重複計入方式加總量能，區塊間並非互斥，解讀時請留意
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
                fname = f"{sym}_{STRATEGY_INFO['short_name']}_signals_{datetime.now().strftime('%Y%m%d')}.csv"
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

            show_cols = ['open', 'high', 'low', 'close', 'volume',
                         'poc_buy_ratio', 'poc_top', 'poc_bottom', 'dominant_state']
            available_cols = [c for c in show_cols if c in raw.columns]
            display_raw = raw[available_cols].tail(200).copy()
            display_raw.index = display_raw.index.strftime('%Y-%m-%d')

            st.dataframe(display_raw, use_container_width=True, height=420)

            csv_data = raw.to_csv(encoding='utf-8-sig')
            fname_data = f"{sym}_data_{datetime.now().strftime('%Y%m%d')}.csv"
            st.download_button("📥 下載完整數據 (CSV)", csv_data, fname_data, "text/csv")


if __name__ == "__main__":
    main()
