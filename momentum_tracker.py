#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════
  MOMENTUM SWING TRACKER
  Markets : Nifty 50 | Nifty 100 | S&P 500
  Strategy: Momentum / Swing (1–30 days hold)
  Run this script after market close each day.
  Output  : momentum_dashboard.html (open in browser)
══════════════════════════════════════════════════════════════
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json, os, sys, warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
#  INDEX CONSTITUENTS
# ──────────────────────────────────────────────────────────────

NIFTY_50 = [
    'RELIANCE.NS','TCS.NS','HDFCBANK.NS','INFY.NS','HINDUNILVR.NS',
    'ICICIBANK.NS','KOTAKBANK.NS','BHARTIARTL.NS','ITC.NS','AXISBANK.NS',
    'BAJFINANCE.NS','BAJAJFINSV.NS','WIPRO.NS','HCLTECH.NS','ASIANPAINT.NS',
    'ONGC.NS','TATAMOTORS.NS','SUNPHARMA.NS','TECHM.NS','MARUTI.NS',
    'NESTLEIND.NS','POWERGRID.NS','NTPC.NS','ULTRACEMCO.NS','TITAN.NS',
    'ADANIPORTS.NS','BAJAJ-AUTO.NS','JSWSTEEL.NS','SBIN.NS','TATACONSUM.NS',
    'LTIM.NS','LT.NS','DRREDDY.NS','M&M.NS','COALINDIA.NS',
    'GRASIM.NS','BRITANNIA.NS','HINDALCO.NS','DIVISLAB.NS','CIPLA.NS',
    'APOLLOHOSP.NS','ADANIENT.NS','EICHERMOT.NS','BPCL.NS','INDUSINDBK.NS',
    'HEROMOTOCO.NS','UPL.NS','SHRIRAMFIN.NS','TRENT.NS','HDFCLIFE.NS'
]

NIFTY_100_EXTRA = [   # Stocks in Nifty 100 but outside Nifty 50
    'PIDILITIND.NS','MPHASIS.NS','COFORGE.NS','PERSISTENT.NS','ABB.NS',
    'SIEMENS.NS','HAVELLS.NS','ICICIGI.NS','ICICIPRULI.NS','SBILIFE.NS',
    'POLYCAB.NS','CHOLAFIN.NS','NYKAA.NS','PAYTM.NS','ZOMATO.NS',
    'NAUKRI.NS','DMART.NS','IRCTC.NS','TATAPOWER.NS','BANKBARODA.NS',
    'PNB.NS','CANFINHOME.NS','MCDOWELL-N.NS','OBEROIRLTY.NS','GODREJCP.NS',
    'DABUR.NS','MARICO.NS','VEDL.NS','NMDC.NS','SAIL.NS',
    'RECLTD.NS','PFC.NS','IRFC.NS','HFCL.NS','LINDEINDIA.NS',
    'CUMMINSIND.NS','ASHOKLEY.NS','MUTHOOTFIN.NS','BAJAJHLDNG.NS','TORNTPHARM.NS',
    'ALKEM.NS','LAURUSLABS.NS','AUBANK.NS','BANDHANBNK.NS','FEDERALBNK.NS',
    'MANAPPURAM.NS','JKCEMENT.NS','AMBUJACEMENT.NS','ACC.NS','COLPAL.NS'
]

SP500_FALLBACK = [   # Top 100 S&P 500 by liquidity (fallback if Wikipedia is blocked)
    'AAPL','MSFT','NVDA','AMZN','GOOGL','GOOG','META','TSLA','BRK-B','JPM',
    'JNJ','V','UNH','AVGO','XOM','MA','LLY','HD','PG','MRK',
    'CVX','ABBV','KO','PEP','BAC','CRM','COST','TMO','ACN','MCD',
    'ABT','LIN','DHR','ORCL','NFLX','ADBE','AMD','PLTR','PYPL','INTC',
    'IBM','GE','CAT','GS','MS','RTX','HON','AXP','BLK','SPGI',
    'ISRG','SYK','ZTS','PFE','AMGN','GILD','REGN','VRTX','BSX','MDT',
    'ELV','HUM','CI','CVS','WMT','TGT','LOWE','TJX','NKE','SBUX',
    'DIS','CMCSA','CHTR','VZ','T','TMUS','COP','SLB','OXY','PSX',
    'SO','DUK','NEE','AEP','EXC','WFC','USB','PNC','TFC','COF',
    'MMM','EMR','ITW','ETN','PH','FDX','UPS','DE','BA','LMT'
]

# ──────────────────────────────────────────────────────────────
#  TECHNICAL INDICATOR FUNCTIONS
# ──────────────────────────────────────────────────────────────

def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line  = ema_f - ema_s
    sig   = line.ewm(span=signal, adjust=False).mean()
    hist  = line - sig
    return line, sig, hist

def adx(high, low, close, period=14):
    tr  = pd.concat([(high - low),
                     (high - close.shift()).abs(),
                     (low  - close.shift()).abs()], axis=1).max(axis=1)
    pdm = high.diff().clip(lower=0)
    ndm = (-low.diff()).clip(lower=0)
    pdm = pdm.where(pdm > (-low.diff()).clip(lower=0), 0)
    ndm = ndm.where(ndm > (high.diff()).clip(lower=0), 0)
    atr = tr.ewm(span=period, adjust=False).mean()
    pdi = 100 * pdm.ewm(span=period, adjust=False).mean() / atr
    ndi = 100 * ndm.ewm(span=period, adjust=False).mean() / atr
    dx  = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean(), pdi, ndi

def atr(high, low, close, period=14):
    tr = pd.concat([(high - low),
                    (high - close.shift()).abs(),
                    (low  - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calc_target_stop(price, score, atr_val, ema20, signal):
    """
    Target  : ATR-based 5-day projection (up for BUY/HOLD, down for SELL)
    Stop    : max(EMA20, Price - 1.5×ATR) for longs; Price + 1×ATR for SELL
    R:R     : (Target - Price) / (Price - Stop)
    """
    # Target multiplier scales with score (1.5× to 2.5× ATR)
    if signal in ('BUY', 'HOLD'):
        t_mult   = 1.5 + (score / 100)
        target   = round(price + t_mult * atr_val, 2)
        stop_atr = round(price - 1.5 * atr_val, 2)
        stop     = round(max(stop_atr, ema20), 2)
    else:  # SELL — downside target
        t_mult   = 1.5 + ((100 - score) / 100)
        target   = round(price - t_mult * atr_val, 2)
        stop     = round(price + 1.0 * atr_val, 2)   # cut-loss if holding long

    risk = price - stop if signal in ('BUY', 'HOLD') else stop - price
    reward = abs(target - price)
    rr = round(reward / risk, 2) if risk > 0 else 0.0
    return target, stop, rr

# ──────────────────────────────────────────────────────────────
#  SCORING & SIGNAL LOGIC
# ──────────────────────────────────────────────────────────────

def momentum_score(r):
    """
    Composite score 0-100 for swing momentum.
    Weighted across RSI | MACD | ADX | EMA trend | Volume.
    """
    s = 0

    # RSI (30 pts) — sweet spot 50-68 for swing entries
    rsi_v = r['RSI']
    if   50 <= rsi_v <= 68: s += 30
    elif 45 <= rsi_v <  50: s += 18
    elif 68 <  rsi_v <= 72: s += 12
    elif rsi_v > 78 or rsi_v < 30: s -= 15

    # MACD histogram (25 pts)
    if r['MACD_Hist'] > 0:
        s += 25
        if r['MACD_Hist'] > r['MACD_Hist_Prev']: s += 5   # expanding = bullish momentum
    else:
        s -= 5

    # ADX trend strength (20 pts)
    adx_v = r['ADX']
    if   adx_v >= 30: s += 20
    elif adx_v >= 25: s += 15
    elif adx_v >= 20: s += 8
    elif adx_v <  15: s -= 5

    # Price vs EMAs (15 pts)
    if   r['Price'] > r['EMA20'] > r['EMA50']: s += 15
    elif r['Price'] > r['EMA20']:               s += 8
    elif r['Price'] < r['EMA50']:               s -= 8

    # Volume surge (10 pts)
    vr = r['Vol_Ratio']
    if   vr >= 1.5: s += 10
    elif vr >= 1.1: s += 5

    return max(0, min(100, s))


def signal(score, rsi_v, macd_hist, price_vs_ema20, adx_v, pdi, ndi):
    """
    BUY  : Strong momentum building — enter trade
    HOLD : In trend but not optimal entry OR already in trade
    SELL : Momentum fading / reversal / overextended
    """
    if (score >= 65
            and rsi_v < 73
            and macd_hist > 0
            and price_vs_ema20 > 0
            and pdi > ndi):
        return 'BUY'
    elif (score <= 32
          or rsi_v > 76
          or (macd_hist < 0 and price_vs_ema20 < 0)
          or (ndi > pdi and adx_v > 22)):
        return 'SELL'
    else:
        return 'HOLD'

# ──────────────────────────────────────────────────────────────
#  DATA FETCHER
# ──────────────────────────────────────────────────────────────

def fetch_sp500_list():
    try:
        tables  = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        tickers = tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
        print(f"  ✓ Fetched full S&P 500 list ({len(tickers)} tickers)")
        return tickers
    except Exception:
        print(f"  ⚠ Wikipedia blocked — using top-100 S&P 500 fallback")
        return SP500_FALLBACK


def analyze(tickers, index_label, batch=25):
    print(f"\n▶ Analyzing {index_label} ({len(tickers)} stocks)...")
    results = []

    for i in range(0, len(tickers), batch):
        chunk = tickers[i : i + batch]
        try:
            raw = yf.download(chunk, period='90d', interval='1d',
                              progress=False, auto_adjust=True, threads=True)
        except Exception as e:
            print(f"  Batch error: {e}")
            continue

        multi = len(chunk) > 1

        for ticker in chunk:
            try:
                if multi:
                    close  = raw['Close'][ticker].dropna()
                    high   = raw['High'][ticker].dropna()
                    low    = raw['Low'][ticker].dropna()
                    volume = raw['Volume'][ticker].dropna()
                else:
                    close  = raw['Close'].dropna()
                    high   = raw['High'].dropna()
                    low    = raw['Low'].dropna()
                    volume = raw['Volume'].dropna()

                if len(close) < 40:
                    continue

                rsi_s            = rsi(close)
                macd_l, macd_sig, macd_h = macd(close)
                adx_s, pdi_s, ndi_s = adx(high, low, close)
                atr_s  = atr(high, low, close)
                ema20  = close.ewm(span=20, adjust=False).mean()
                ema50  = close.ewm(span=50, adjust=False).mean()
                ema200 = close.ewm(span=200, adjust=False).mean()
                vol20  = volume.rolling(20).mean()

                px    = float(close.iloc[-1])
                px_p  = float(close.iloc[-2])
                chg   = round((px - px_p) / px_p * 100, 2)
                week_chg = round((px - float(close.iloc[-6])) / float(close.iloc[-6]) * 100, 2)

                row = dict(
                    Ticker       = ticker.replace('.NS', ''),
                    FullTicker   = ticker,
                    Index        = index_label,
                    Price        = round(px, 2),
                    Change_Pct   = chg,
                    Week_Chg     = week_chg,
                    RSI          = round(float(rsi_s.iloc[-1]), 1),
                    MACD_Hist    = round(float(macd_h.iloc[-1]), 4),
                    MACD_Hist_Prev= round(float(macd_h.iloc[-2]), 4),
                    MACD_Line    = round(float(macd_l.iloc[-1]), 3),
                    ADX          = round(float(adx_s.iloc[-1]), 1),
                    Plus_DI      = round(float(pdi_s.iloc[-1]), 1),
                    Minus_DI     = round(float(ndi_s.iloc[-1]), 1),
                    EMA20        = round(float(ema20.iloc[-1]), 2),
                    EMA50        = round(float(ema50.iloc[-1]), 2),
                    EMA200       = round(float(ema200.iloc[-1]), 2),
                    Vol_Ratio    = round(float(volume.iloc[-1] / vol20.iloc[-1]), 2)
                                    if float(vol20.iloc[-1]) > 0 else 1.0,
                )

                row['Score']  = momentum_score(row)
                row['Signal'] = signal(
                    row['Score'], row['RSI'], row['MACD_Hist'],
                    px - float(ema20.iloc[-1]),
                    row['ADX'], row['Plus_DI'], row['Minus_DI']
                )
                atr_val = float(atr_s.iloc[-1])
                row['ATR'] = round(atr_val, 2)
                t5d, sl, rr = calc_target_stop(
                    px, row['Score'], atr_val,
                    float(ema20.iloc[-1]), row['Signal']
                )
                row['Target_5D'] = t5d
                row['Stop_Loss'] = sl
                row['RR']        = rr
                results.append(row)

            except Exception:
                pass

        done = min(i + batch, len(tickers))
        print(f"  {done}/{len(tickers)} processed…", end='\r')

    print(f"  ✓ {len(results)} stocks analysed for {index_label}    ")
    return results


# ──────────────────────────────────────────────────────────────
#  HTML DASHBOARD GENERATOR
# ──────────────────────────────────────────────────────────────

def generate_html(all_results, output_file='momentum_dashboard.html'):
    data_json = json.dumps(all_results)
    ts = datetime.now().strftime('%d %b %Y  %H:%M')

    buy_count  = sum(1 for r in all_results if r['Signal'] == 'BUY')
    hold_count = sum(1 for r in all_results if r['Signal'] == 'HOLD')
    sell_count = sum(1 for r in all_results if r['Signal'] == 'SELL')
    total      = len(all_results)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Momentum Swing Tracker</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Sora:wght@300;400;600;700&display=swap" rel="stylesheet"/>
<style>
:root{{
  --bg:#05030F;
  --surface:#0B0720;
  --card:#100D28;
  --border:#1E1545;
  --purple:#59058F;
  --teal:#00A8A8;
  --blue:#0388BC;
  --navy:#180D5B;
  --buy:#00D4AA;
  --hold:#0388BC;
  --sell:#C0205A;
  --text:#E8E4FF;
  --muted:#7B71B0;
  --mono:'Space Mono',monospace;
  --sans:'Sora',sans-serif;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:var(--sans);min-height:100vh;
     background-image:radial-gradient(ellipse at 10% 0%,rgba(89,5,143,.15) 0%,transparent 60%),
                      radial-gradient(ellipse at 90% 100%,rgba(0,168,168,.08) 0%,transparent 50%);}}
header{{display:flex;align-items:center;justify-content:space-between;padding:20px 32px 16px;
        border-bottom:1px solid var(--border);background:rgba(11,7,32,.6);backdrop-filter:blur(12px);
        position:sticky;top:0;z-index:100}}
.logo{{display:flex;align-items:center;gap:12px}}
.logo-icon{{width:36px;height:36px;border-radius:8px;
            background:linear-gradient(135deg,var(--purple),var(--teal));
            display:flex;align-items:center;justify-content:center;font-size:16px}}
.logo-text{{font-weight:700;font-size:18px;letter-spacing:-.3px}}
.logo-sub{{font-size:11px;color:var(--muted);font-family:var(--mono)}}
.ts{{font-family:var(--mono);font-size:11px;color:var(--muted);text-align:right}}
.ts span{{display:block;color:var(--teal);font-size:10px;margin-top:2px}}

.wrap{{max-width:1600px;margin:0 auto;padding:24px 32px}}

/* ── Summary cards ── */
.summary-grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:14px;margin-bottom:28px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px;position:relative;overflow:hidden}}
.card::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px}}
.card.buy-card::before{{background:var(--buy)}}
.card.hold-card::before{{background:var(--hold)}}
.card.sell-card::before{{background:var(--sell)}}
.card.nifty50-card::before{{background:var(--purple)}}
.card.nifty100-card::before{{background:var(--teal)}}
.card.sp500-card::before{{background:var(--blue)}}
.card-label{{font-size:10px;font-family:var(--mono);color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}}
.card-val{{font-size:26px;font-weight:700;font-family:var(--mono)}}
.card-val.buy-val{{color:var(--buy)}}
.card-val.hold-val{{color:var(--hold)}}
.card-val.sell-val{{color:var(--sell)}}
.card-sub{{font-size:11px;color:var(--muted);margin-top:4px}}
.index-price{{font-size:20px;font-weight:600;font-family:var(--mono)}}
.index-chg{{font-size:12px;font-family:var(--mono);margin-top:3px}}
.pos{{color:var(--buy)}}.neg{{color:var(--sell)}}

/* ── Controls ── */
.controls{{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:20px}}
.filter-group{{display:flex;gap:6px}}
.filter-btn{{background:var(--card);border:1px solid var(--border);color:var(--muted);
             font-family:var(--mono);font-size:11px;padding:7px 14px;border-radius:6px;
             cursor:pointer;transition:all .2s;text-transform:uppercase;letter-spacing:.5px}}
.filter-btn:hover{{border-color:var(--blue);color:var(--text)}}
.filter-btn.active{{border-color:transparent;color:#fff;font-weight:700}}
.filter-btn.active.all-btn{{background:var(--navy)}}
.filter-btn.active.buy-btn{{background:linear-gradient(135deg,#006845,#00D4AA44);border-color:var(--buy);color:var(--buy)}}
.filter-btn.active.hold-btn{{background:linear-gradient(135deg,#013d5c,#0388BC44);border-color:var(--blue);color:var(--blue)}}
.filter-btn.active.sell-btn{{background:linear-gradient(135deg,#5c0028,#C0205A44);border-color:var(--sell);color:var(--sell)}}
.filter-btn.active.idx-btn{{background:var(--purple);color:#fff}}
.sep{{width:1px;height:28px;background:var(--border)}}
.search-box{{flex:1;min-width:180px;background:var(--card);border:1px solid var(--border);
             color:var(--text);font-family:var(--mono);font-size:12px;padding:7px 12px;
             border-radius:6px;outline:none}}
.search-box:focus{{border-color:var(--blue)}}
.sort-info{{font-family:var(--mono);font-size:10px;color:var(--muted);margin-left:auto}}

/* ── Table ── */
.table-wrap{{overflow-x:auto;border-radius:12px;border:1px solid var(--border)}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
thead tr{{background:var(--navy)}}
th{{padding:12px 14px;text-align:left;font-family:var(--mono);font-size:10px;
    text-transform:uppercase;letter-spacing:.8px;color:var(--muted);
    cursor:pointer;user-select:none;white-space:nowrap;position:relative}}
th:hover{{color:var(--text)}}
th.sorted-asc::after{{content:' ▲';color:var(--teal);font-size:9px}}
th.sorted-desc::after{{content:' ▼';color:var(--teal);font-size:9px}}
tbody tr{{border-bottom:1px solid rgba(30,21,69,.5);transition:background .15s}}
tbody tr:hover{{background:rgba(3,136,188,.07)}}
tbody tr:nth-child(even){{background:rgba(16,13,40,.4)}}
tbody tr:nth-child(even):hover{{background:rgba(3,136,188,.07)}}
td{{padding:11px 14px;white-space:nowrap}}

.rank{{font-family:var(--mono);font-size:11px;color:var(--muted);text-align:right;width:36px}}
.ticker{{font-family:var(--mono);font-weight:700;font-size:13px;color:var(--text)}}
.idx-badge{{font-size:9px;font-family:var(--mono);padding:2px 7px;border-radius:3px;
            font-weight:700;letter-spacing:.5px}}
.idx-nifty50{{background:rgba(89,5,143,.3);color:#C060FF;border:1px solid rgba(89,5,143,.5)}}
.idx-nifty100{{background:rgba(0,168,168,.2);color:var(--teal);border:1px solid rgba(0,168,168,.4)}}
.idx-sp500{{background:rgba(3,136,188,.2);color:#60C8FF;border:1px solid rgba(3,136,188,.4)}}
.price{{font-family:var(--mono);font-size:12px}}
.chg{{font-family:var(--mono);font-size:11px}}
.sig-badge{{display:inline-block;padding:4px 10px;border-radius:4px;
            font-family:var(--mono);font-size:10px;font-weight:700;letter-spacing:.8px}}
.sig-BUY{{background:rgba(0,212,170,.15);color:var(--buy);border:1px solid rgba(0,212,170,.35)}}
.sig-HOLD{{background:rgba(3,136,188,.15);color:var(--hold);border:1px solid rgba(3,136,188,.35)}}
.sig-SELL{{background:rgba(192,32,90,.15);color:var(--sell);border:1px solid rgba(192,32,90,.35)}}
.rsi-val{{font-family:var(--mono);font-size:11px}}
.rsi-buy{{color:var(--buy)}}.rsi-hold{{color:var(--blue)}}.rsi-sell{{color:var(--sell)}}
.macd-pos{{color:var(--buy);font-family:var(--mono);font-size:11px}}
.macd-neg{{color:var(--sell);font-family:var(--mono);font-size:11px}}
.adx-val{{font-family:var(--mono);font-size:11px}}
.score-bar{{display:flex;align-items:center;gap:8px}}
.bar-bg{{width:80px;height:6px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:3px;transition:width .3s}}
.score-num{{font-family:var(--mono);font-size:11px;color:var(--muted);min-width:26px}}
.ema-ok{{color:var(--buy)}}.ema-bad{{color:var(--sell)}}.ema-neutral{{color:var(--muted)}}

.empty-state{{text-align:center;padding:60px;color:var(--muted);font-family:var(--mono);font-size:13px}}
.count-label{{font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:8px}}

footer{{text-align:center;padding:24px;font-family:var(--mono);font-size:10px;
        color:var(--muted);border-top:1px solid var(--border);margin-top:32px}}
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">📈</div>
    <div>
      <div class="logo-text">MOMENTUM TRACKER</div>
      <div class="logo-sub">NIFTY 50 · NIFTY 100 · S&amp;P 500</div>
    </div>
  </div>
  <div class="ts">Last updated<span id="ts">{ts}</span></div>
</header>

<div class="wrap">

  <!-- Summary Cards -->
  <div class="summary-grid">
    <div class="card buy-card">
      <div class="card-label">▲ BUY</div>
      <div class="card-val buy-val" id="cnt-buy">{buy_count}</div>
      <div class="card-sub">momentum setups</div>
    </div>
    <div class="card hold-card">
      <div class="card-label">◆ HOLD</div>
      <div class="card-val hold-val" id="cnt-hold">{hold_count}</div>
      <div class="card-sub">watch / in-trade</div>
    </div>
    <div class="card sell-card">
      <div class="card-label">▼ SELL</div>
      <div class="card-val sell-val" id="cnt-sell">{sell_count}</div>
      <div class="card-sub">exit / avoid</div>
    </div>
    <div class="card nifty50-card">
      <div class="card-label">NIFTY 50</div>
      <div class="index-price" id="n50-px">—</div>
      <div class="index-chg" id="n50-chg">loading…</div>
    </div>
    <div class="card nifty100-card">
      <div class="card-label">NIFTY 100</div>
      <div class="index-price" id="n100-px">—</div>
      <div class="index-chg" id="n100-chg">loading…</div>
    </div>
    <div class="card sp500-card">
      <div class="card-label">S&amp;P 500</div>
      <div class="index-price" id="sp500-px">—</div>
      <div class="index-chg" id="sp500-chg">loading…</div>
    </div>
  </div>

  <!-- Controls -->
  <div class="controls">
    <div class="filter-group">
      <button class="filter-btn all-btn active" onclick="setSignal('ALL')">All ({total})</button>
      <button class="filter-btn buy-btn" onclick="setSignal('BUY')">▲ Buy ({buy_count})</button>
      <button class="filter-btn hold-btn" onclick="setSignal('HOLD')">◆ Hold ({hold_count})</button>
      <button class="filter-btn sell-btn" onclick="setSignal('SELL')">▼ Sell ({sell_count})</button>
    </div>
    <div class="sep"></div>
    <div class="filter-group">
      <button class="filter-btn idx-btn active" onclick="setIndex('ALL')">All Indices</button>
      <button class="filter-btn idx-btn" onclick="setIndex('NIFTY 50')">Nifty 50</button>
      <button class="filter-btn idx-btn" onclick="setIndex('NIFTY 100')">Nifty 100</button>
      <button class="filter-btn idx-btn" onclick="setIndex('S&P 500')">S&amp;P 500</button>
    </div>
    <input class="search-box" id="searchBox" placeholder="Search ticker…" oninput="render()"/>
    <div class="sort-info" id="sort-info">Click column headers to sort</div>
  </div>

  <!-- Table -->
  <div class="table-wrap">
    <table id="mainTable">
      <thead>
        <tr>
          <th onclick="sortBy('rank')">#</th>
          <th onclick="sortBy('Ticker')">Ticker</th>
          <th onclick="sortBy('Index')">Index</th>
          <th onclick="sortBy('Price')">Price</th>
          <th onclick="sortBy('Change_Pct')">1D %</th>
          <th onclick="sortBy('Week_Chg')">5D %</th>
          <th onclick="sortBy('RSI')">RSI</th>
          <th onclick="sortBy('MACD_Hist')">MACD Hist</th>
          <th onclick="sortBy('ADX')">ADX</th>
          <th onclick="sortBy('Vol_Ratio')">Vol ×</th>
          <th onclick="sortBy('Target_5D')">🎯 Target 5D</th>
          <th onclick="sortBy('Stop_Loss')">🛑 Stop Loss</th>
          <th onclick="sortBy('RR')">R:R</th>
          <th onclick="sortBy('Score')">Score</th>
          <th onclick="sortBy('Signal')">Signal</th>
        </tr>
      </thead>
      <tbody id="tbody"></tbody>
    </table>
    <div class="empty-state" id="empty" style="display:none">No stocks match current filters</div>
  </div>
  <div class="count-label" id="count-label"></div>
</div>

<footer>Momentum Swing Tracker · Data via Yahoo Finance · Not financial advice · For personal trading research only</footer>

<script>
const RAW = {data_json};
let signalFilter = 'ALL', indexFilter = 'ALL';
let sortKey = 'Score', sortDir = -1;

function setSignal(v){{
  signalFilter = v;
  document.querySelectorAll('.filter-group:first-child .filter-btn').forEach(b=>b.classList.remove('active'));
  document.querySelector('.'+ (v==='ALL'?'all':v.toLowerCase()) +'-btn').classList.add('active');
  render();
}}
function setIndex(v){{
  indexFilter = v;
  document.querySelectorAll('.filter-group:last-child .filter-btn').forEach(b=>b.classList.remove('active'));
  event.target.classList.add('active');
  render();
}}
function sortBy(k){{
  if(sortKey===k) sortDir*=-1; else {{sortKey=k; sortDir=-1;}}
  document.querySelectorAll('th').forEach(t=>{{t.classList.remove('sorted-asc','sorted-desc')}});
  const headers = ['rank','Ticker','Index','Price','Change_Pct','Week_Chg','RSI','MACD_Hist','ADX','Vol_Ratio','Target_5D','Stop_Loss','RR','Score','Signal'];
  const idx = headers.indexOf(k);
  if(idx>=0){{
    const th = document.querySelectorAll('th')[idx];
    th.classList.add(sortDir===-1?'sorted-desc':'sorted-asc');
  }}
  render();
}}
function scoreColor(s){{
  if(s>=65) return 'linear-gradient(90deg,#00D4AA,#00A8A8)';
  if(s>=40) return 'linear-gradient(90deg,#0388BC,#0388BC88)';
  return 'linear-gradient(90deg,#C0205A,#C0205A88)';
}}
function rsiClass(r){{
  if(r>=73) return 'rsi-sell';
  if(r>=50) return 'rsi-buy';
  if(r>=35) return 'rsi-hold';
  return 'rsi-sell';
}}
function idxClass(idx){{
  if(idx==='NIFTY 50') return 'idx-nifty50';
  if(idx==='NIFTY 100') return 'idx-nifty100';
  return 'idx-sp500';
}}
function idxShort(idx){{
  if(idx==='NIFTY 50') return 'N50';
  if(idx==='NIFTY 100') return 'N100';
  return 'SP5';
}}
function emaClass(r){{
  if(r.Price > r.EMA20 && r.EMA20 > r.EMA50) return 'ema-ok';
  if(r.Price < r.EMA50) return 'ema-bad';
  return 'ema-neutral';
}}
function emaText(r){{
  if(r.Price > r.EMA20 && r.EMA20 > r.EMA50) return '▲▲';
  if(r.Price > r.EMA20) return '▲—';
  if(r.Price < r.EMA50) return '▼▼';
  return '—▼';
}}

function render(){{
  const search = document.getElementById('searchBox').value.toLowerCase();
  let data = RAW.filter(r=>{{
    if(signalFilter!=='ALL' && r.Signal!==signalFilter) return false;
    if(indexFilter!=='ALL' && r.Index!==indexFilter) return false;
    if(search && !r.Ticker.toLowerCase().includes(search)) return false;
    return true;
  }});

  if(sortKey==='rank'){{
    data.sort((a,b)=>sortDir*(b.Score-a.Score));
  }} else if(typeof data[0]?.[sortKey]==='number'){{
    data.sort((a,b)=>sortDir*(a[sortKey]-b[sortKey]));
  }} else {{
    data.sort((a,b)=>sortDir*(a[sortKey]||'').localeCompare(b[sortKey]||''));
  }}

  const tbody = document.getElementById('tbody');
  if(data.length===0){{
    tbody.innerHTML='';
    document.getElementById('empty').style.display='block';
    document.getElementById('count-label').textContent='';
    return;
  }}
  document.getElementById('empty').style.display='none';

  tbody.innerHTML = data.map((r,i)=>{{
    const chgCls = r.Change_Pct>=0?'pos':'neg';
    const wkCls  = r.Week_Chg>=0?'pos':'neg';
    const macdCls= r.MACD_Hist>=0?'macd-pos':'macd-neg';
    const macdSym= r.MACD_Hist>=0?'▲':'▼';
    const tgtCls = r.Signal==='SELL'?'neg':'pos';
    const slCls  = 'neg';
    const rrColor= r.RR>=2?'#00D4AA':r.RR>=1?'#F5A623':'#C0205A';
    const tgtPct = r.Signal==='SELL'
      ? ((r.Target_5D-r.Price)/r.Price*100).toFixed(1)
      : ('+'+((r.Target_5D-r.Price)/r.Price*100).toFixed(1));
    const slPct  = (((r.Stop_Loss-r.Price)/r.Price)*100).toFixed(1);
    return `<tr>
      <td class="rank">${{i+1}}</td>
      <td class="ticker">${{r.Ticker}}</td>
      <td><span class="idx-badge ${{idxClass(r.Index)}}">${{idxShort(r.Index)}}</span></td>
      <td class="price">${{r.Price.toLocaleString()}}</td>
      <td class="chg ${{chgCls}}">${{r.Change_Pct>=0?'+':''}}${{r.Change_Pct}}%</td>
      <td class="chg ${{wkCls}}">${{r.Week_Chg>=0?'+':''}}${{r.Week_Chg}}%</td>
      <td class="rsi-val ${{rsiClass(r.RSI)}}">${{r.RSI}}</td>
      <td class="${{macdCls}}">${{macdSym}} ${{Math.abs(r.MACD_Hist).toFixed(3)}}</td>
      <td class="adx-val" style="color:${{r.ADX>=25?'#00D4AA':r.ADX>=20?'#0388BC':'#7B71B0'}}">${{r.ADX}}</td>
      <td class="price" style="color:${{r.Vol_Ratio>=1.5?'#00D4AA':r.Vol_Ratio>=1.1?'#0388BC':'#7B71B0'}}">${{r.Vol_Ratio}}×</td>
      <td class="price ${{tgtCls}}" title="${{tgtPct}}%">${{r.Target_5D.toLocaleString()}} <span style="font-size:10px;opacity:.7">(${{tgtPct}}%)</span></td>
      <td class="price ${{slCls}}" title="${{slPct}}%">${{r.Stop_Loss.toLocaleString()}} <span style="font-size:10px;opacity:.7">(${{slPct}}%)</span></td>
      <td style="font-family:var(--mono);font-size:11px;font-weight:700;color:${{rrColor}}">${{r.RR}}×</td>
      <td>
        <div class="score-bar">
          <div class="bar-bg"><div class="bar-fill" style="width:${{r.Score}}%;background:${{scoreColor(r.Score)}}"></div></div>
          <span class="score-num">${{r.Score}}</span>
        </div>
      </td>
      <td><span class="sig-badge sig-${{r.Signal}}">${{r.Signal}}</span></td>
    </tr>`;
  }}).join('');

  document.getElementById('count-label').textContent = `Showing ${{data.length}} of ${{RAW.length}} stocks`;
}}

// Fetch live index data
async function fetchIndex(sym, priceId, chgId){{
  try{{
    const r = await fetch(`https://query1.finance.yahoo.com/v8/finance/chart/${{sym}}?interval=1d&range=2d`);
    const d = await r.json();
    const closes = d.chart.result[0].indicators.quote[0].close;
    const prev = closes[closes.length-2], curr = closes[closes.length-1];
    const chg = ((curr-prev)/prev*100).toFixed(2);
    const cls = chg>=0?'pos':'neg';
    document.getElementById(priceId).textContent = curr.toLocaleString('en-IN',{{maximumFractionDigits:2}});
    document.getElementById(chgId).innerHTML = `<span class="${{cls}}">${{chg>=0?'+':''}}${{chg}}%</span>`;
  }}catch(e){{}}
}}
fetchIndex('%5ENSEI','n50-px','n50-chg');
fetchIndex('%5ECNX100','n100-px','n100-chg');
fetchIndex('%5EGSPC','sp500-px','sp500-chg');

render();
</script>
</body>
</html>"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n✅ Dashboard saved → {output_file}")
    print(f"   Open in browser after market close each day.")


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("  MOMENTUM SWING TRACKER")
    print("  Run after market close for fresh signals")
    print("=" * 55)

    all_results = []

    # Nifty 50
    all_results += analyze(NIFTY_50, 'NIFTY 50')

    # Nifty 100 (extra 50 stocks)
    all_results += analyze(NIFTY_100_EXTRA, 'NIFTY 100')

    # S&P 500 (full or fallback)
    sp500_tickers = fetch_sp500_list()
    all_results += analyze(sp500_tickers, 'S&P 500')

    print(f"\n📊 Total stocks analysed: {len(all_results)}")
    buy  = sum(1 for r in all_results if r['Signal']=='BUY')
    hold = sum(1 for r in all_results if r['Signal']=='HOLD')
    sell = sum(1 for r in all_results if r['Signal']=='SELL')
    print(f"   ▲ BUY:  {buy}  |  ◆ HOLD: {hold}  |  ▼ SELL: {sell}")

    generate_html(all_results)
