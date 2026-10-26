"""All NSE stocks + industry equal-weighted indices + position vs 10/20-day EMA -> industry.json
Setup:  pip install yfinance pandas requests
Run:    python nse_industry.py --start 2005-01-01  # history start date (default 2005-01-01)
        python nse_industry.py                 # Yahoo 'industry' (~140 groups) for every stock
        python nse_industry.py --level sector  # Yahoo 'sector' (11 broad groups) instead
        python nse_industry.py --keep-nse      # prefer NSE index-file industries where available,
                                               # Yahoo only for the rest (mixes two taxonomies)
Upload industry.json in the portal.

Yahoo classifications are cached in yahoo_industry_cache.json, so the first run is slow
(one request per stock) and later runs only look up new symbols. Failed lookups are retried
next run. Optional extra_industries.csv (Symbol,Industry) overrides everything.
"""
import io, os, sys, json, time
from concurrent.futures import ThreadPoolExecutor
import requests, pandas as pd, yfinance as yf

CHUNK, STALE_DAYS, WORKERS = 250, 7, 6
START = sys.argv[sys.argv.index("--start") + 1] if "--start" in sys.argv else "2005-01-01"  # history start
LEVEL = sys.argv[sys.argv.index("--level") + 1] if "--level" in sys.argv else "industry"
KEEP_NSE = "--keep-nse" in sys.argv
CACHE = f"yahoo_{LEVEL}_cache.json"
HDR = {"User-Agent": "Mozilla/5.0"}


def csv(url):
    r = requests.get(url, headers=HDR, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    return df


# 1) all NSE EQ-series symbols
eq = csv("https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv")
eq = eq[eq["SERIES"].str.strip() == "EQ"]
syms = sorted({s.strip() for s in eq["SYMBOL"].dropna()})

# 2) prices
frames = []
for k in range(0, len(syms), CHUNK):
    part = [s + ".NS" for s in syms[k:k + CHUNK]]
    for _ in range(3):
        try:
            c = yf.download(part, start=START, auto_adjust=True, progress=False, threads=True)
            frames.append(c[["Open", "High", "Low", "Close", "Volume"]])
            break
        except Exception as e:
            print("retry", e); time.sleep(5)
    print(f"  prices {min(k + CHUNK, len(syms))}/{len(syms)}")
px = pd.concat(frames, axis=1)
px = px.loc[:, ~px.columns.duplicated()].dropna(how="all")
OHLC = {f: px[f].dropna(how="all", axis=1) for f in ["Open", "High", "Low", "Close", "Volume"]}
close = OHLC["Close"]
last = close.apply(lambda s: s.last_valid_index())
keep = close.columns[last >= close.index[-1] - pd.Timedelta(days=STALE_DAYS)]
for f in OHLC:
    OHLC[f] = OHLC[f].reindex(columns=keep)
    OHLC[f].columns = [c.replace(".NS", "") for c in keep]
close = OHLC["Close"]
live = list(close.columns)

# 3) industry map
ind = {}
if KEEP_NSE:
    base = "https://www.niftyindices.com/IndexConstituent/"
    for f in ["ind_niftytotalmarket_list.csv", "ind_niftymicrocap250_list.csv", "ind_nifty500list.csv"]:
        try:
            for s, i in csv(base + f)[["Symbol", "Industry"]].dropna().values:
                ind.setdefault(s.strip(), i.strip())
        except Exception as e:
            print("skip", f, e)

cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def lookup(s):
    for attempt in range(3):
        try:
            v = yf.Ticker(s + ".NS").get_info().get(LEVEL)
            if v:
                return s, v.strip()
            return s, None
        except Exception:
            time.sleep(2 + attempt * 3)
    return s, None


todo = [s for s in live if s not in ind and s not in cache]
print(f"Yahoo {LEVEL} lookups needed: {len(todo)} (cached: {len(cache)})")
if todo:
    with ThreadPoolExecutor(WORKERS) as ex:
        for n, (s, v) in enumerate(ex.map(lookup, todo), 1):
            if v:
                cache[s] = v
            if n % 100 == 0:
                json.dump(cache, open(CACHE, "w"))
                print(f"  yahoo {n}/{len(todo)}")
    json.dump(cache, open(CACHE, "w"))
for s in live:
    ind.setdefault(s, cache.get(s, "Unclassified"))
if os.path.exists("extra_industries.csv"):
    for s, i in pd.read_csv("extra_industries.csv")[["Symbol", "Industry"]].dropna().values:
        ind[s.strip()] = i.strip()
print(f"{len(live)} live stocks, {sum(ind[s] != 'Unclassified' for s in live)} classified")

# 4) stock metrics: distance from 10 / 20-day EMA
ema10, ema20 = close.ewm(span=10, adjust=False, min_periods=10).mean(), close.ewm(span=20, adjust=False, min_periods=20).mean()
ret = close.pct_change(fill_method=None)
rows = []
for s in live:
    c, a, b = close[s].iloc[-1], ema10[s].iloc[-1], ema20[s].iloc[-1]
    if pd.isna(c) or pd.isna(a) or pd.isna(b):
        continue
    ch = ret[s].iloc[-1]
    rows.append({"s": s, "i": ind[s], "c": round(float(c), 2), "ch": round(float(ch * 100), 2) if pd.notna(ch) else 0,
                 "d10": round(float((c / a - 1) * 100), 2), "d20": round(float((c / b - 1) * 100), 2)})

# 5) equal-weighted industry index OHLC (base 100). Each day, O/H/L/C = previous index close x
#    (1 + average member return of that field vs the members' previous close). Daily only -
#    the portal builds weekly/monthly bars from it.
def build_index(mem):
    prev = close[mem].shift(1)
    rr = lambda f: (OHLC[f][mem] / prev - 1).mean(axis=1)
    rc = rr("Close").fillna(0)
    rc.iloc[0] = 0
    c = 100 * (1 + rc).cumprod()
    p = c.shift(1)
    o, h, l = (p * (1 + rr(f).fillna(0)) for f in ["Open", "High", "Low"])
    o.iloc[0] = h.iloc[0] = l.iloc[0] = 100
    h = pd.concat([h, o, c], axis=1).max(axis=1)
    l = pd.concat([l, o, c], axis=1).min(axis=1)
    res = {k: [round(float(v), 2) for v in x] for k, x in zip("ohlc", (o, h, l, c))}
    # turnover (Rs crore) = sum of close x volume over member stocks, from Yahoo data (approximate)
    res["t"] = [round(float(v), 2) for v in (close[mem] * OHLC["Volume"][mem]).sum(axis=1) / 1e7]
    return res


inds = {name: build_index([r["s"] for r in rows if r["i"] == name]) for name in sorted({r["i"] for r in rows})}
dates = [str(d.date()) for d in close.index]
out = {"asof": dates[-1], "dates": dates, "inds": inds, "stocks": rows}
json.dump(out, open("industry.json", "w"), separators=(",", ":"))
print("Wrote industry.json -", out["asof"], f"({len(rows)} stocks, {len(inds)} industries, {len(dates)} days)")
