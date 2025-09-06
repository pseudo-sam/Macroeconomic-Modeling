# =========================
# Macro–Equity Research Pipeline
# =========================
# What it does:
# - Download FRED macro series (GDP, CPI, debt/GDP, unemployment, federal interest payments)
# - Download S&P 500 (^GSPC)
# - Consolidate to a single monthly panel
# - Run OLS (drivers of federal interest payments)
# - Fit ARIMA to S&P 500 returns
# - Granger causality tests: macro -> S&P 500 returns (and vice versa if desired)
# - Rolling correlations between S&P 500 returns and macro indicators
#
# Note: You can swap/extend series IDs as needed.

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pandas_datareader import data as fred
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA

# ------------- Config -------------
START = "1990-01-01"
END   = None  # None = today

FRED_SERIES = {
    "GDP": "GDPC1",                     
    "CPI": "CPIAUCSL",                  
    "UNRATE": "UNRATE",                 
    "DEBT_GDP": "GFDEGDQ188S",          
    "FED_INT": "A091RC1Q027SBEA"
}

SP_TICKER = "^GSPC"  # S&P 500

def _dl_fred(series_id, start=START, end=END):
    s = fred.DataReader(series_id, "fred", start=start, end=end).dropna()
    s.name = series_id
    return s

def download_macro():
    data = {}
    for pretty, sid in FRED_SERIES.items():
        data[pretty] = _dl_fred(sid)
    return data

def download_sp500(start=START, end=END):
    px = yf.download(SP_TICKER, start=start, end=end, progress=False)["Adj Close"].dropna()
    px_m = px.resample("M").last()
    rets_m = np.log(px_m).diff()  
    rets_m.name = "SPX_RET"
    return px_m.rename("SPX"), rets_m

def monthlyize(series: pd.Series, how="ffill"):
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    if pd.infer_freq(s.index) and pd.infer_freq(s.index).startswith("Q"):
        s_qe = s.asfreq("Q")  
        s_m = s_qe.resample("M").ffill()
    else:
        s_m = s.resample("M").last()
        if how == "ffill":
            s_m = s_m.ffill()
    return s_m

def build_panel():
    macro = download_macro()
    spx, spx_ret = download_sp500()

    gdp_m      = monthlyize(macro["GDP"])      
    cpi_m      = monthlyize(macro["CPI"])      
    unrate_m   = monthlyize(macro["UNRATE"])   
    debtgdp_m  = monthlyize(macro["DEBT_GDP"]) 
    fedint_m   = monthlyize(macro["FED_INT"])  

    infl_yoy   = np.log(cpi_m).diff(12) * 100
    infl_yoy.name = "INF_YoY"

    gdp_yoy    = np.log(gdp_m).diff(12) * 100
    gdp_yoy.name = "GDP_YoY"

    d_unrate   = unrate_m.diff()
    d_unrate.name = "dUNRATE"

    d_debtgdp  = debtgdp_m.diff()
    d_debtgdp.name = "dDEBT_GDP"

    fedint_yoy = np.log(fedint_m).diff(12) * 100
    fedint_yoy.name = "FEDINT_YoY"

    panel = pd.concat(
        [spx, spx_ret, cpi_m.rename("CPI"), infl_yoy, gdp_m.rename("GDP"),
         gdp_yoy, unrate_m.rename("UNRATE"), d_unrate, debtgdp_m.rename("DEBT_GDP"),
         d_debtgdp, fedint_m.rename("FEDINT"), fedint_yoy],
        axis=1
    ).dropna()

    return panel

def run_ols(panel: pd.DataFrame):
    df = panel.copy()
    y = df["FEDINT_YoY"]
    X = df[["GDP_YoY", "INF_YoY", "UNRATE", "DEBT_GDP"]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing="drop")
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    return res

def run_arima(panel: pd.DataFrame, order=(1,0,1)):
    y = panel["SPX_RET"].dropna()
    model = ARIMA(y, order=order)
    res = model.fit()
    return res

def run_granger(panel: pd.DataFrame, maxlag=6):
    out = {}
    df = panel[["SPX_RET", "INF_YoY", "GDP_YoY", "dUNRATE", "dDEBT_GDP"]].dropna()
    for var in ["INF_YoY", "GDP_YoY", "dUNRATE", "dDEBT_GDP"]:
        sub = df[["SPX_RET", var]].dropna()
        try:
            out[var] = grangercausalitytests(sub.values, maxlag=maxlag, verbose=False)
        except Exception as e:
            out[var] = {"error": str(e)}
    return out

def rolling_correlations(panel: pd.DataFrame, window=36):
    df = panel[["SPX_RET", "INF_YoY", "GDP_YoY", "dUNRATE", "dDEBT_GDP"]].dropna()
    rollcorr = pd.DataFrame(index=df.index)
    for col in ["INF_YoY", "GDP_YoY", "dUNRATE", "dDEBT_GDP"]:
        rollcorr[f"corr(SPX_RET,{col})"] = df["SPX_RET"].rolling(window).corr(df[col])
    return rollcorr.dropna()

if __name__ == "__main__":
    panel = build_panel()
    panel.to_csv("macro_panel_monthly.csv")
    print("Saved: macro_panel_monthly.csv")
    print(panel.tail())

    print("\n=== OLS: Drivers of Federal Interest Payments (YoY) ===")
    ols_res = run_ols(panel)
    print(ols_res.summary())

    print("\n=== ARIMA(1,0,1) on Monthly S&P 500 Log Returns ===")
    arima_res = run_arima(panel, order=(1,0,1))
    print(arima_res.summary())

    print("\n=== Granger Causality: Do macros help predict SPX returns? ===")
    granger_res = run_granger(panel, maxlag=6)
    for var, res in granger_res.items():
        if isinstance(res, dict) and "error" in res:
            print(f"{var}: ERROR -> {res['error']}")
            continue
        best_p = 1.0
        best_lag = None
        for lag, r in res.items():
            if "ssr_ftest" in r[0]:
                p = r[0]["ssr_ftest"][1]
                if p < best_p:
                    best_p, best_lag = p, lag
        print(f"{var}: best p={best_p:.4g} at lag {best_lag}")

    print("\n=== 36m Rolling Correlations (SPX returns vs macro) ===")
    rollcorr = rolling_correlations(panel, window=36)
    rollcorr.to_csv("rolling_correlations_36m.csv")
    print("Saved: rolling_correlations_36m.csv")
    print(rollcorr.tail())
