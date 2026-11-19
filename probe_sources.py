import traceback, datetime as dt
def t(name, f):
    try:
        r = f(); print(f"[OK] {name}: {r}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:200]}")

def naver_bond():
    import requests, pandas as pd, io
    url="https://finance.naver.com/marketindex/interestDailyQuote.naver?marketindexCd=IRR_GOVT03Y&page=1"
    h={"User-Agent":"Mozilla/5.0"}
    r=requests.get(url,headers=h,timeout=15); r.encoding="euc-kr"
    df=pd.read_html(io.StringIO(r.text))[0]
    return f"status {r.status_code}, rows {len(df)}, head {df.iloc[0].tolist()}"

def naver_bond_deep():
    import requests, pandas as pd, io
    h={"User-Agent":"Mozilla/5.0"}
    r=requests.get("https://finance.naver.com/marketindex/interestDailyQuote.naver?marketindexCd=IRR_GOVT03Y&page=150",headers=h,timeout=15); r.encoding="euc-kr"
    df=pd.read_html(io.StringIO(r.text))[0]
    return f"page150 rows {len(df)}, {df.iloc[0].tolist()}"

def pykrx_foreign():
    from pykrx import stock
    df=stock.get_market_trading_value_by_date("20241101","20241130","KOSPI")
    return f"rows {len(df)}, cols {list(df.columns)[:6]}, last {df.iloc[-1].tolist()[:6]}"

def pykrx_foreign_old():
    from pykrx import stock
    df=stock.get_market_trading_value_by_date("20190101","20190131","KOSPI")
    return f"rows {len(df)}"

def fred():
    import requests, pandas as pd, io
    r=requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2",timeout=20)
    df=pd.read_csv(io.StringIO(r.text)); return f"DGS2 rows {len(df)}, last {df.iloc[-1].tolist()}"

def naver_foreign():
    import requests, pandas as pd, io
    h={"User-Agent":"Mozilla/5.0"}
    r=requests.get("https://finance.naver.com/sise/investorDealTrendDay.naver?bizdate=20241129&sosok=01&page=1",headers=h,timeout=15); r.encoding="euc-kr"
    dfs=pd.read_html(io.StringIO(r.text)); return f"status {r.status_code}, tables {len(dfs)}, {dfs[0].dropna().head(2).values.tolist()}"

def yf_extra():
    import yfinance as yf
    d=yf.download(["EWY","CNH=X","CNY=X","JPY=X","TWD=X","^SOX","^GSPC","2YY=F","^FVX"],start="2019-01-01",progress=False)["Close"]
    return {c:int(d[c].notna().sum()) for c in d.columns}

def ecos_nokey():
    import requests
    r=requests.get("https://ecos.bok.or.kr/api/StatisticSearch/sample/json/kr/1/5/817Y002/D/20240101/20240110/010200000",timeout=15)
    return r.text[:200]

for n,f in [("naver 국고채3년",naver_bond),("naver 국고채 오래된 페이지",naver_bond_deep),("pykrx 외국인(최근)",pykrx_foreign),
            ("pykrx 외국인(2019)",pykrx_foreign_old),("naver 투자자별 매매동향",naver_foreign),("FRED DGS2",fred),
            ("yfinance 추가",yf_extra),("ECOS sample key",ecos_nokey)]:
    t(n,f)
