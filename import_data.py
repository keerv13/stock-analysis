
import yfinance as yf
import pandas as pd

stocks = {
    "Maybank": "1155.KL",
    "MRDIY": "5296.KL",
    "Tenaga_Nasional": "5347.KL",
}

for company, ticker in stocks.items():
    df = yf.download(
        ticker,
        period="5y",
        auto_adjust=False
    )

    if df.empty:
        print(f"No data returned for {company} ({ticker})")
        continue

    df.to_csv(f"assets/{company}.csv")
    print(company, df.shape)
    print(df.tail(3))