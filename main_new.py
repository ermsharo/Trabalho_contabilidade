import pandas as pd
from pandas_datareader import data
import numpy as np
import matplotlib.pyplot as plt


wallet_list = ["AMZN", "GOOG", "MSFT", "NFLX"]


def generate_df_closed(wallet_list, start_date):
    df = pd.DataFrame()
    for wall in wallet_list:
        df[wall] = data.DataReader(wall, data_source="yahoo", start=start_date)["Close"]
    df.head()
    df.pct_change().head()

    return df


def list_to_array(list):
    arr = np.array(list)
    return arr


def df_to_file(list):
    arr = np.array(list)
    return arr



def generate_wallets(df_close, num_portfolios=10000, risk_free=0):
    # vetores de dados
    portfolio_weights = []
    portfolio_exp_returns = []
    portfolio_vol = []
    portfolio_sharpe = []

    # retorno simples
    r = df_close.pct_change()
    mean_returns = r.mean() * 252

    # matriz de covariância
    covariance = np.cov(r[1:].T)

    for i in range(num_portfolios):
        # gerando pesos aleatórios
        k = np.random.rand(len(df_close.columns))
        w = k / sum(k)

        # retorno
        R = np.dot(mean_returns, w)

        # risco
        vol = np.sqrt(np.dot(list_to_array(w).T, np.dot(covariance, w))) * np.sqrt(252)

        # sharpe ratio
        sharpe = (R - risk_free) / vol

        portfolio_weights.append(w)
        portfolio_exp_returns.append(R)
        portfolio_vol.append(vol)
        portfolio_sharpe.append(sharpe)

    wallets = {
        "weights": portfolio_weights,
        "returns": portfolio_exp_returns,
        "vol": portfolio_vol,
        "sharpe": portfolio_sharpe,
    }

    return wallets



print(" \n \n Gerando o dataframe que usaremos"); 
generated_closed_df = generate_df_closed(wallet_list,"1-1-2010")
generated_wallets = generate_wallets(generated_closed_df)

