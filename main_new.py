import pandas as pd
from pandas_datareader import data
import numpy as np
import matplotlib.pyplot as plt


wallet_list = ["AMZN", "GOOG", "MSFT", "NFLX"]


def list_to_array(list):
    arr = np.array(list)
    return arr


def generate_df_closed(wallet_list, start_date, end_date):
    df = pd.DataFrame()
    for wall in wallet_list:
        df[wall] = data.DataReader(wall, data_source="yahoo", start=start_date, end=end_date)["Close"]
    df.head()
    df.pct_change().head()
    return df


def ative_retuns(df):
    r = df.pct_change()
    return r


def portfolio_retuns(df, w):
    R = np.dot(ative_retuns(df), w)
    return R  # retorno do portfólio


def get_risk(df, w):
    # retorno simples
    r = df.pct_change()
    # média dos retornos anualizados
    mean_returns = r.mean() * 252
    # matriz de covariância
    covariance = np.cov(r[1:].T)
    # Risco do portfólio anualizado
    risc_vol = np.sqrt(np.dot(list_to_array(w).T, np.dot(covariance, w))) * np.sqrt(252)

    return risc_vol


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
        vol = np.sqrt(np.dot(w.T, np.dot(covariance, w))) * np.sqrt(252)

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


def best_portfolio(wallets, method="sharpe_ratio"):
    vol = wallets["vol"]
    sharpe = wallets["sharpe"]
    weights = wallets["weights"]
    returns = wallets["returns"]

    if method == "sharpe_ratio":

        indice = np.array(sharpe).argmax()

    elif method == "volatility":

        indice = np.array(vol).argmin()

    elif method == "return":

        indice = np.array(returns).argmax()

    return weights[indice]


def plot_efficient_frontier(wallets, method="sharpe_ratio"):
    vol = wallets["vol"]
    returns = wallets["returns"]
    sharpe = wallets["sharpe"]

    if method == "sharpe_ratio":

        indice = np.array(sharpe).argmax()
        y_axis = returns[indice]
        X_axis = vol[indice]

    elif method == "volatility":

        indice = np.array(vol).argmin()
        y_axis = returns[indice]
        X_axis = vol[indice]

    elif method == "return":

        indice = np.array(returns).argmax()
        y_axis = returns[indice]
        X_axis = vol[indice]

    plt.scatter(vol, returns, c=sharpe, cmap="viridis")
    plt.scatter(X_axis, y_axis, c="red", s=50)
    plt.title("Efficient Frontier")
    plt.xlabel("Volatility")
    plt.ylabel("Expected return")
    plt.show()


## Atribuindo nossos pesos
w = [0.3, 0.3, 0.2, 0.2]

## Gerando o dataframe que usaremos
generated_closed_df = generate_df_closed(wallet_list, "1-1-2010","1-1-2022")

# Retorno dos nossos ativos
df_active_returns = ative_retuns(generated_closed_df)

# Retorno do nosso portfolio
df_portf_returns = portfolio_retuns(generated_closed_df, w)

## Analisando o risco
df_risk_returns = get_risk(generated_closed_df, w)

## Gerando as nossas carteiras
generated_wallets = generate_wallets(generated_closed_df)

## Filtrando o melhor portfolio
best_port = best_portfolio(generated_wallets, method="sharpe_ratio")

## Executando nosso grafico
plot_efficient_frontier(generated_wallets, method="sharpe_ratio")
plot_efficient_frontier(generated_wallets, method="volatility")

## Executando um segundo caso com outras carteiras


wallet_list_2 = ["GE", "GOOG", "IBM", "NFLX"]


generated_closed_df_2= generate_df_closed(wallet_list_2, "1-1-2010","1-1-2022")
generated_wallets_2 = generate_wallets(generated_closed_df_2)
plot_efficient_frontier(generated_wallets_2, method="sharpe_ratio")
plot_efficient_frontier(generated_wallets_2, method="volatility")
