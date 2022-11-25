import numpy as np
import pandas as pd


def balance(
    s: pd.Series,
    gain: float = 0.5,
) -> pd.Series:
    """
    calulate rough balansed log lambda. 
    because coupled product demand seems to be similar.

    Parameters
    ----------
    s
        series
    gain
        demand increase ratio to lower demand products in a couple

    Returns
    -------
        series with rough balansed log lambda
    """

    if -1 not in s['C_ID']:
        _max = s['LOG_LAMBDA'].max()
        s['LOG_LAMBDA'] = s['LOG_LAMBDA'].apply(lambda x : x + (abs(_max - x) * gain))
    
    return s


def simulate_coupling_demand(
    df: pd.DataFrame,
    n_couples: int,
    max_products: int = 3,
) -> pd.DataFrame:
    """
    Simulates coupling demand.
    In many cases, Bread and Jam are usually bought same time.

    Parameters
    ----------
    df
        dataframe
    n_couples
        number of couples to simulate
    max_products
        max number of products included in one-couple

    Returns
    -------
        dataframe with added ``C_ID`` and coupling demand effect added on the log lambda column.
        if C_ID is -1, it means no coupled products.
    """

    couple_map = {}
    p_id = df['P_ID'].unique()

    # couple
    for c_id in range(n_couples):
        prods = np.random.randint(low=2, high=max_products+1)
        couple = np.random.choice(p_id, size=prods, replace=False)
        for c in couple:
            couple_map[c] = c_id
        p_id = list(set(p_id) - set(couple)) # not allow overlapping

    # not couple
    for c in p_id:
        couple_map[c] = -1

    # add couple id as column
    df['C_ID'] = pd.Series([couple_map[x] for x in df['P_ID']])

    # add coupling demand effect
    df['LOG_LAMBDA'] = df.groupby(["C_ID", "L_ID"], group_keys=False).apply(balance, 0.5)['LOG_LAMBDA']

    return df