import numpy as np
import pandas as pd


def get_ragged_nested_index_lists(df, id_col_list):
    """Gets the ragged nested lists of indices (= row numbers of ``df``).
    Hierarchy in the nesting is set up by the `df` columns with names
    in ``id_col_list``.
    
    :param df: Pandas Data Frame with the sample. It ias to contain columns listed in ``id_col_list``.
    :param id_col_list: list of columns to ``df`` which correspond to the levels of nesting in the 
        resulting index list. Higher-level groups have to be mentioned first, e.g. ``['grp_id', 'subgrp_id']``.
    
    :returns: Pandas DF with two columns: copy of ``df[id_col_list[0]]`` and column ``'_row_'`` containing
        nested list of row numbers, which is input to decorator ``xgb_tf_loss()``.
    """ 
    df_idx = df[id_col_list].copy()
    df_idx['_row_'] = list(range(len(df_idx)))
    remaining_cols = list(id_col_list)
    while remaining_cols:
        c = remaining_cols.pop()
        df_idx = df_idx.groupby(c).agg({'_row_':lambda x: list(x), **{rc:max for rc in remaining_cols}}).reset_index()
    return df_idx


def gen_random_dataset(n, n_subgrp, n_grp, beta, sigma):
    """
    Generate random Pandas Data Frame with ``n`` observations split to ``n_subgrp``
    distinct subgroups described by column ``'subgrp_id'`` and ``'n_grp'`` distinct 
    groups described by column ``'grp_id'``. The target in column ``'y'`` is
    linear combination of feature vector in column ``'X'`` with true coefficient 
    vector ``beta`` and standard error ``sigma``. Intercept is zero.
    
    :param n: number of observations
    :param n_subgrp: number of subgroups
    :param n_grp: number of groups
    :param beta: true coefficients
    :param sigma: standard error
    
    :returns: random dataset
    """
    assert n >= n_subgrp
    assert n_subgrp >= n_grp

    def gen_rand_sample(n_obs, beta, sigma):
        n_feats = len(beta)
        X = np.random.randn(n_obs, n_feats)
        epsilon = np.random.randn(n_obs)*sigma
        y = np.matmul(X, beta) + epsilon
        return X,y

    def gen_rand_grp_id(n_obs, n_grp, prefix='GRP'):
        grp_num = np.concatenate((
            np.arange(n_grp, dtype='I'),
            np.random.randint(0,n_grp,size=n_obs-n_grp)
            ))
        grp = np.asarray([f'{prefix}{i}' for i in grp_num])
        return grp

    X,y = gen_rand_sample(n, beta, sigma)
    df = pd.DataFrame(data={f'_row_':list(range(len(y))), 'X':X.tolist(), 'y':y})
    if n_subgrp:
        subgrp_ids = gen_rand_grp_id(n, n_subgrp, prefix='SUBGRP')
        subgrp_ids_distinct = list(set(subgrp_ids.tolist()))
        df['subgrp_id'] = subgrp_ids
        if n_grp:
            grp_ids = gen_rand_grp_id(len(subgrp_ids_distinct), n_grp, prefix='GRP')
            subgrp2grp_id_dict = {k:v for k,v in zip(subgrp_ids_distinct, grp_ids)}
            df['grp_id'] = df['subgrp_id'].map(subgrp2grp_id_dict)
    return df