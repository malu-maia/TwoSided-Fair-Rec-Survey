# %%
import numpy as np
import pandas as pd

import torch
import json

from scipy.sparse import coo_matrix
from sklearn.decomposition import NMF
from nmf import run_nmf

# %%
def get_q(S: np.ndarray) -> np.ndarray:
    """
    Compute the sum of the scores along the user dimension.

    Parameters:
    - S: input array of shape (n, m)

    Returns:
    - Array containing the quality of each item
    """
    return S.sum(axis=0)

def get_u(P: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Efficiently compute vector u of size m where each element is derived from P and v.

    Parameters:
    - P: array of shape (n, m, k)
    - v: array of shape (k,)

    Returns:
    - Array of shape (m,)
    """
    n, m, k = P.shape

    # Element-wise product with broadcasting
    products = P * v.reshape(1, 1, k)

    # Sum over k and n dimensions
    summed = products.sum(axis=2)
    result = summed.sum(axis=0)

    return result

def get_P(S: np.ndarray, v: np.ndarray, k: int) -> np.ndarray:
    """
    Generate array P based on scores S, expose vector v, and recommendation size k.

    Parameters:
    - S: np.ndarray, score matrix of shape (n, m)
    - v: np.ndarray, value vector of shape (m,)
    - k: int, dimension to expand the array to

    Returns:
    - P: np.ndarray, array of shape (n, m, k)
    """
    n, m = S.shape
    # Add new axis and broadcast to k elements in the third dimension
    P = np.broadcast_to(S[..., np.newaxis], (n, m, k))
    return P

def get_data_name(data_path):
    name = data_path.split('/')[-1]
    name = name.split('.')[0]
    if name == 'movielens_100k_u1':
        name = 'movielens'
    print(f'Data name: {name}')
    return name


def get_utility(pred_scores: np.array, top_k: np.array, k: int, axis=0):
    """
    Use this function or plot the Lorenz curve
    axis: 0 if we want to compute item utility, and 1 to compute user utility
    """
    mask = np.zeros_like(pred_scores)
    weights = [1/(np.log2(i)+2) for i in range(k)]
    for i in range(mask.shape[0]):
        mask[i, top_k[i,:k]] = 1 * weights
    if axis == 0:
        print('Item utility')
    else:
        print('User utility')
    return pd.DataFrame((mask * pred_scores).sum(axis=axis))

def topk_from_P(P_stochastic, k: int=10):
    _, topk_positions = torch.max(P_stochastic, dim=2)
    fair_topk = torch.argsort(topk_positions, dim=1)
    return _, fair_topk[:, :k]

def get_active_users(data_path, run_i:int, columns_df: list[str]=['user_id', 'item_id', 'rating', 'timestamp']):
    # get temp_indexes_data that stores the sampled rows to get the most active users
    data = pd.read_csv(data_path, sep="\t", names=columns_df, index_col=False)
    data = data[data.columns[:2]]

    with open(f"data/temp_data/temp_indexes_{data_path.split('/')[-1].split('.')[0]}_run{run_i}.npz", 'rb') as f:
        indexes = np.load(f)

    sampled_data = data.loc[indexes]
    users_idx = np.unique(sampled_data['user_id'])
    mapping_users_id = {users_idx[i]: i for i in range(len(users_idx))}
    
    users_count = sampled_data['user_id'].value_counts()

    most_active_percent = int(0.05*sampled_data.shape[0])
    top_active = list(users_count[:most_active_percent].index)
    
    top_active = [mapping_users_id[u] for u in top_active]

    groups = {}
    for i in mapping_users_id.values():
        if i in top_active:
            groups[i] = -1
        else:
            groups[i] = 1 

    return groups

def get_top_items(data_path, run_i:int, columns_df: list[str]=['user_id', 'item_id', 'rating', 'timestamp']):
    # get temp_indexes_data that stores the sampled rows to get the most interacted items
    data = pd.read_csv(data_path, sep="\t", names=columns_df, index_col=False)
    data = data[data.columns[:2]]

    with open(f"data/temp_data/temp_indexes_{data_path.split('/')[-1].split('.')[0]}_run{run_i}.npz", 'rb') as f:
        indexes = np.load(f)

    sampled_data = data.loc[indexes]
    items_idx = np.unique(sampled_data['item_id'])
    mapping_items_id = {items_idx[i]: i for i in range(len(items_idx))}

    items_count = sampled_data['item_id'].value_counts()

    most_interated_percent = int(0.2*sampled_data.shape[0])
    top_interacted_items = list(items_count[:most_interated_percent].index)
    
    top_interacted_items = [mapping_items_id[i] for i in top_interacted_items]

    groups = {}
    for i in mapping_items_id.values():
        if i in top_interacted_items:
            groups[i] = -1
        else:
            groups[i] = 1 

    return groups


def mapping_indexes(scores: np.ndarray, rows_idx: np.ndarray, cols_idx: np.ndarray) -> np.ndarray:
    # Ensure all input arrays have the same length
    if not (len(scores) == len(rows_idx) == len(cols_idx)):
        raise ValueError("scores, rows_idx, and cols_idx must have the same length")

    # Map original IDs to a contiguous range starting from 0
    unique_rows = sorted(list(set(rows_idx)))
    unique_cols = sorted(list(set(cols_idx)))
    row_mapping = {id: index for index, id in enumerate(unique_rows)}
    col_mapping = {id: index for index, id in enumerate(unique_cols)}

    mapped_rows_idx = [row_mapping[id] for id in rows_idx]
    mapped_cols_idx = [col_mapping[id] for id in cols_idx]

    return mapped_rows_idx, mapped_cols_idx

def pivot_gen_score_matrix(scores: np.ndarray, rows_idx: np.ndarray, cols_idx: np.ndarray) -> np.ndarray:
    return coo_matrix((scores, (rows_idx, cols_idx))).toarray().astype(float)

def load_data(path: str, run_i: int, n=None, limit: bool=True, random_state=42):
    data = pd.read_csv(path, sep='\t', header=None)
    #if limit:
    if n != None:
        data = data.sample(n=n, random_state=random_state)
    indexes_data = data.index
    with open(f"data/temp_data/temp_indexes_{path.split('/')[-1].split('.')[0]}_run{run_i}.npz", 'wb') as f:
        np.save(f, indexes_data)

    data = data.to_numpy()

    if data.shape[1] == 4: # for data that has timestamp
        data = data[:, :-1]
    max_score = data[:, 2].max()
    scores = data[:, 2] / float(max_score)

    rows_idx, cols_idx = mapping_indexes(scores, data[:, 0], data[:, 1])
    pivot_scores = pivot_gen_score_matrix(scores, rows_idx, cols_idx)

    data_name = get_data_name(path)

    f = pd.DataFrame(pivot_scores)
    f.to_parquet(f'results/scores_{data_name}_{n}_run{run_i}.parquet')
    # with open(f'results/scores_{data_name}_{n}.npz', 'wb') as f:
    #     np.savez(f, pivot_scores)
    #     f.close()

    # ajeitar pra mapear os indices corretamente
    # if path == 'data/black_friday.csv':
    #     df = pd.read_csv('data/black_friday_groups.csv', index_col=0)
    #     df = df.loc[indexes_data]
    #     df.index = df['User_ID']
        
    #     df['User_ID'], df['Product_ID'] = mapping_indexes(np.zeros(len(df)), df['User_ID'], df['Product_ID'])
        
        
    #     create_groups(df)

    print("shape scores: ",pivot_scores.shape)
    return pivot_scores

def square_matrix_for_reciprocal(pref_user_item: np.ndarray, pref_item_user: np.ndarray):
    N, I = pref_user_item.shape
    n = N + I
    S = torch.zeros(n,n)
    S[:N, N:] = torch.tensor(pref_user_item)
    S[N:, :N] = torch.tensor(pref_item_user)
    return S


def pref_estimation(data: np.ndarray):
    pref_estimator = NMF()
    print('Estimating scores...')
    U = pref_estimator.fit_transform(data)
    V = pref_estimator.components_
    scores = U.dot(V)

    return scores


def generate_reciprocal_synthetic_data(n_left, m_right, lamb, seed=42):
    torch.manual_seed(seed)
    M = torch.rand((n_left, m_right))
    N = torch.rand((m_right, n_left))

    right_general_pop = torch.tile(torch.linspace(1, 0, m_right), dims=(n_left,1))
    M = torch.clip((1 - lamb) * M + lamb * right_general_pop, 0., 1.)

    left_general_pop = torch.tile(torch.linspace(1, 0, n_left), dims=(m_right,1))
    N = torch.clip((1 - lamb) * N + lamb * left_general_pop, 0., 1.)

    return M, N


def create_groups(df):
    def save_json(output_path, groups_dict):
        with open(output_path, 'w') as f:
            json.dump(groups_dict, f)
        
    #df = pd.read_csv('data/black_friday_groups.csv')
    df.index = df['User_ID']
    df.to_csv('data/temp_file_black_friday_groups.csv')
    gender_groups = df[['User_ID', 'Gender']].groupby('Gender').groups
    gender_groups_idx = list(gender_groups.keys())
    for i in range(len(gender_groups)):
        idx = gender_groups_idx[i]
        gender_groups[idx] = list(gender_groups[idx])
    save_json('data/bf_gender_groups.json', gender_groups)

    age_groups = df[['User_ID', 'Age']].groupby('Age').groups
    age_groups_idx = list(age_groups.keys())
    for i in range(len(age_groups)):
        idx = age_groups_idx[i]
        age_groups[idx] = list(age_groups[idx])
    save_json('data/bf_age_groups.json', age_groups)

    occ_groups = df[['User_ID', 'Occupation']].groupby('Occupation').groups
    occ_groups_idx = list(occ_groups.keys())
    for i in range(len(occ_groups)):
        idx = occ_groups_idx[i]
        occ_groups[idx] = list(occ_groups[idx])
    save_json('data/bf_occ_groups.json', occ_groups)

    df.index = df['Product_ID']
    prod_groups = df[['Product_ID', 'Product_Category_1']].groupby('Product_Category_1').groups
    prod_groups_idx = list(prod_groups.keys())
    for i in range(len(prod_groups)):
        idx = prod_groups_idx[i]
        prod_groups[idx] = list(prod_groups[idx])
    save_json('data/bf_prod_groups.json', prod_groups)