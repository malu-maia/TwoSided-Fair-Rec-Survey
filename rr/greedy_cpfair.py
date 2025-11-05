import torch
import random

import numpy as np
import pandas as pd

from utils import *
from evaluation import *

from itertools import chain

NAME = 'CPFair'

def get_customer_metric(base_topk, scores: torch.Tensor, user_val: int, user_mapping: dict, k: int=10):
    # arg: mapping user_val: grou user
    _, base_topk = torch.topk(scores, k=k)

    protected, not_protected = [], []
    for u, v in user_mapping.items():
        if v == 1:
            protected.append(u)
        else:
            not_protected.append(u)
    
    if user_val == 1:
        base_topk_group = base_topk[protected]
        _, topk_groups = torch.topk(scores[protected], k=k)
    else:    
        base_topk_group = base_topk[not_protected]
        _, topk_groups = torch.topk(scores[not_protected], k=k)
 
    ndcgs = []
    for u in range(len(topk_groups)):
        ndcg = ndcg_score([list(base_topk_group[u].cpu().numpy())], [list(topk_groups[u].cpu().numpy())])
        #ndcg = ndcg_score([list(map(int, base_topk_group[u]))], [list(map(int, topk_groups[u]))])
        ndcgs.append(ndcg)
    
    return np.mean(ndcgs)

def get_producer_metric(scores: torch.Tensor, item_val: int, item_mapping: dict, k: int=10):
    # arg: mapping item_val: item group
    _, topk_idxs = torch.topk(scores, k=k)

    protected, not_protected = [], []
    #exposure_prot, exposure_not_prot = [], []
    ## calcular a exposição como o somatório de 1/(log(k)+2)
    for i, v in item_mapping.items():
        if v == 1:
            protected.append(i)
        else:
            not_protected.append(i)
    ##################
    positions = torch.arange(k, device=topk_idxs.device)
    exposure_weights = 1/(torch.log2(positions+2))

    protected_tensor = torch.tensor(protected, device=topk_idxs.device)
    is_protected_mask = torch.isin(topk_idxs, protected_tensor)

    protected_exposure = (is_protected_mask * exposure_weights).sum()
    unprotected_exposure = (~is_protected_mask * exposure_weights).sum()

    exposures = {
        -1: unprotected_exposure.item(),
        1: protected_exposure.item()
    }

    total_exposure = sum(exposures.values())

    return exposures[item_val]/total_exposure


def cpfair(scores: torch.Tensor,
           user_groups: dict,
           item_groups: dict,
           base_topk: np.array,
           lambda1: float = 0.5,
           lambda2: float = 0.5,
           k: int = 10):
    new_scores = torch.full_like(scores, -float('inf')) # Initialize new scores for all items

    # Iterate through each user to re-score their initial top-N list
    for user_id, initial_rec_items in enumerate(base_topk):
        user_group_val = user_groups.get(user_id, -1) # Default to advantaged if not in dict
        consumer_fairness_metric = get_customer_metric(base_topk, scores, user_group_val, user_groups, k)

        cf_term = user_group_val * consumer_fairness_metric
        for item_id in initial_rec_items:
            original_score = scores[user_id, item_id]
            item_group_val = item_groups.get(item_id, -1)

            producer_fairness_metric = get_producer_metric(scores, item_group_val, item_groups, k)
            
            pf_term = item_group_val * producer_fairness_metric
            # S'_ui = S_ui + lambda1 * CF_ui + lambda2 * PF_ui
            fair_score = original_score + (lambda1 * cf_term) + (lambda2 * pf_term)
            new_scores[user_id, item_id] = fair_score

        if (user_id+1)%50 == 0:
            print(f'Computing new scores for user {user_id}/{len(base_topk)}')

    _, fair_topk = torch.topk(new_scores, k)

    return new_scores, fair_topk

def get_active_users(data_path, columns_df: list[str]=['user_id', 'item_id', 'rating', 'timestamp']):
    # get temp_indexes_data that stores the sampled rows to get the most active users
    data = pd.read_csv(data_path, sep="\t", names=columns_df, index_col=False)
    data = data[data.columns[:2]]

    with open(f"data/temp_data/temp_indexes_{data_path.split('/')[-1].split('.')[0]}.npz", 'rb') as f:
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

def get_top_items(data_path, columns_df: list[str]=['user_id', 'item_id', 'rating', 'timestamp']):
    # get temp_indexes_data that stores the sampled rows to get the most interacted items
    data = pd.read_csv(data_path, sep="\t", names=columns_df, index_col=False)
    data = data[data.columns[:2]]

    with open(f"data/temp_data/temp_indexes_{data_path.split('/')[-1].split('.')[0]}.npz", 'rb') as f:
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


def teste(
    data_path: str='data/movielens_100k_u1.base', 
    n: int=100, k: int=10, protected_item_group:int=2, 
    protected_user_group: int=0, columns_df: list[str]=['user_id', 'item_id', 'rating', 'timestamp']):

    data = load_data(data_path, n=n)
    scores = torch.tensor(pref_estimation(data))


    user_groups = get_active_users(data_path, columns_df)
    item_groups = get_top_items(data_path, columns_df)
    
    _, base_top_k = torch.topk(scores, k=k)

    fair_scores, fair_topk = cpfair(scores,
                                     user_groups=user_groups,
                                     item_groups=item_groups,
                                     base_topk=base_top_k.numpy(),
                                     lambda1=0.5,
                                     lambda2=0.5,
                                     k=k)

    return scores, fair_scores, fair_topk