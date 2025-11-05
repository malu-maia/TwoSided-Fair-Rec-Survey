from sklearn.metrics import ndcg_score
from scipy.special import softmax
from utils import *

import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch

import math
import os

def sort_utility_profiles(u_p: torch.Tensor):
    return torch.sort(u_p, descending=False)

def cumulative_u_Lorenz(sorted_u_p: torch.Tensor):
    return torch.cumsum(sorted_u_p, dim=0)

def lorenz_curve(utility_profiles, approach: str, dataset: str, n_interactions: int, type='u'):
    """
    Calculates and plots the Lorenz curve.

    Args:
      data: A 1D PyTorch tensor of income/wealth values.
    """
    #utility_profiles.to('cpu')

    utility_profiles = utility_profiles.to_numpy()
    #sorted_profiles, _ = sort_utility_profiles(utility_profiles)
    sorted_profiles = sorted(utility_profiles)

    #cumulative_utility = cumulative_u_Lorenz(sorted_profiles)
    cumulative_utility = np.cumsum(sorted_profiles)

    #population = torch.arange(1, len(utility_profiles) + 1)
    population = np.arange(1, len(utility_profiles) + 1)

    #plt.plot(population.cpu().detach().numpy(), cumulative_utility.cpu().detach().numpy(), label='Lorenz Curve')
    plt.plot(population, cumulative_utility, label='Lorenz Curve')

    # Plot the line of equality
    #plt.plot([0, len(utility_profiles.cpu())], [0, utility_profiles.cpu().sum()], linestyle='--', color='gray', label='Line of Equality')
    plt.plot([0, len(utility_profiles)], [0, utility_profiles.sum()], linestyle='--', color='gray', label='Line of Equality')

    plt.xlabel('Cumulative Population')
    plt.ylabel('Cumulative Utility')
    plt.title('Lorenz Curve')
    plt.legend()
    plt.grid(True)

    base_path = f'results_plots/lorenz_curves/{approach}'
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    if type == 'u':
        plt.savefig(f'{base_path}/utility_profiles_users_{dataset}_{n_interactions}.png')
        df_cum = pd.DataFrame(cumulative_utility).to_csv(f'results_plots/lorenz_curves/utilities_csv/utility_profiles_users_{approach}_{dataset}_{n_interactions}.csv')
    else:
        plt.savefig(f'{base_path}/utility_profiles_items_{dataset}_{n_interactions}.png')
        df_cum = pd.DataFrame(cumulative_utility).to_csv(f'results_plots/lorenz_curves/utilities_csv/utility_profiles_items_{approach}_{dataset}_{n_interactions}.csv')
    plt.show()
    plt.close()

def envy_freeness(pred_scores, top_k):
    return

def NDCG(scores, pred_topk, k: int):
    ndcgs = []

    scores = torch.tensor(scores).detach().clone()
    pred_topk = torch.tensor(pred_topk).detach().clone()

    sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=1)
    true_topk = sorted_indices[:, :k]

    fair_scores, fair_top_k = None, pred_topk[:, :k]

    for i in range(scores.size(0)):
        ndcg = ndcg_score([true_topk[i]], [fair_top_k[i]])
        ndcgs.append(ndcg)

    return ndcgs


def var_NDCG(scores, pred_scores, k):
    ndcgs = NDCG(scores, pred_scores, k)
    return np.var(ndcgs)

def mean_NDCG(scores, pred_scores, k):
    ndcgs = NDCG(scores, pred_scores, k)
    return np.mean(ndcgs)

def compute_exposure(scores, fair_topk, k:int, groups=None, equal_weight=False):
    scores = torch.tensor(scores).detach().clone()
    fair_topk = torch.tensor(fair_topk).detach().clone()

    fair_topk = fair_topk[:, :k]

    all_items = torch.arange(scores.size(1))

    positions = torch.arange(k)

    exposures = {int(item): 0 for item in all_items}

    for rec_user in fair_topk:
        for k in range(len(rec_user)):
            item_id = int(rec_user[k])
            if equal_weight:
                exposures[item_id] += 1
            else:
                exposures[item_id] += 1/(np.log2(k+2))

    return exposures


def var_exposure(scores, fair_topk, k, equal_weight=False):
    exposures = compute_exposure(scores, fair_topk, k=k, equal_weight=equal_weight)
    return np.var(list(exposures.values()))


def gini_coefficient(scores, fair_topk, k):
    # The lower the fairer. Domain [0, 1].
    scores = scores.to_numpy()

    n_items = scores.shape[1]

    exposures = compute_exposure(scores, fair_topk, k)
    
    sorted_exposures = dict(sorted(exposures.items(), key=lambda item: item[1]))
    gini = 0

    for j in range(n_items):
        jth_exp = list(sorted_exposures.values())[j]
        gini += ((2*j - n_items - 1) * jth_exp)

    return gini/(n_items * sum(list(exposures.values())))


def entropy(top_k: np.ndarray, k: int):
    # The higher the value, the more uniform the results are.
    # We put the log base as the number of unique items saw in list, so that the entropy is in [0,1]
    n_users = top_k.shape[0]
    top_k = top_k.flatten()
    item, item_frequency = np.unique(top_k, return_counts=True)
    item_frequency = item_frequency/(k*n_users)
    frequencies = {int(i): float(f) for i, f in zip(item, item_frequency)}
    ent = 0
    for item in frequencies.keys():
        p_i = frequencies[item]
        ent -= p_i*math.log(p_i, len(frequencies))

    return ent


def maximin_producer(topk, num_items, k: int):
    # The higher the value, the fairer the result. Domain: [0,1]
    """
    Parameters: 
    topk dict or array that contains the items with higher score for each user.
    num_items: the number of total items

    Returns:
    the MMS value for the producers which is the (num_users*k)//num_items
    """
    num_users = len(topk)

    concated = np.concat(list(topk))
    _, counts = np.unique(concated, return_counts=True)

    l = (num_users*k)//num_items

    return sum([1 for c in range(len(counts)) if counts[c] >= l])/num_items

    

def ii_f(scores, pred_scores):
    scores = torch.tensor(scores)
    pred_scores = torch.tensor(pred_scores)

    scores = scores/torch.max(scores)
    pred_scores = pred_scores/torch.max(pred_scores)

    system_exposure = pred_scores
    target_exposure = (torch.ones_like(system_exposure) * scores.mean())

    delta_E = system_exposure - target_exposure

    assert torch.all(delta_E.isnan().logical_not()), f"delta_E contains NaN values: {delta_E}"

    return float(torch.mean(delta_E * delta_E).cpu().numpy())


def gg_f(scores, pred_scores, user_groups, item_groups, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # read json groups: pd.read_json('results/user_groups_{dataset}_{n}_run{run_i}.json', orient='index')
    # user_groups = user_groups.to_dict() ...

    system_exposure = softmax(pred_scores, axis=1)  # E_ij
    target_scores = np.ones_like(scores) * np.array(scores).mean()
    target_exposure = softmax(target_scores, axis=1)      # E*_ij

    exposure_deviation = system_exposure - target_exposure # E_ij - E*_ij

    user_group_indices = {}
    for user_idx, group_label in user_groups[0].items():
        user_group_indices.setdefault(group_label, []).append(user_idx)

    item_group_indices = {}
    for item_idx, group_label in item_groups[0].items():
        item_group_indices.setdefault(group_label, []).append(item_idx)

    total_squared_group_deviation = 0.0

    unique_user_groups = list(user_group_indices.keys())
    unique_item_groups = list(item_group_indices.keys())

    for u_group in unique_user_groups:
        for i_group in unique_item_groups:
            # Get the row and column indices for the current group combination.
            u_indices = user_group_indices[u_group]
            i_indices = item_group_indices[i_group]
            sub_matrix = exposure_deviation[np.ix_(u_indices, i_indices)]
            
            if sub_matrix.size > 0:
                mean_group_deviation = np.mean(sub_matrix)
            else:
                mean_group_deviation = 0.0
                
            total_squared_group_deviation += mean_group_deviation ** 2

    num_user_groups = len(unique_user_groups)
    num_item_groups = len(unique_item_groups)
    
    if num_user_groups == 0 or num_item_groups == 0:
        return 0.0
        
    ggf = total_squared_group_deviation / (num_user_groups * num_item_groups)

    return ggf