import pandas as pd
import numpy as np
import torch
import os
import csv
import math

from scipy.sparse import coo_matrix
from sklearn.metrics import ndcg_score
from sklearn.decomposition import NMF

from utils import *
from evaluation import *

NAME = 'TFROM'

def tfrom_uniform_pytorch(scores: torch.Tensor, k: int, item_to_provider: torch.Tensor):
    device = scores.device
    m_users, n_items = scores.shape
    unique_providers = torch.unique(item_to_provider)
    n_providers = len(unique_providers)
    provider_map = {p.item(): i for i, p in enumerate(unique_providers)}
    mapped_item_to_provider = torch.tensor([provider_map[p.item()] for p in item_to_provider], device=device)
    
    discounts = 1 / torch.log2(torch.arange(2, k + 2, device=device).float())
    
    total_exposure = m_users * torch.sum(discounts)
    provider_item_counts = torch.bincount(mapped_item_to_provider, minlength=n_providers).float()
    fair_exposure = total_exposure * (provider_item_counts / n_items)
    l_ori = torch.argsort(scores, dim=1, descending=True)
    
    recommendations = torch.full((m_users, k), -1, device=device) # -1 indica nenhuma recomendação
    provider_exposure = torch.zeros_like(fair_exposure, device=device)
    customer_satisfaction = torch.zeros(m_users, device=device)

    # Máscara booleana para itens disponíveis
    available_items_mask = torch.ones((m_users, n_items), dtype=torch.bool, device=device)

    for rank in range(k):
        if rank == 0:
            # Ordem aleatória para o primeiro rank
            user_order = torch.randperm(m_users, device=device)
        else:
            # Ordena usuários pela satisfação acumulada
            user_order = torch.argsort(customer_satisfaction, descending=True)
        
        current_rank_discount = discounts[rank]

        # Loop sobre os usuários na ordem definida
        for user_idx in user_order:
            user_scores = scores[user_idx].clone()
            user_scores[~available_items_mask[user_idx]] = -torch.inf # Invalida itens já usados

            # Restrição de 'fair_exposure' para tds os itens de uma vez
            current_providers_exposure = provider_exposure[mapped_item_to_provider]
            
            # Cria uma máscara de itens que satisfazem a restrição de exposição
            exposure_constraint_mask = (current_providers_exposure + current_rank_discount <= fair_exposure[mapped_item_to_provider])
            
            # Invalida os scores dos itens que não cumprem a restrição
            user_scores[~exposure_constraint_mask] = -torch.inf
            
            best_score, best_item_idx = torch.max(user_scores, dim=0)

            if best_score > -torch.inf:
                provider_idx = mapped_item_to_provider[best_item_idx]
                
                # Atualiza as estruturas de dados
                recommendations[user_idx, rank] = best_item_idx
                provider_exposure[provider_idx] += current_rank_discount
                customer_satisfaction[user_idx] += scores[user_idx, best_item_idx] * current_rank_discount
                
                # Marca o item como indisponível para este usuário
                available_items_mask[user_idx, best_item_idx] = False
        if (rank*len(user_order))%1000 == 0:
            print(f'Filling positions: {rank*len(user_order)}/{k*len(user_order)}')

    # fill any remaining empty slots
    for user_idx in range(m_users):
        for rank in range(k):
            if recommendations[user_idx, rank] == -1:
                # get all items that were not yet recommended to this user
                available_indices = available_items_mask[user_idx].nonzero(as_tuple=True)[0]
                
                if len(available_indices) == 0:
                    continue 

                # Find the item from the provider with the minimum current exposure 
                available_providers = mapped_item_to_provider[available_indices]
                exposures_of_available = provider_exposure[available_providers]
                
                # In case of ties in exposure, argmin picks the first one.
                # A more advanced tie-breaker could consider item scores.
                min_exposure_item_in_available = torch.argmin(exposures_of_available)
                item_to_add = available_indices[min_exposure_item_in_available]
                provider_idx = mapped_item_to_provider[item_to_add]

                # Assign item and update metrics
                recommendations[user_idx, rank] = item_to_add
                provider_exposure[provider_idx] += discounts[rank]
                available_items_mask[user_idx, item_to_add] = False
                
    return recommendations


def calculate_ndcg_sklearn(true_scores: np.ndarray, recommendations: np.ndarray, k: int) -> float:
    """Calcula o NDCG usando a função da biblioteca Scikit-learn."""
    y_pred = np.zeros_like(true_scores)
    for user_idx, rec_list in enumerate(recommendations):
        for rank, item_idx in enumerate(rec_list):
            if item_idx != -1:
                y_pred[user_idx, int(item_idx)] = k - rank
    return ndcg_score(true_scores, y_pred, k=k)


def define_random_providers(n_items, min_size=1, max_size=50):
    """
    Simulates providers of different sizes as described in the paper.
    Each item is assigned to a provider.
    """
    item_to_provider_map = torch.zeros(n_items, dtype=torch.long)
    current_item_idx = 0
    provider_id = 0
    while current_item_idx < n_items:
        # Determine size of the current provider
        size = torch.randint(min_size, max_size + 1, (1,)).item()
        end_idx = min(current_item_idx + size, n_items)
        
        # Assign items to this provider
        item_to_provider_map[current_item_idx:end_idx] = provider_id
        
        provider_id += 1
        current_item_idx = end_idx
        
    return item_to_provider_map

def tfrom(scores: torch.Tensor, k: int=10):
    top_scores, top_indexes = torch.topk(scores, k=k)

    item_provider_mapping = define_random_providers(n_items=scores.size(1), max_size=int(scores.size(1)*0.3))
    fair_recs = tfrom_uniform_pytorch(scores=scores, k=k, item_to_provider=item_provider_mapping)
    
    return scores, fair_recs