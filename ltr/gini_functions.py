import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim # Still imported, but not used for model optimization, only for torch utilities
import torch.nn.functional as F
import matplotlib.pyplot as plt

from enum import Enum
# from sklearn.isotonic import isotonic_regression # Not directly used for PAV, custom PAV is implemented
from sklearn.metrics import mean_squared_error
from utils import *
from scipy.optimize import minimize, isotonic_regression

import warnings
warnings.filterwarnings("ignore")

NAME='Gini'

# %%
def get_u(mu: torch.Tensor, P_stochastic: torch.Tensor, v_weights: torch.Tensor) -> torch.Tensor:
    """
    Calculates the utility vector for all users.
    Formula: u_i(P) = sum_{j,k} mu_ij * P_ijk * v_k

    Parameters:
    - mu: Relevance scores tensor, shape (N, M)
    - P_stochastic: Stochastic ranking tensor, shape (N, M, K)
    - v_weights: Exposure weight vector, shape (K,)

    Returns:
    - Tensor of user utilities, shape (N,)
    """
    v_broadcastable = v_weights.view(1, 1, -1)
    expected_exposure = (P_stochastic * v_broadcastable).sum(dim=2) # Shape: (N, M)
    user_utilities = (mu * expected_exposure).sum(dim=1) # Shape: (N,)
    return user_utilities

def get_v(P_stochastic: torch.Tensor, v_weights: torch.Tensor) -> torch.Tensor:
    """
    Calculates the utility (exposure) vector for all items.
    Formula: u_j(P) = sum_{i,k} P_ijk * v_k

    Parameters:
    - P_stochastic: Stochastic ranking tensor, shape (N, M, K)
    - v_weights: Exposure weight vector, shape (K,)

    Returns:
    - Tensor of item utilities, shape (M,)
    """
    v_broadcastable = v_weights.view(1, 1, -1)
    expected_exposure = (P_stochastic * v_broadcastable).sum(dim=2) # Shape: (N, M)
    item_utilities = expected_exposure.sum(dim=0) # Shape: (M,)
    return item_utilities

def get_b(k: int) -> torch.Tensor:
    """
    Returns the exposure weight vector
    """
    return torch.tensor([1.0 / np.log2(i + 2) for i in range(k)], dtype=torch.float32)


def get_P(S: torch.Tensor, k: int) -> torch.Tensor:
    """
    Generate a policy tensor P based on scores S, and recommendation size k.
    For each user, it identifies the top-k items based on scores S
    and creates a one-hot like encoding for the policy matrix.
    """
    n, m = S.size()
    _, topk_indices = torch.topk(S, k, dim=1, largest=True, sorted=True)
    P = F.one_hot(topk_indices, num_classes=m).to(torch.float32)
            
    return P


# %%
def get_w(n: int):
    """
    Returns the Gini weights w_i = (n - i + 1) / n, sorted in decreasing order.
    These weights give more importance to lower-ranked (worse-off) individuals.
    """
    # Equivalent to (n, n-1, ..., 1) / n
    return torch.arange(n, 0, -1, dtype=torch.float32) / n

# def build_gini_functions(x: torch.Tensor, w: torch.Tensor): 
#     """
#     Calculates the Generalized Gini Welfare Function (GGF) for a given vector x.
#     g_w(x) = sum(w_i * x_i_sorted_increasing)
#     This function is primarily for understanding the GGF definition,
#     the Moreau envelope approach avoids direct use of this for gradients.
#     """   
#     n = x.shape[0]
#     x = torch.from_numpy(x)
#     w = get_w(n) # w is already in decreasing order (w_1 >= ... >= w_n)
#     sorted_x, _ = x.sort(descending=False) # x_i_dagger is sorted in increasing order

#     return (w * sorted_x).sum()


# def moreau_envelope(z: torch.Tensor, beta: torch.float32, w: torch.Tensor):
#     def objective(z_tilde: torch.Tensor):
#         return -build_gini_functions(z_tilde, w) + (1/(2*beta)) * torch.linalg.norm((z-z_tilde))
#     result = minimize(objective, x0=z)
#     #print(f'result: {result.x}\ninitial z: {z}')
#     return result.fun

def pav(z: torch.Tensor, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> torch.Tensor:
    isotonic_func = isotonic_regression(z.cpu())
    isotonic_z = torch.from_numpy(isotonic_func.x).to(device)
    return isotonic_z

def projection(z: torch.Tensor, w: torch.Tensor, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> torch.Tensor:
    """
    The projection onto the permutahedron is equivalent to the gradient of the Moreau envelope.
    """
    flipped_w = w.view(1, -1).flip([0,1]).squeeze(0).to(device)
    w_tilde = -flipped_w

    z_sorted, idx_sorted = torch.sort(z, descending=True)
    x = pav(z_sorted - w_tilde)
    inverse_sort = torch.argsort(idx_sorted)
    y = z + x[inverse_sort]

    return y

def loss_function(u: torch.Tensor, v: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, lamb: torch.float, beta: float=.1):
    u_sorted_ascending, _ = torch.sort(u, descending=False)
    g_u = (w1 * u_sorted_ascending).sum()

    # Calculate GGF for items, g_w2(v)
    v_sorted_ascending, _ = torch.sort(v, descending=False)
    g_v = (w2 * v_sorted_ascending).sum()

    # The full objective function F(P) to be MAXIMIZED
    F_P = (1 - lamb) * g_u + lamb * g_v

    # Optimizers perform minimization, so the loss is the negative of the objective.
    return -F_P

def train(scores: torch.Tensor, beta: float=1000, epochs: int=5000, lamb: float=.5, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    n, m = scores.size()
    k = m

    beta = torch.tensor(beta).to(device)

    _, sorted_indices = torch.sort(scores, descending=True, dim=1)
    top_k_indices = sorted_indices[:, :k]
    P_stochastic = F.one_hot(top_k_indices, num_classes=m).to(torch.float32).to(device)
    
    w1 = torch.ones(n).to(device)
    w2 = get_w(m).to(device)

    b = get_b(k).to(device)

    losses = []

    for t in range(1, epochs+1):
        beta_t = beta / np.sqrt(t)
        u = get_u(scores, P_stochastic, b).to(device)
        v = get_v(P_stochastic, b).to(device)
        y1 = projection(u/beta_t, w1).to(device)
        y2 = projection(v/beta_t, w2).to(device)

        #scores_tilde = (1 - lamb) * torch.einsum('i,ij->ij', y1, scores) + lamb * y2
        scores_tilde = (1 - lamb) * y1.view(-1, 1) * scores + lamb * y2.view(1, -1)

        _, sorted_indices = torch.sort(-scores_tilde, descending=True, dim=1)
        top_k_indices = sorted_indices[:, :k]
    
        Q_t = F.one_hot(top_k_indices, num_classes=m).to(torch.float32).to(device)
        P_stochastic = (1 - (2/(t+2))) * P_stochastic + (2/(t+2)) * Q_t

        loss = loss_function(u, v, w1, w2, lamb)
        losses.append(-loss)

        if (t+1)%100==0:
            print(f'Epoch {t+1}/{epochs}: Loss {-loss}')

    return scores_tilde, P_stochastic, u, v


# %%
# functions to compute lorenz curves
def sort_utility_profiles(u_p: torch.Tensor):
    return torch.sort(u_p, descending=False)

def cumulative_u_Lorenz(sorted_u_p: torch.Tensor):
    return torch.cumsum(sorted_u_p, dim=0)

def lorenz_curve(utility_profiles, type='u'):
    """
    Calculates and plots the Lorenz curve.

    Args:
      data: A 1D PyTorch tensor of income/wealth values.
    """
    #utility_profiles.to('cpu')

    sorted_profiles, _ = sort_utility_profiles(utility_profiles)
    cumulative_utility = cumulative_u_Lorenz(sorted_profiles)

    population = torch.arange(1, len(utility_profiles) + 1)
    plt.plot(population.cpu().detach().numpy(), cumulative_utility.cpu().detach().numpy(), label='Lorenz Curve')
    plt.plot([0, len(utility_profiles.cpu())], [0, utility_profiles.cpu().sum()], linestyle='--', color='gray', label='Line of Equality')

    plt.xlabel('Cumulative Population')
    plt.ylabel('Cumulative Utility')
    plt.title('Lorenz Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    if type == 'u':
        plt.savefig('lorenz_curves/Gini/utility_profiles_users.png')
    else:
        plt.savefig('lorenz_curves/Gini/utility_profiles_items.png')
    plt.close()


def teste(data, n=100, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Main function to run the training process with dummy data.
    """
    print(f"Generating dummy score data with {data} users...")
    #scores = load_data('data/movielens_100k_u1.base', n=n) 
    scores = torch.from_numpy(pref_estimation(data))

    n_users, n_items = scores.size()
    print(f"Score matrix shape: {n_users} users, {n_items} items")

    print("Starting Frank-Wolfe optimization...")
    pred, final_P, u, v = train(scores.to(device), beta=100., epochs=500)

    return scores, pred, final_P, u, v