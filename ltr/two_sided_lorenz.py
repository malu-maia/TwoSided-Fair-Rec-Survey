import torch
import torch.nn.functional as F
import numpy as np
from utils import *

import matplotlib.pyplot as plt

NAME = 'Lorenz'

def get_u(mu: torch.Tensor, P_stochastic: torch.Tensor, v_weights: torch.Tensor, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> torch.Tensor:
    """
    Calculates the utilitys vector for all users.
    Formula: u_i(P) = sum_{j,k} mu_ij * P_ijk * v_k
    """
    v_broadcastable = v_weights.view(1, 1, -1).to(device)
    expected_exposure = (P_stochastic * v_broadcastable).sum(dim=2) # Shape: (N, M)
    user_utilities = (mu * expected_exposure).sum(dim=1) # Shape: (N,)
    return user_utilities

def get_v(P_stochastic: torch.Tensor, v_weights: torch.Tensor, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> torch.Tensor:
    """
    Calculates the utility (exposure) vector for all items.
    Formula: u_j(P) = sum_{i,k} P_ijk * v_k
    """
    v_broadcastable = v_weights.view(1, 1, -1).to(device)

    # Expected exposure for each item for each user
    expected_exposure = (P_stochastic * v_broadcastable).sum(dim=2) # Shape: (N, M)

    # Item utility: sum over all users
    item_utilities = expected_exposure.sum(dim=0) # Shape: (M,)
    return item_utilities

def get_b(k: int) -> torch.Tensor:
    weights = torch.tensor([1 / np.log2(i + 2) for i in range(k)], dtype=torch.float32)
    return weights

def psi(x: torch.Tensor, alpha: float, eta: float = 1e-9) -> torch.Tensor:
    """
    Implements the psi function from the paper's welfare definition.
    A small eta is added to prevent log(0) or division by zero.
    """
    x_safe = x + eta
    if alpha == 0:
        return torch.log(x_safe)
    elif alpha > 0:
        return torch.pow(x_safe, alpha)
    else: # alpha < 0
        return -torch.pow(x_safe, alpha)
    

def psi_prime(x: torch.Tensor, alpha: float, eta: float = 1e-9) -> torch.Tensor:
    """
    Implements the derivative of the psi function.
    """
    x_safe = x + eta
    if alpha == 0:
        return 1 / x_safe
    elif alpha > 0:
        return alpha * torch.pow(x_safe, alpha - 1)
    else: # alpha < 0
        return -alpha * torch.pow(x_safe, alpha - 1)
    

def phi(x, alpha, eta):
    return (1/2)*psi(x, alpha, eta)

def phi_prime(x, alpha, eta):
    return 1/2 * (psi_prime(x, alpha, eta))


def compute_welfare(user_utils: torch.Tensor, item_utils: torch.Tensor, lamb: float, alpha_1: float, alpha_2: float, eta: float) -> torch.Tensor:
    """
    Calculates the total welfare W_theta(u).
    """
    user_welfare = (1 - lamb) * psi(user_utils, alpha_1, eta).sum()
    item_welfare = lamb * psi(item_utils, alpha_2, eta).sum()
    return user_welfare + item_welfare


# %%
# functions to compute lorenz curves
def sort_utility_profiles(u_p: torch.Tensor):
    return torch.sort(u_p, descending=False)

def cumulative_u_Lorenz(sorted_u_p: torch.Tensor):
    return torch.cumsum(sorted_u_p, dim=0)

def lorenz_curve(utility_profiles, type='u'):
    """
    Calculates and plots the Lorenz curve.
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
        plt.savefig('lorenz_curves/Lorenz/utility_profiles_users.png')
    else:
        plt.savefig('lorenz_curves/Lorenz/utility_profiles_items.png')
    plt.close()

# %%
def train(scores: np.ndarray, epochs: int = 1000, k: int = 10, alpha: list=[0.,0.], lamb: float=0.5, eta: float = 1e-9, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
  alpha_1, alpha_2 = alpha
  N, M = scores.size()
  S = scores
  k = M
  # Exposure weights for ranks
  v_exposure_weights = get_b(k).to(device)

  print(f'Running with: {N} users, {M} items, Top-{k} recommendations')
  print(f'Parameters: lambda={lamb}, alpha=[{alpha_1}, {alpha_2}], epochs={epochs}')
  print("-" * 30)

  # Start with a deterministic ranking where each user gets the items they have the highest score for.
  _, sorted_indices = torch.sort(S, descending=True, dim=1)
  top_k_indices = sorted_indices[:, :k]
  # P_stochastic is the probability tensor. Initially, it's deterministic (0 or 1).
  P_stochastic = F.one_hot(top_k_indices, num_classes=M).to(torch.float32).to(device)

  welfare_store_vector = []

  # Frank-Wolfe
  for t in range(epochs):
      # Current user and item utilities
      user_utilities = get_u(S, P_stochastic, v_exposure_weights).to(device)
      item_utilities = get_v(P_stochastic, v_exposure_weights).to(device)

      # Gradient components
      phi_prime_u = phi_prime(user_utilities, alpha_1, eta) # Shape: (N,)
      phi_prime_i = phi_prime(item_utilities, alpha_2, eta) # Shape: (M,)

      # Computing the score matrix A for the linear subproblem.
      # A_ij = (1-lambda)*psi'(u_i)*mu_ij + lambda*psi'(u_j)
      term_user = (1 - lamb) * phi_prime_u.unsqueeze(1) * S # Broadcasts to (N, M)
      term_item = lamb * phi_prime_i.unsqueeze(0)           # Broadcasts to (1, M)
 
      A = term_user + term_item # Broadcasts to (N, M) + (N, M)

      # Finding the next ranking direction P_tilde by sorting based on scores A
      _, sorted_indices_tilde = torch.sort(A, descending=True, dim=1)
      top_k_indices_tilde = sorted_indices_tilde[:, :k]
      P_tilde = F.one_hot(top_k_indices_tilde, num_classes=M).to(torch.float32).to(device)

      # Updating the stochastic ranking tensor P
      gamma = 2 / (t + 2) # Step size
      P_stochastic = (1 - gamma) * P_stochastic + gamma * P_tilde

      if (t + 1) % 100 == 0:
        total_welfare = compute_welfare(user_utilities, item_utilities, lamb=lamb, alpha_1=alpha_1, alpha_2=alpha_2, eta=eta)
        welfare_store_vector.append(total_welfare)
        print(f'Epoch {t+1}/{epochs}: Welfare = {total_welfare.item():.4f}')

  print("-" * 30)
  print("Optimization finished.")

  return A, P_stochastic, user_utilities, item_utilities

def teste(data='data/movielens_100k_u1.base', n=None, alpha: list=[0.5, 0.5], lamb: float=0.5, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    scores = pref_estimation(data)
    scores = torch.from_numpy(scores)
    print('Training...')
    pred, P, welfare, u, v = train(scores.to(device), alpha=alpha, lamb=lamb)
    return scores, pred, P, welfare, u, v