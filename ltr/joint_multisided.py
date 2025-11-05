import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from torch.utils.data import DataLoader, TensorDataset
from torch_scatter import scatter_mean
from utils import *

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.autograd.set_detect_anomaly(True)
NAME = 'JME'

class MatrixFactorization(nn.Module):
    """
    A simple Matrix Factorization model.
    """
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # Initialize embeddings with Xavier initialization as mentioned in the paper
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)

    def forward(self, user_indices, item_indices):
        """
        Computes relevance scores for given user-item pairs.
        """
        user_embs = self.user_embeddings(user_indices)
        item_embs = self.item_embeddings(item_indices)

        # In a training loop, user_indices might be (batch_size) and
        # item_indices might be (num_all_items).
        # We need to compute the score for each user with all items.
        return torch.matmul(user_embs, item_embs.t())


class JMEFairnessLoss(nn.Module):
    """
    Joint Multisided Exposure (JME) Fairness loss.
    Loss = II-F + alpha * GG-F
    """
    def __init__(self,
                 alpha=1.0,
                 gamma=0.8,
                 tau=0.1,
                 num_samples=10,
                 top_k=100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.num_samples = num_samples
        self.top_k = top_k

    def _calculate_exposure(self, scores, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        Calculates the expected exposure matrix E based on relevance scores.
        """
        #device = scores.device
        batch_size, k = scores.shape

        total_exposure = torch.zeros_like(scores, device=device)

        for _ in range(self.num_samples):
            # Add Gumbel noise to make sampling differentiable
            noisy_scores = scores + torch.distributions.Gumbel(0, 1).sample(scores.shape).to(device)

            assert noisy_scores.isnan().sum() == 0, "noisy scores contains NaN values"
            assert noisy_scores.isinf().sum() == 0, "noisy scores contains Inf values"
            # Compute smooth rank for each item
            diffs = noisy_scores.unsqueeze(2) - noisy_scores.unsqueeze(1) # [B, k, k]

            assert diffs.isnan().sum() == 0, f"diffs  contains NaN values: Noise Size: {noisy_scores.size()}"
            # This computes how many items have a score greater than item 'j'
            ranks = 1 + torch.sigmoid(diffs / self.tau).sum(dim=2) - torch.sigmoid(torch.tensor(0.))

            # Compute exposure using the RBP user model
            exposure_sample = torch.pow(self.gamma, ranks - 1)
            total_exposure += exposure_sample

        return total_exposure / self.num_samples

    def forward(self, pred_scores, true_scores, user_groups, item_groups, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        Computes the final loss value.
        """
        #device = pred_scores.device
        batch_size = pred_scores.shape[0]
        num_total_items = pred_scores.shape[1]

        # Get top_k items based on prediction scores
        _, topk_indices = torch.topk(pred_scores, self.top_k, dim=1) # [B, k]

        # Gather the scores and group info for these top_k items
        pred_scores_topk = torch.gather(pred_scores, 1, topk_indices).to(device) # [B, k]
        true_scores_topk = torch.gather(true_scores, 1, topk_indices).to(device) # [B, k]
        item_groups_topk = item_groups[topk_indices] # [B, k]

        system_exposure = self._calculate_exposure(pred_scores_topk).to(device)
        target_exposure = (torch.ones_like(system_exposure) * true_scores.mean()).to(device)

        delta_E = system_exposure - target_exposure

        assert torch.all(delta_E.isnan().logical_not()), f"delta_E contains NaN values: {delta_E}"

        ii_f_loss = torch.mean(delta_E * delta_E)

        delta_E_flat = delta_E.flatten() # [B*k]

        # Create corresponding group indices for the flattened tensor
        user_indices_flat = torch.arange(batch_size, device=device).repeat_interleave(self.top_k)
        user_groups_flat = user_groups[user_indices_flat] # [B*k]
        item_groups_flat = item_groups_topk.flatten() # [B*k]

        # Combine user and item group IDs to create a unique ID for each (user_group, item_group) pair
        num_item_groups = int(item_groups.max()) + 1
        pair_ids_flat = user_groups_flat * num_item_groups + item_groups_flat

        avg_delta_E_per_pair = scatter_mean(delta_E_flat, pair_ids_flat)
        gg_f_loss = torch.mean(torch.pow(avg_delta_E_per_pair, 2))

        total_loss = ii_f_loss + self.alpha * gg_f_loss

        return total_loss, ii_f_loss, gg_f_loss


def teste(k=10):
    n_users, n_items = 50, 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    true_interactions = torch.zeros(n_users, n_items, device=device)

    for u in range(n_users):
        # Each user has rated ~30 random items
        rated_items = torch.randint(0, n_items, (30,))
        true_interactions[u, rated_items] = 1.0

    ### testing groups 
    n_user_groups = 2
    n_item_groups = 5

    user_groups = torch.randint(0, n_user_groups, (n_users,), device=device)
    item_groups = torch.randint(0, n_item_groups, (n_items,), device=device)

    print(item_groups)

    losses, pred_scores = train(true_interactions, user_groups, item_groups, alpha=1.)
    _, top_k = pred_scores.topk(k=k)

    return losses, pred_scores, top_k

def jme(data_path: str, user_groups: torch.Tensor, run_i: int, item_groups: torch.Tensor,
        emb_size: int=32, batch_size: int=100, epochs: int=200, lr: float=1e-2,
        alpha: float=.0, gamma: float=.8, tau: float=0.1, k: int=100, n_samples: int=100, 
        device: str=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    #### loading data
    try:
        df = pd.read_csv(data_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    except:
        df = pd.read_csv(data_path, sep='\t', names=['user_id', 'item_id', 'rating'])

    #### preparing data
    df = df[['user_id', 'item_id', 'rating']]

    user_groups = torch.tensor(list(user_groups.values())).to(device)
    item_groups = torch.tensor(list(item_groups.values())).to(device)
    
    indexes_sample = np.load(f'data/temp_data/temp_indexes_{data_path.split('/')[-1].split('.')[0]}_run{run_i}.npz')
    data = df.loc[indexes_sample]

    rows_idx, cols_idx = mapping_indexes(np.zeros(len(data)), data.to_numpy()[:, 0], data.to_numpy()[:, 1])
    data.iloc[:,0], data.iloc[:,1] = rows_idx, cols_idx

    true_interactions = torch.tensor(pivot_gen_score_matrix(data['rating'], data['user_id'], data['item_id']))
    
    # normalizing between 0-1
    true_interactions = true_interactions/torch.max(true_interactions)
    true_interactions = true_interactions.to(device)
    #print(true_interactions.device, user_groups.device, item_groups.device)
    _, pred_scores = train(true_interactions, user_groups, item_groups, k=k, epochs=epochs)
    print(pred_scores.size(), k)
    _, fair_topk = pred_scores.topk(k=k)

    fair_topk = fair_topk.to(device)

    # normalizing pred_scores between 0-1
    pred_scores = (pred_scores - torch.min(pred_scores))/(torch.max(pred_scores) - torch.min(pred_scores))

    return pred_scores, fair_topk



# train while updating the matrix factorization embeddings
def train(true_interactions: torch.Tensor, user_groups: torch.Tensor, item_groups: torch.Tensor,
             emb_size: int=32, batch_size: int=100, epochs: int=200, lr: float=1e-2,
             alpha: float=.0, gamma: float=.8, tau: float=0.1, k: int=100, n_samples: int=100, 
             device: str=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    n_users, n_items = true_interactions.size()
    n_user_groups = user_groups.size() 
    n_item_groups = item_groups.size() 

    # assert n_users == len(user_groups), f"User count mismatch: {n_users} from interactions vs {len(user_groups)} from groups."
    # assert n_items == len(item_groups), f"Item count mismatch: {n_items} from interactions vs {len(item_groups)} from groups."

    print(f"Using device: {device}")

    print(f"Using device: {device}")

    # DataLoader for user batches
    user_dataset = TensorDataset(torch.arange(n_users, device=device))
    user_loader = DataLoader(user_dataset, batch_size=batch_size)

    model = MatrixFactorization(n_users, n_items, emb_size).to(device)
    loss_fn = JMEFairnessLoss(alpha=alpha, gamma=gamma, tau=tau, top_k=k, num_samples=n_samples)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        #progress bar
        pbar = tqdm.tqdm(user_loader, desc=f"Epoch {epoch+1}/{epochs}")

        losses = []

        for (batch_user_ids,) in pbar:
            optimizer.zero_grad()

            # Get scores for all items for the users in the batch
            pred_scores = model(batch_user_ids, torch.arange(n_items, device=device))
        
            # Get corresponding ground truth and group info for the batch
            true_scores_batch = true_interactions[batch_user_ids]
            user_groups_batch = user_groups[batch_user_ids]

            #print(pred_scores.device, true_scores_batch.device)

            # Compute loss
            # loss = loss_fn(pred_scores, true_scores_batch, user_groups_batch, item_groups)
            loss = F.mse_loss(pred_scores.to(float), true_scores_batch.to(float))

            #print(loss.dtype, pred_scores.dtype, true_scores_batch.dtype)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        if (epoch+1)%100 == 0:
            avg_loss = total_loss / len(user_loader)
            print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
            losses.append(avg_loss)

    # print("Training finished.")
    # print(f'pred_scores: {pred_scores.size()}')

    pred_scores = model(torch.arange(n_users, device=device), torch.arange(n_items, device=device))
    print(f'pred_scores: {pred_scores.size()}, true interactions: {true_interactions.size()}')
    return losses, pred_scores

# train using a predefined preference matrix
# def train(scores: torch.Tensor, user_groups: torch.Tensor, item_groups: torch.Tensor,
#          emb_size: int=32, batch_size: int=100, epochs: int=200, lr: float=1e-2,
#          alpha: float=.0, gamma: float=.8, tau: float=0.1, k: int=100, n_samples: int=100, 
#          device: str=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
   
#     n_users, n_items = scores.size()
#     n_user_groups = user_groups.size()  # e.g., based on age bins
#     n_item_groups = item_groups.size() # e.g., based on genre

#     # DataLoader for user batches
#     user_dataset = TensorDataset(torch.arange(n_users, device=device))
#     loss_fn = JMEFairnessLoss(alpha=alpha, gamma=gamma, tau=tau, top_k=k, num_samples=n_samples)
#     user_loader = DataLoader(user_dataset, batch_size=batch_size)
#     torch.autograd.set_detect_anomaly(True)
#     print("Adjust preference matrix")

#     losses = []

#     for (batch_user_ids,) in user_loader:
#         with torch.no_grad():
#             original_scores = scores
#             #original_scores = model(batch_user_ids, torch.arange(n_items, device=device))
#         theta = torch.nn.Parameter(torch.rand(n_items, device=device), requires_grad=True)
#         b = torch.nn.Parameter(torch.rand(n_items, device=device), requires_grad=True)
#         optimizer = torch.optim.Adam([theta, b], lr=lr)

#         for epoch in range(epochs):
#             optimizer.zero_grad()

#             new_scores = original_scores * theta + b
#             new_scores = new_scores.clamp(min=0)

#             true_scores_batch = scores[batch_user_ids]
#             user_groups_batch = user_groups[batch_user_ids]

#             loss, ii_f_loss, gg_f_loss = loss_fn(new_scores, true_scores_batch.detach(), user_groups_batch, item_groups)

#             loss.backward()
#             optimizer.step()

#             if epoch % 10 == 0:
#                 print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} II-F: {ii_f_loss.item():.4f}, GG-F: {gg_f_loss.item():.4f}")
    
#     losses.append(loss)
    
#     print(f'std pred scores: {new_scores.std()}')#, std pred scores items: {new_scores.std(dim=0)}, std pred scores users: {new_scores.std(dim=1)}')
    
#     return losses, new_scores