from utils import *
from evaluation import *

import json
import os

# Learning to Rank models
import ltr.fair_reciprocal as fair_reciprocal
import ltr.gini_functions as gini_functions
import ltr.joint_multisided as jme
import ltr.two_sided_lorenz as two_sided_lorenz

# Greedy/opt models
import opt.fairrec_patro as fairrec 
import opt.ITFR as itfr

# Rerank models
import rr.greedy_cpfair as cpfair
import rr.pct as pct
import rr.tfrom as tfrom

from typing import Any, Dict, List

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def save_results(data_name: str, model_name: str, run_i, n_interactions: int, result_type: str, result: Any) -> None:
    df = pd.DataFrame(result)
    df.to_parquet(f'results/{model_name}/{data_name}_{result_type}_{n_interactions}_interactions_{run_i}.parquet')


def run(data_path: str, models: list, interactions: list, run_i: int, n=None, epochs: int=500, lr: float=.01, seed:int=42, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    ## OBS: TESTAR FAIR RECIPROCAL SÓ ATÉ UMA DETERMINADA QUANTIDADE DE INTERAÇÕES POIS 
    ## O MÉTODO É MAIS PESADO E A MEMÓRIA DA GPU ESTOURA COM MUITOS DADOS
    #models = [fair_reciprocal, gini_functions, jme, two_sided_lorenz, fairrec, itfr, cpfair, pct, tfrom]
    # n_interactions = [300, 350, 400, 450]

    # models = [itfr, pct, tfrom, jme, gini_functions, two_sided_lorenz, fairrec, cpfair]
    # n_interactions = [500, 700, 1000, 1500]

    models_w_utility = [gini_functions, two_sided_lorenz, fair_reciprocal]
    models_for_groups = [itfr, fairrec]

    # TEM DOIS ALGORITMOS GREEDY: FAIRREC E CPFAIR
    # vram does not support more than 1000 interactions. With 500 users and 546 items for random seed=42. 
    for n in n_interactions:
        torch.cuda.empty_cache()
        data = load_data(data_path, n=n, run_i=run_i, random_state=seed)
        data = torch.from_numpy(data).to(device)

        print(data.size())

        data_name = get_data_name(data_path)

        k = data.size(1)

        _, base_topk = torch.topk(data, k=k)

        # Dividing users into active and inactive, and items into short-head and long-tail
        user_groups = get_active_users(data_path, run_i=run_i)
        item_groups = get_top_items(data_path, run_i=run_i)
        ## SAVING GROUPS INFO
        data_name_to_save = data_path.split('/')[-1].split('.')[0]
        with open(f'results/groups/user_groups_{data_name_to_save}_{n}_run{run_i}.json', 'w') as f:
            json.dump(user_groups, f)
        with open(f'results/groups/item_groups_{data_name_to_save}_{n}_run{run_i}.json', 'w') as f:
            json.dump(item_groups, f)

        for model in models:
            print('*'*20)
            print(model.NAME)
            if model in models_w_utility:
                if model.NAME == 'FR':
                    data_recip = (data.T).to(device)
                    S = square_matrix_for_reciprocal(data, data_recip).to(device)
                    fair_scores, P_stochastic, u, v = model.train(S, n_users=data.size(0), welfare_function='NSW', epochs=epochs)
                else:
                    fair_scores, P_stochastic, u, v = model.train(data, epochs=epochs)
                
                fair_scores_topk, fair_topk = topk_from_P(P_stochastic, k)
                results_list = [('fair_scores', fair_scores), ('top_k', fair_topk), ('users_utility', u), ('items_utility', v)]


            elif model in models_for_groups:
                if model.NAME == 'ITFR':
                    item_groups_df, user_groups_df = groups_as_df(item_groups, user_groups)
                    interactions = get_dict_interaction(data_path=data_path, run_i=run_i, item_groups=item_groups_df, user_groups=user_groups_df)
                    fair_scores, fair_topk = model.train(user_map=user_groups_df, item_map=item_groups_df, interactions=interactions, epochs=epochs, k=k)
                else:
                    fair_scores, fair_topk = model.train(scores=data, k=k)
                results_list = [('fair_scores', fair_scores), ('top_k', fair_topk)]
            
            else:
                if model.NAME == 'JME':
                    fair_scores, fair_topk = model.jme(data_path, run_i=run_i, user_groups=user_groups, item_groups=item_groups, k=k, epochs=epochs)

                # Fair scores unchanged for the models below (rerankers)
                elif model.NAME == 'CPFair':
                    fair_scores, fair_topk = model.cpfair(data, user_groups=user_groups, item_groups=item_groups, base_topk=base_topk.cpu().numpy(), k=k)
                elif model.NAME == 'PCT':
                    fair_scores, fair_topk = model.pct(data_path, run_i=run_i, item_mapping=item_groups, k=k)
                elif model.NAME == 'TFROM':
                    fair_scores, fair_topk = model.tfrom(data, k=k)
                            
                results_list = [('fair_scores', fair_scores), ('top_k', fair_topk)]
    
            for result_type, result in results_list:
                result = result.cpu()
                if result.requires_grad:
                    result = result.detach()
                result = result.numpy()

                save_results(data_name, model.NAME, run_i=run_i, n_interactions=n, result_type=result_type, result=result)

def groups_as_df(item_groups, user_groups):
    df_item_groups = pd.DataFrame(item_groups.items(), columns=['item_id', 'item_group'])
    df_user_groups = pd.DataFrame(user_groups.items(), columns=['user_id', 'user_group'])

    return df_item_groups, df_user_groups


def get_dict_interaction(data_path: str, run_i: int, item_groups: pd.DataFrame, user_groups: pd.DataFrame):
    try:
        df = pd.read_csv(data_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    except:
        df = pd.read_csv(data_path, sep='\t', names=['user_id', 'item_id', 'rating'])

    #### preparing data
    df = df[['user_id', 'item_id', 'rating']]
    
    indexes_sample = np.load(f'data/temp_data/temp_indexes_{data_path.split('/')[-1].split('.')[0]}_run{run_i}.npz')
    data = df.loc[indexes_sample]

    rows_idx, cols_idx = mapping_indexes(np.zeros(len(data)), data.to_numpy()[:, 0], data.to_numpy()[:, 1])
    data.iloc[:,0], data.iloc[:,1] = rows_idx, cols_idx

    df_groups = data.copy()
    df_groups = pd.merge(df_groups, item_groups, on='item_id')
    df_groups = pd.merge(df_groups, user_groups, on='user_id')

    # Group interactions by intersectional group
    interactions = {}
    for (user_grp, item_grp), group_df in df_groups.groupby(['user_group', 'item_group']):
        interactions[(user_grp, item_grp)] = list(zip(group_df['user_id'], group_df['item_id']))

    return interactions

    
if __name__ == '__main__':
    datasets = ['data/amazon.dat', 'data/black_friday.csv', 'data/movielens_100k_u1.base', 'data/yelp.base']   
    # models = [jme, fair_reciprocal, gini_functions, two_sided_lorenz, fairrec, itfr, cpfair, pct, tfrom]
    # n_interactions = [300, 350]

    # for i in range(5):
    #     for data_path in datasets:
    #         print('running for fr...')
    #         run(data_path=data_path, epochs=500, models=models, interactions=n_interactions, run_i=i)

    models = [jme, itfr, pct, tfrom, gini_functions, two_sided_lorenz, fairrec, cpfair]
    n_interactions = [500, 700, 1000, 1500]

    for i in range(5):
        for data_path in datasets:
            print('Running...')
            run(data_path=data_path, epochs=500, models=models, interactions=n_interactions, run_i=i)

