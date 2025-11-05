from utils import *
from evaluation import *

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

import matplotlib.pyplot as plt

def read_parquet(dataset_name, run_i, n_interactions, result_type=None, method_name=None):
    """
    dataset_name: [amazon, black_friday, movielens, yelp]
    result_type: [scores, top_k, users_utility, items_utility]
    n_interactions: [300, 350, 400, 500, 700, 1000, 1500]
    """
    if method_name == None:
        return pd.read_parquet(f'results/scores_{dataset_name}_{n_interactions}_run{run_i}.parquet')
    return pd.read_parquet(f'results/{method_name}/{dataset_name}_{result_type}_{n_interactions}_interactions_{run_i}.parquet')

def save_all_results():
    ks = [20, 40, 60, 80, 100]
    datasets = ['amazon', 'black_friday', 'movielens', 'yelp']
    methods = [fair_reciprocal, gini_functions, two_sided_lorenz, jme, itfr, pct, tfrom, fairrec, cpfair]
    n_interactions = [300, 350, 500, 700, 1000, 1500]

    results_table = pd.DataFrame(columns=['Approach', 'Dataset', 'Users', 'Items', 'Interactions', \
    'k', 'VarianceNDCG', 'MeanNDCG', 'VarianceExposure', 'VarianceExposureEqWeight', 'Entropy', \
    'Gini', 'MMS', 'IIF', 'GGF'])

    for k in ks:
        for method in methods:
            method_name = method.NAME
            print(method_name)
            for dataset in datasets:
                for n in n_interactions:
                    for i in range(5):
                        user_groups = pd.read_json(f'results/groups/user_groups_{dataset}_{n}_run{i}.json', orient='index')
                        item_groups = pd.read_json(f'results/groups/item_groups_{dataset}_{n}_run{i}.json', orient='index')
                        try: 
                            # Adding baseline
                            scores = read_parquet(dataset_name=dataset, n_interactions=n, run_i=i)
                            n_users, m_items = scores.shape
                            _, sorted_indices = torch.sort(torch.tensor(scores.to_numpy()), descending=True, dim=1)
                            baseline_topk = sorted_indices[:, :k]

                            baseline_results = {
                                'Approach': 'Baseline',
                                'Dataset': dataset,
                                'Users': n_users,
                                'Items': m_items,
                                'Interactions': n,
                                'k': k,
                                'VarianceNDCG': 0., # Not applicable for baseline
                                'MeanNDCG': 1.,
                                'VarianceExposure': var_exposure(scores.to_numpy(), baseline_topk, k=k),
                                'VarianceExposureEqWeight': var_exposure(scores.to_numpy(), baseline_topk, k=k, equal_weight=True),
                                'Entropy': entropy(baseline_topk, k),
                                'Gini': gini_coefficient(scores, baseline_topk, k),
                                'MMS': maximin_producer(baseline_topk, m_items, k),
                                'IIF': ii_f(scores.to_numpy(), scores.to_numpy()), #np.nan, # Not applicable for baseline 
                                'GGF': gg_f(scores, scores, user_groups, item_groups)
                            }

                            results_table.loc[len(results_table)] = baseline_results

                            # Adding other methods results                        
                            fair_scores = read_parquet(dataset_name=dataset, run_i=i, n_interactions=n, result_type='fair_scores', method_name=method_name)
                            fair_topk = read_parquet(dataset_name=dataset, run_i=i, n_interactions=n, result_type='top_k', method_name=method_name)
                            fair_topk = fair_topk.to_numpy()[:, :k]

                            results = {
                                'Approach': method_name,
                                'Dataset': dataset,
                                'Users': n_users,
                                'Items': m_items,
                                'Interactions': n,
                                'k': k,
                                'VarianceNDCG': var_NDCG(scores.to_numpy(), fair_scores.to_numpy(), k=k), 
                                'MeanNDCG': mean_NDCG(scores.to_numpy(), fair_scores.to_numpy(), k=k),
                                'VarianceExposure': var_exposure(fair_scores.to_numpy(), fair_topk, k=k),
                                'VarianceExposureEqWeight': var_exposure(fair_scores.to_numpy(), fair_topk, k=k, equal_weight=True),
                                'Entropy': entropy(fair_topk, k),
                                'Gini': gini_coefficient(scores, fair_topk, k),
                                'MMS': maximin_producer(fair_topk, m_items, k),
                                'IIF': ii_f(scores.to_numpy(), fair_scores.to_numpy()),
                                'GGF': gg_f(scores, fair_scores, user_groups, item_groups)
                            }

                            results_table.loc[len(results_table)] = results

                            print(f'{'*'*10} {method_name}: {dataset} -- {n} -- {k}{'*'*10}')
                        except:
                            print(f'Method {method_name} for {n} interactions not found.')
                            # if file not found continue to run. Some methods do not have the full n_interactions, because 
                            # of fair reciprocal method that is limited since it is very costly.
                            continue

    results_table = results_table.drop_duplicates()

    results_table.to_csv('results/new_table_results.csv', sep='\t')


def plot_lorenz():
    ltr = [fair_reciprocal, jme, gini_functions, two_sided_lorenz]
    datasets = ['amazon', 'black_friday', 'movielens', 'yelp']
    n_interactions = [350, 1500]

    for method in ltr:
        method_name = method.NAME
        for data in datasets:
            for n in n_interactions:
                if n > 350 and method_name == 'FR':
                    continue
                
                df_user_utilities = pd.DataFrame()
                df_item_utilities = pd.DataFrame()
                for i in range(5):
                    if method_name == 'JME':
                        fair_scores = read_parquet(dataset_name=data, run_i=i, n_interactions=n, \
                                            result_type='fair_scores', method_name=method_name).to_numpy()
                        fair_top_k = read_parquet(dataset_name=data, run_i=i, n_interactions=n, \
                                            result_type='top_k', method_name=method_name).to_numpy()
                        df_user_utilities = get_utility(fair_scores, fair_top_k, k=fair_scores.shape[1], axis=1)
                        df_item_utilities = get_utility(fair_scores, fair_top_k, k=fair_scores.shape[1], axis=0)

                    else:
                        df_user = read_parquet(dataset_name=data, run_i=i, n_interactions=n, \
                        result_type='users_utility', method_name=method_name)
                    
                        df_item = read_parquet(dataset_name=data, run_i=i, n_interactions=n, \
                        result_type='items_utility', method_name=method_name)
                    
                    df_item_utilities = pd.concat([df_item_utilities, df_item], axis=1)
                    df_user_utilities = pd.concat([df_user_utilities, df_user], axis=1)

                df_user_utilities = df_user_utilities.mean(axis=1)
                df_item_utilities = df_item_utilities.mean(axis=1)
                lorenz_curve(df_user_utilities, method_name, data, n, type='u')
                lorenz_curve(df_item_utilities, method_name, data, n, type='v')
                

def group_results():
    datasets = ['amazon', 'black_friday', 'movielens', 'yelp']
    y_axis_cols = ['VarianceNDCG', 'MeanNDCG', 'VarianceExposure', 'VarianceExposureEqWeight', \
    'Entropy','Gini', 'MMS', 'IIF', 'GGF']
    x_axis_cols = ['Interactions', 'k']

    for dataset in datasets:
        for x in x_axis_cols:
            for y in y_axis_cols:
                results = pd.read_csv('results/new_table_results.csv', sep='\t', index_col=0)
                if x == 'Interactions':
                    # fixing k=80 
                    metric_table = results[['Approach', 'Dataset', x, y]][(results['Dataset'] == dataset) & (results['k'] == 40)]
                else:
                    # fixing 1000 interactions
                    metric_table = results[['Approach', 'Dataset', x, y]][(results['Dataset'] == dataset) & (results['Interactions'] == 1500)]
                
                new_table = metric_table.groupby(['Approach', x])[y].agg(['mean', 'std'])

                new_table = new_table.reset_index()
                new_table['std'] = new_table['std'].replace({np.nan: 0})
                new_table.to_csv(f'results_plots/tables_plots/{dataset}/{y}_{x}.csv')

if __name__ == '__main__':
    #save_all_results()
    #group_results()
    plot_lorenz()

