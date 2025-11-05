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

def read_parquet(dataset_name, n_interactions, result_type=None, method_name=None):
    """
    dataset_name: [amazon, black_friday, movielens, yelp]
    result_type: [scores, top_k, users_utility, items_utility]
    n_interactions: [300, 350, 400, 500, 700, 1000, 1500]
    """
    if method_name == None:
        return pd.read_parquet(f'results/scores_{dataset_name}_{n_interactions}.parquet')
    return pd.read_parquet(f'results/{method_name}/{dataset_name}_{result_type}_{n_interactions}_interactions.parquet')

def save_all_results():
    ks = [20, 40, 60, 80, 100]
    datasets = ['amazon', 'black_friday', 'movielens', 'yelp']
    methods = [fair_reciprocal, gini_functions, two_sided_lorenz, jme, itfr, pct, tfrom, fairrec, cpfair]
    n_interactions = [300, 350, 400, 500, 700, 1000, 1500]

    results_table = pd.DataFrame(columns=['Approach', 'Dataset', 'Users', 'Items', 'Interactions', \
    'k', 'Variance-NDCG', 'Variance-Exposure', 'Variance-ExposureEqWeight', 'Entropy', \
    'Gini', 'MMS', 'IIF'])

    for k in ks:
        for method in methods:
            method_name = method.NAME
            for dataset in datasets:
                for n in n_interactions:
                    try: 
                        # Adding baseline
                        scores = read_parquet(dataset_name=dataset, n_interactions=n)
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
                            'Variance-NDCG': np.nan, # Not applicable for baseline
                            'Variance-Exposure': var_exposure(scores.to_numpy(), baseline_topk, k=k),
                            'Variance-ExposureEqWeight': var_exposure(scores.to_numpy(), baseline_topk, k=k, equal_weight=True),
                            'Entropy': entropy(baseline_topk, k),
                            'Gini': gini_coefficient(scores, baseline_topk, k),
                            'MMS': maximin_producer(baseline_topk, m_items, k),
                            'IIF': np.nan, # Not applicable for baseline
                        }

                        results_table.loc[len(results_table)] = baseline_results

                        # Adding other methods results                        
                        fair_scores = read_parquet(dataset_name=dataset, n_interactions=n, result_type='fair_scores', method_name=method_name)
                        fair_topk = read_parquet(dataset_name=dataset, n_interactions=n, result_type='top_k', method_name=method_name)
                        fair_topk = fair_topk.to_numpy()[:, :k]

                        results = {
                            'Approach': method_name,
                            'Dataset': dataset,
                            'Users': n_users,
                            'Items': m_items,
                            'Interactions': n,
                            'k': k,
                            'Variance-NDCG': var_NDCG(scores.to_numpy(), fair_scores.to_numpy(), k=k), 
                            'Variance-Exposure': var_exposure(fair_scores.to_numpy(), fair_topk, k=k),
                            'Variance-ExposureEqWeight': var_exposure(fair_scores.to_numpy(), fair_topk, k=k, equal_weight=True),
                            'Entropy': entropy(fair_topk, k),
                            'Gini': gini_coefficient(scores, fair_topk, k),
                            'MMS': maximin_producer(fair_topk, m_items, k),
                            'IIF': ii_f(scores.to_numpy(), fair_scores.to_numpy()),
                            # 'Baseline-Entropy': entropy(baseline_topk, k),
                            # 'Baseline-Variance-Exposure': var_exposure(scores.to_numpy(), baseline_topk, k=k),
                            # 'Baseline-Variance-ExposureEqWeight)': var_exposure(scores.to_numpy(), baseline_topk, k=k, equal_weight=True),
                            # 'Baseline-Gini': gini_coefficient(scores, baseline_topk, k),
                            # 'Baseline-MMS': maximin_producer(baseline_topk, m_items, k)
                        }

                        results_table.loc[len(results_table)] = results

                        print(f'{'*'*10} {method_name}: {dataset} -- {n} {'*'*10}')
                        #print(results)

                        try:
                            user_utilities = read_parquet(dataset_name=dataset, n_interactions=n, result_type='users_utility', method_name=method_name)
                            items_utilities = read_parquet(dataset_name=dataset, n_interactions=n, result_type='items_utility', method_name=method_name)
                            lorenz_curve(user_utilities, method_name, dataset=dataset, n_interactions=n, type='u')
                            lorenz_curve(items_utilities, method_name, dataset=dataset, n_interactions=n, type='v')      
                        
                        except:
                            # some methods do not compute users' and items' utilities.
                            continue
                    except:
                        # if file not found continue to run. Some methods do not have the full n_interactions, because 
                        # of fair reciprocal method that is limited since it is very costly.
                        continue
    results_table.to_csv('results/table_results.csv', sep='\t')



if __name__ == '__main__':
    save_all_results()
    # results = pd.read_csv('results/table_results.csv', sep='\t', index_col=0)
    # entropies = results[['Approach', 'Dataset', '# Items', 'Entropy', 'Baseline Entropy']][results['Dataset'] == 'amazon']

    # print(entropies)

    # plt.figure(figsize=(8,5))

    # for approach, group in entropies.groupby('Approach'):
    #     plt.plot(group['# Items'], group['Entropy'], marker='o', label=approach)

    # plt.xlabel('# Items')
    # plt.ylabel('Entropy')
    # plt.title('Entropy of two-sided fair methods')
    # plt.legend(title='Approach')
    # plt.grid(True, linestyle='--', alpha=.6)
    # plt.savefig('testefig_entropies.png')
    # plt.show()
    # plt.close()