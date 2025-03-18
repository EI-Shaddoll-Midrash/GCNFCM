from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import community as community_louvain
import networkx as nx

def nmi(y_true, y_pred):
    nmi = normalized_mutual_info_score(y_true, y_pred)
    print(f'Normalized Mutual Information (NMI): {nmi:.4f}')


def ari(y_true, y_pred):
    ari = adjusted_rand_score(y_true, y_pred)
    print(f'Adjusted Rand Index (ARI): {ari:.4f}')
