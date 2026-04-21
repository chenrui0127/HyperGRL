import argparse
import os
import re

import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn import metrics
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.datasets import Amazon, Planetoid, Coauthor, WikiCS, CoraFull
from torch_geometric.nn import GCNConv, TransformerConv
from torch_geometric.transforms import ToUndirected, NormalizeFeatures, NormalizeScale, NormalizeRotation

# from alpha_decay import combined_cosine_decay
from transforms import get_graph_drop_transform
from sklearn.model_selection import train_test_split

import torch
import torch_geometric
import random
import numpy as np

torch.serialization.add_safe_globals([torch_geometric.data.data.DataTensorAttr])

torch.manual_seed(3456)
np.random.seed(3456)
random.seed(3456)


def compute_alpha_t(alpha_min, alpha_max, beta, H_target, H_proxy_t):
    """
    Compute alpha_t based on the Entropy-Guided Adaptive Balancing formula.

    Parameters:
    - alpha_min (float): Minimum value for alpha.
    - alpha_max (float): Maximum value for alpha.
    - beta (float): Sensitivity hyperparameter.
    - H_target (float): Target entropy value.
    - H_proxy_t (float): Proxy entropy at time t.

    Returns:
    - float: Computed alpha_t.
    """
    if H_target == 0:
        raise ValueError("H_target cannot be zero to avoid division by zero.")

    normalized_diff = (H_target - H_proxy_t) / H_target
    sigmoid_arg = beta * normalized_diff
    sigmoid = 1 / (1 + torch.exp(-sigmoid_arg))
    alpha_t = alpha_min + (alpha_max - alpha_min) * sigmoid
    return alpha_t


def normalize(v): return F.normalize(v, dim=-1, p=2, eps=1e-5)


def eval_logistic_regression(X, y, data_random_seed=3456, repeat=3):
    one_hot_encoder = OneHotEncoder(categories='auto')

    y = y.squeeze()
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(bool)

    rng = np.random.RandomState(data_random_seed)
    accuracies = []
    f1_scores = []
    for _ in range(repeat):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=rng)

        logreg = LogisticRegression(solver='liblinear')
        c = 2.0 ** np.arange(-10, 11)
        cv = ShuffleSplit(n_splits=5, test_size=0.5)
        clf = GridSearchCV(estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c),
                           n_jobs=10, cv=cv, verbose=0)
        clf.fit(x_train, y_train)

        y_pred = clf.predict_proba(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)

        test_acc = metrics.accuracy_score(y_test, y_pred)
        accuracies.append(test_acc)
        f1 = metrics.f1_score(y_test, y_pred, average='macro')
        f1_scores.append(f1)
    return accuracies, f1_scores


def eval_clustering(X, y, num_classes, method="kmeans"):
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    y = y.squeeze()
    if method == "kmeans":
        km = KMeans(n_clusters=num_classes, n_init=5, random_state=0).fit(X)
        pred = km.labels_
        nmi = normalized_mutual_info_score(y, pred, average_method='arithmetic')
        ari = adjusted_rand_score(y, pred)
    else:
        dbscan = HDBSCAN().fit(X)
        pred = dbscan.labels_
        nmi = normalized_mutual_info_score(y, pred, average_method='arithmetic')
        ari = adjusted_rand_score(y, pred)

    return nmi, ari


def load_dataset(name, root='../../datasets'):
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        ds = Planetoid(root, name)
    elif name in ['CS', 'Physics']:
        ds = Coauthor(root, name)
    elif name in ['Computers', 'Photo']:
        ds = Amazon(root, name)
    elif name in ['WikiCS']:
        ds = WikiCS(root, is_undirected=True)
        ds.name = 'WikiCS'
    elif name in ['CoraFull']:
        ds = CoraFull(root)
        ds.name = 'CoraFull'
    elif name in ['ogbn-arxiv']:
        ds = PygNodePropPredDataset(name=name)
    else:
        raise ValueError('Unknown dataset, available datasets: Cora, CiteSeer, PubMed, Computers, Photo, WikiCS')
    return ds


# ---------------- models ----------------
class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.activation = torch.nn.SiLU()
        self.layers = nn.ModuleList()
        self.layers.append(TransformerConv(input_dim, hidden_dim // 16, heads=16, beta=False))

        for _ in range(num_layers - 1):
            self.layers.append(TransformerConv(hidden_dim, hidden_dim // 4, heads=4, beta=False))

    def forward(self, x, edge_index):
        z = x
        outputs = []
        for i, conv in enumerate(self.layers):
            h = conv(z, edge_index)

            if i > 0:
                z = h + z
            else:
                z = h
            z = self.activation(z)
            outputs.append(z)
        return z, outputs


# ---------------- helpers ----------------
def neighbor_mean(z, edge_index, num_nodes, add_self=False):
    row, col = edge_index
    dev = z.device
    acc = torch.zeros((num_nodes, z.shape[1]), device=dev)
    cnt = torch.zeros((num_nodes, 1), device=dev)
    acc.index_add_(0, row, z[col])
    cnt.index_add_(0, row, torch.ones((row.shape[0], 1), device=dev))
    if add_self:
        acc = acc + z
        cnt = cnt + 1.0
    return acc / (cnt + 1e-9), cnt


def l2_normalize(x, eps=1e-16):
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))


def loss_uniform(x, t=2):
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()


def adversarial_loss(z, edge_index, num_nodes, alpha=1.0, k=1):
    """
    z: [N, d] (assumed already L2-normalized; if not, we normalize)
    edge_index: [2, E]
    neighbor_mean: a function that returns per-node neighbor mean vectors
    """
    # Compute k-order neighbor mean targets
    nb_mean = z
    nb_count = torch.zeros((num_nodes, 1), device=z.device)
    for i in range(k):
        nb_mean, nb_count = neighbor_mean(nb_mean, edge_index, num_nodes, add_self=True)
        nb_mean = l2_normalize(nb_mean)

    # Align node representations to the neighbor mean targets
    cos_sim = (z * nb_mean).sum(dim=1)
    w = torch.pow(torch.sigmoid(nb_count), 5)
    align_loss = w * (1. - cos_sim)
    align_loss = align_loss.mean()

    collapse_metric = z.mean(dim=0)
    collapse_metric = (collapse_metric * collapse_metric).sum()
    uniformity_loss = collapse_metric

    # Objective function
    loss = align_loss + alpha * uniformity_loss
    loss = loss.mean()

    components = {
        "align": align_loss.mean().item(),
        "uniformity": uniformity_loss.mean().item(),
        "loss": loss.item(),
    }
    return loss, components, collapse_metric


# ---------------- load data ----------------
def run(args, trial):
    dataset = load_dataset(args.dataset, root=args.dataset_dir)

    # prepare transforms
    transform = get_graph_drop_transform(drop_edge_p=args.drop_edge_p, drop_feat_p=args.drop_feat_p)

    print(dataset.name, dataset[0], '\n')

    if len(dataset[0].y.shape) > 1:
        dataset[0].y = dataset[0].y.squeeze(1)

    num_nodes = dataset.x.shape[0]
    input_dim = dataset.x.shape[1]
    data = ToUndirected()(dataset[0]).to(args.device)

    print(dataset.name, data, '\n')

    # ---------------- instantiate models ----------------
    encoder = GraphEncoder(input_dim, args.hidden_dim, num_layers=args.num_layers).to(args.device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---------------- training ----------------
    epochs = args.num_epochs
    loss_hist = []
    nmi_hist = {}
    ari_hist = {}

    best_loss = 1e9
    best_epoch = 0
    patience = 50

    alpha_max = 10
    alpha_min = 0.01
    alpha = alpha_max
    gamma = 0.1

    for epoch in range(epochs):
        encoder.train()

        arg_data = transform(data)

        h, o = encoder(arg_data.x, arg_data.edge_index)  # [N,z_dim]
        z = l2_normalize(h)

        loss, comps, cm = adversarial_loss(z, arg_data.edge_index, data.num_nodes,
                                           alpha=alpha, k=args.k)

        # update alpha
        hp = -torch.log(cm + 1e-6).detach()
        alpha_pred = compute_alpha_t(alpha_min, alpha_max, 5, args.h_target, hp)
        alpha = (1 - gamma) * alpha + gamma * alpha_pred

        loss_hist.append(comps)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            loss_msg = " | ".join([f"{k} {v:.5f}" for k, v in loss_hist[-1].items()])

            nmi, ari = eval_clustering(z.cpu().detach().numpy(), data.y.cpu().detach().numpy(), dataset.num_classes)
            nmi_hist[epoch] = nmi
            ari_hist[epoch] = ari

            print(f"Epoch {epoch + 1:>3d}/{epochs} | {loss_msg} | NMI {nmi:.4f} | ARI {ari:.4f} | alpha {alpha:.4f}")

        if best_loss > loss.item():
            # print("Reset patience")
            patience = 50
            best_loss = loss.item()
            best_epoch = epoch
            os.makedirs(f"checkpoints", exist_ok=True)
            torch.save(encoder.state_dict(), f"checkpoints/model_{dataset.name}_{trial}_{best_epoch + 1}.pth")
        else:
            # print(f'patience {patience}')
            patience = patience - 1
            if patience <= 0:
                print(f"\nEarly stopping at epoch {epoch + 1}/{epochs}, best epoch {best_epoch + 1}.")
                break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    best_path = f"checkpoints/model_{dataset.name}_{trial}_{best_epoch + 1}.pth"
    print(f"Load BEST model: {best_path}, loss: {best_loss:.5f}, epoch: {best_epoch + 1}")
    state_dict = torch.load(best_path, weights_only=True)
    encoder.load_state_dict(state_dict)

    h, o = encoder(data.x, data.edge_index)  # [N,z_dim]
    z = l2_normalize(h)

    print("\nEvaluating ...\n")

    x = z.cpu().detach().numpy()
    y = data.y.cpu().numpy()

    nmi, ari = eval_clustering(x, y, dataset.num_classes)
    print(f"NMI: {nmi:.4f} \nARI: {ari:.4f}")

    acc_scores, f1_scores = eval_logistic_regression(x, y, repeat=10)
    acc = np.mean(acc_scores)
    var = np.var(acc_scores)
    print(f"ACC: {acc:.4f} {var:.6f}")

    f1 = np.mean(f1_scores)
    var = np.var(f1_scores)
    print(f" F1: {f1:.4f} {var:.6f}\n")

    return acc, f1, nmi, ari


def main(args):
    acc_list, f1_list, nmi_list, ari_list = [], [], [], []
    for trial in range(args.trials):
        print(f"\nTrial {trial + 1:>3d}/{args.trials}")
        acc, f1, nmi, ari = run(args, trial + 1)

        acc_list.append(acc)
        f1_list.append(f1)
        nmi_list.append(nmi)
        ari_list.append(ari)

    acc_list = np.array(acc_list)
    f1_list = np.array(f1_list)
    nmi_list = np.array(nmi_list)
    ari_list = np.array(ari_list)

    with open(f'results/results_{args.dataset}_{args.trials}.txt', 'a') as f:
        cfg = f"""
            k: {args.k} 
            h_target: {args.h_target:.2f} 
            drop_edge_p: {args.drop_edge_p:.2f} 
            drop_feat_p: {args.drop_feat_p:.2f} 
            hidden_dim: {args.hidden_dim}\n
            """
        cfg = re.sub(r"\n", "", cfg)
        cfg = re.sub(r"\s+", " ", cfg).strip()
        f.write(cfg + "\n")
        f.write(f"Acc: {np.mean(acc_list):.4f} {np.std(acc_list):.4f}\n")
        f.write(f"F1 : {np.mean(f1_list):.4f} {np.std(f1_list):.4f}\n")
        f.write(f"NMI: {np.mean(nmi_list):.4f} {np.std(nmi_list):.4f}\n")
        f.write(f"ARI: {np.mean(ari_list):.4f} {np.std(ari_list):.4f}\n")

    print("\n======= Summary =======")
    print(cfg)
    print(f"Acc: {np.mean(acc_list):.4f} {np.std(acc_list):.4f}")
    print(f"F1 : {np.mean(f1_list):.4f} {np.std(f1_list):.4f}")
    print(f"NMI: {np.mean(nmi_list):.4f} {np.std(nmi_list):.4f}")
    print(f"ARI: {np.mean(ari_list):.4f} {np.std(ari_list):.4f}")

    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Adaptive Neighbor-Mean Alignment')
    parser.add_argument('--dataset', type=str, default='Cora', help='dataset name')
    parser.add_argument('--dataset_dir', type=str, default='../../datasets', help='dataset dir')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--num_epochs', type=int, default=1000, help='max epochs')
    parser.add_argument('--k', type=int, default=1, help='k order neighbor mean')
    parser.add_argument('--h_target', type=float, default=1.5, help='target entropy')
    parser.add_argument('--drop_edge_p', type=float, default=0.8, help='drop_edge_p')
    parser.add_argument('--drop_feat_p', type=float, default=0.1, help='drop_feat_p')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='hidden_dim')
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
    parser.add_argument('--trials', type=int, default=5, help='trials')
    parser.add_argument('--cluster_method', type=str, default="kmeans")
    parser.add_argument('--device', type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help='cuda or cpu')
    parsed_args = parser.parse_args()
    main(parsed_args)
    exit(0)
