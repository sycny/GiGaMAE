import numpy as np
import torch
from torch.utils.data import DataLoader
import functools

from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn import preprocessing

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.metrics.cluster import (
    completeness_score,
    homogeneity_score,
    v_measure_score,
    adjusted_rand_score,  
)

def svc_classify(x,y):
    kf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = None)
    #accuracies = []
    F1Mi = []
    F1Ma = []
    accuracies_val = []
    for train_index, test_index in kf.split(x,y):
        
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        params = {'C':[0.0001,0.001,0.01,0.1,1,10,100,1000]}
        classifier = GridSearchCV(SVC(),params, cv = 3, scoring = 'accuracy', verbose=0)
        classifier.fit(x_train,y_train)
        #accuracies.append(accuracy_score(y_test,classifier.predict(x_test)))
        F1Mi.append(f1_score(y_test,classifier.predict(x_test), average = 'micro'))
        F1Ma.append(f1_score(y_test,classifier.predict(x_test), average = 'macro'))
        
    
    return np.mean(F1Mi),np.mean(F1Ma)



def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            #print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()


@repeat(10)
def label_classification(embeddings, y, ratio):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)

    X = normalize(X, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=1 - ratio)
    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=4, cv=5,
                       verbose=0)
                       
    
    
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    
    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")

    return {
        #'Acc' :accuracy,
        'F1Mi': micro,
        'F1Ma': macro
    }

def clustering(embeddings, y, num_class):
    kmeans_input = embeddings.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=num_class, random_state=0).fit(kmeans_input)
    pred = kmeans.predict(kmeans_input)

    labels = y.cpu().numpy()
    #completeness = completeness_score(labels, pred)
    #hm = homogeneity_score(labels, pred)
    nmi = v_measure_score(labels, pred)
    ari = adjusted_rand_score(labels, pred)
    auccuary = accuracy_score(labels, pred)
    return nmi, ari, auccuary


def evaluate_auc(train_pred, train_true, val_pred, val_true, test_pred, test_true):
     
    train_auc = roc_auc_score(train_true, train_pred)
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    train_ap = average_precision_score(train_true, train_pred)
    valid_ap = average_precision_score(val_true, val_pred)
    test_ap = average_precision_score(test_true, test_pred)
    results = dict()
    results['AUC'] = (train_auc, valid_auc, test_auc)
    results['AP'] = (train_ap, valid_ap, test_ap)
    return results


@torch.no_grad()
def test(model, predictor, data, adj, split_edge, batch_size):
    model.eval()
    h = model.encoder(data.x, data.edge_index)

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    neg_train_edge = split_edge['train']['edge_neg'].to(data.x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(data.x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.x.device)
    pos_test_edge = split_edge['test']['edge'].to(data.x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(data.x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h, edge).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h, edge).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_train_preds = []
    for perm in DataLoader(range(neg_train_edge.size(0)), batch_size):
        edge = neg_train_edge[perm].t()
        neg_train_preds += [predictor(h, edge).squeeze().cpu()]
    neg_train_pred = torch.cat(neg_train_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h, edge).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h, edge).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h, edge).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    train_pred = torch.cat([pos_train_pred, neg_train_pred], dim=0)
    train_true = torch.cat([torch.ones_like(pos_train_pred), torch.zeros_like(neg_train_pred)], dim=0)

    val_pred = torch.cat([pos_valid_pred, neg_valid_pred], dim=0)
    val_true = torch.cat([torch.ones_like(pos_valid_pred), torch.zeros_like(neg_valid_pred)], dim=0)

    test_pred = torch.cat([pos_test_pred, neg_test_pred], dim=0)
    test_true = torch.cat([torch.ones_like(pos_test_pred), torch.zeros_like(neg_test_pred)], dim=0)

    results = evaluate_auc(train_pred, train_true, val_pred, val_true, test_pred, test_true)
    return results