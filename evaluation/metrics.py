from sklearn.metrics import roc_auc_score

def compute_auroc(y_true, y_score):
    return roc_auc_score(y_true, y_score)