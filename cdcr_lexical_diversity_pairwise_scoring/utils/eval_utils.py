from sklearn.metrics import confusion_matrix


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred).ravel()


def precision_recall_f1(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if tp + fp != 0 else 0.0
    recall = tp / (tp + fn) if tp + fn != 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0.0
    return precision, recall, f1
