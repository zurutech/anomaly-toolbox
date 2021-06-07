"""Function used in benchmarks to compute metrics."""

import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
)

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)


def compute_prc(
    true_labels: np.array,
    scores: np.array,
    file_name: Optional[str] = None,
    plot: bool = False,
) -> Tuple:
    """PRC Curve."""
    ap = average_precision_score(true_labels, scores)
    precision, recall, thresholds = precision_recall_curve(true_labels, scores)
    prc_auc = auc(recall, precision)
    print("Anomaly Scores ScikitLearn PR-Thresholds: ", thresholds)

    if plot:
        plt.figure()
        plt.step(recall, precision, color="b", alpha=0.2, where="post")
        plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title("Precision-Recall curve: AUC=%0.4f" % (prc_auc))
        if file_name:
            plt.savefig(file_name)
        plt.close()

    return prc_auc, ap


# def roc(labels, scores, saveto=None):
#     """Compute ROC curve and ROC area for each class"""
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()

#     labels = labels.cpu()
#     scores = scores.cpu()

#     # True/False Positive Rates.
#     fpr, tpr, _ = roc_curve(labels, scores)
#     roc_auc = auc(fpr, tpr)

#     # Equal Error Rate
#     eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

#     if saveto:
#         plt.figure()
#         lw = 2
#         plt.plot(
#             fpr,
#             tpr,
#             color="darkorange",
#             lw=lw,
#             label="(AUC = %0.2f, EER = %0.2f)" % (roc_auc, eer),
#         )
#         plt.plot([eer], [1 - eer], marker="o", markersize=5, color="navy")
#         plt.plot([0, 1], [1, 0], color="navy", lw=1, linestyle=":")
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.05])
#         plt.xlabel("False Positive Rate")
#         plt.ylabel("True Positive Rate")
#         plt.title("Receiver operating characteristic")
#         plt.legend(loc="lower right")
#         plt.savefig(os.path.join(saveto, "ROC.pdf"))
#         plt.close()

#     return roc_auc


# def evaluate(labels, scores, metric="roc"):
#     if metric == "roc":
#         return roc(labels, scores)
#     elif metric == "auprc":
#         return auprc(labels, scores)
#     elif metric == "f1_score":
#         threshold = 0.20
#         scores[scores >= threshold] = 1
#         scores[scores < threshold] = 0
#         return f1_score(labels.cpu(), scores.cpu())
#     else:
#         raise NotImplementedError("Check the evaluation metric.")


# def auprc(labels, scores):
#     ap = average_precision_score(labels.cpu(), scores.cpu())
#     return ap
