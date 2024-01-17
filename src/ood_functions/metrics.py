# Code adapted from
# laplace-redux/utils/utils.py
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error
import scipy.stats as st
import jax.numpy as jnp
import jax


def compute_metrics(i, id, all_y_prob, test_loader, all_y_prob_in, all_y_var, benchmark):
    """compute evaluation metrics"""
    metrics = {}

    # compute Brier, ECE and MCE for distribution shift and WILDS benchmarks
    if benchmark in ["R-MNIST", "R-FMNIST", "CIFAR-10-C", "ImageNet-C"] and (benchmark != "WILDS-poverty"):
        print(f"{benchmark} with distribution shift intensity {i}")
        labels = np.concatenate([data[1].numpy() for data in test_loader])
        metrics["brier"] = get_brier_score(all_y_prob, labels)
        metrics["ece"], metrics["mce"] = get_calib(all_y_prob, labels)

    # compute AUROC and FPR95 for OOD benchmarks
    if benchmark in ["MNIST-OOD", "FMNIST-OOD", "CIFAR-10-OOD"]:
        print(f"{benchmark} - dataset: {id}")
        if i > 0:
            # compute other metrics
            metrics["auroc"] = get_auroc(all_y_prob_in, all_y_prob)
            metrics["fpr95"], _ = get_fpr95(all_y_prob_in, all_y_prob)

    # compute regression calibration
    if benchmark == "WILDS-poverty":
        print(f"{benchmark} with distribution shift intensity {i}")
        labels = torch.cat([data[1] for data in test_loader])
        metrics["calib_regression"] = get_calib_regression(
            all_y_prob.numpy(), all_y_var.sqrt().numpy(), labels.numpy()
        )

    return metrics


def get_auroc(py_in, py_out):
    labels = np.zeros(len(py_in) + len(py_out), dtype="int32")
    labels[: len(py_in)] = 1
    examples = np.concatenate([py_in.max(1), py_out.max(1)])
    return roc_auc_score(labels, examples)


def get_fpr95(py_in, py_out):
    conf_in, conf_out = py_in.max(1), py_out.max(1)
    tpr = 95
    perc = np.percentile(conf_in, 100 - tpr)
    # fp = np.sum(conf_out >= perc)
    fpr = np.sum(conf_out >= perc) / len(conf_out)
    return fpr.item(), perc.item()


def get_brier_score(probs, targets):
    # targets = jax.nn.one_hot(targets, num_classes=probs.shape[1])
    return jnp.mean(jnp.sum((probs - targets) ** 2, axis=1)).item()


def get_calib(pys, y_true, M=100):
    # Put the confidence into M bins
    _, bins = np.histogram(pys, M, range=(0, 1))

    labels = pys.argmax(1)
    y_true = y_true.argmax(1) #This line?
    confs = np.max(pys, axis=1)
    conf_idxs = np.digitize(confs, bins)

    # Accuracy and avg. confidence per bin
    accs_bin = []
    confs_bin = []
    nitems_bin = []

    for i in range(M):
        labels_i = labels[conf_idxs == i]
        y_true_i = y_true[conf_idxs == i]
        confs_i = confs[conf_idxs == i]
        acc = np.nan_to_num(np.mean(labels_i == y_true_i), 0)
        conf = np.nan_to_num(np.mean(confs_i), 0)

        accs_bin.append(acc)
        confs_bin.append(conf)
        nitems_bin.append(len(labels_i))

    accs_bin, confs_bin = np.array(accs_bin), np.array(confs_bin)
    nitems_bin = np.array(nitems_bin)

    ECE = np.average(np.abs(confs_bin - accs_bin), weights=nitems_bin / nitems_bin.sum())
    MCE = np.max(np.abs(accs_bin - confs_bin))

    return ECE, MCE


def get_calib_regression(pred_means, pred_stds, y_true, return_hist=False, M=10):
    """
    Kuleshov et al. ICML 2018, eq. 9
    * pred_means, pred_stds, y_true must be np.array's
    * Set return_hist to True to also return the "histogram"---useful for visualization (see paper)
    """
    T = len(y_true)
    ps = np.linspace(0, 1, M)
    cdf_vals = [st.norm(m, s).cdf(y_t) for m, s, y_t in zip(pred_means, pred_stds, y_true)]
    p_hats = np.array([len(np.where(cdf_vals <= p)[0]) / T for p in ps])
    cal = T * mean_squared_error(ps, p_hats)  # Sum-squared-error

    return (cal, ps, p_hats) if return_hist else cal
