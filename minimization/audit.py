"""
This module implements a black-box, label-only MIA audit layer.
"""
import json
import math
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, roc_curve

from .distillation import sample_region, predict_with_student


def _clamp(x, low, high):
    return max(low, min(high, x))


def _noise_schedule_from_arg(arg):
    if isinstance(arg, (list, tuple, np.ndarray)):
        return [float(x) for x in arg]
    if isinstance(arg, str):
        return [float(x) for x in arg.split(",") if x.strip() != ""]
    return [0.01, 0.02, 0.05, 0.1]


def generate_probes(leaf_regions, feature_data, n_in, n_out, seed=0,
                    categorical_features=None, prioritize_small_leaves=True,
                    verbose=True):
    """Generate in-like and out-like probes from leaf regions.

    Parameters
    ----------
    leaf_regions : dict
        Mapping {leaf_id: Region}.
    feature_data : dict
        Per-feature numeric bounds with keys 'min' and 'max'.
    n_in : int
        Number of in-like probes to generate.
    n_out : int
        Number of out-like probes to generate.
    seed : int, optional
        Random seed.
    categorical_features : dict, optional
        Mapping {feature: [allowed_values]}.
    prioritize_small_leaves : bool, optional
        If True, sample more from smaller leaves.
    verbose : bool, optional
        If True, prints progress.

    Returns
    -------
    X_in_like : pandas.DataFrame
        In-like probes.
    X_out_like : pandas.DataFrame
        Out-like probes.
    meta_in : list
        Metadata for in-like probes.
    meta_out : list
        Metadata for out-like probes.
    """
    rng = np.random.RandomState(seed)
    categorical_features = categorical_features or {}
    leaf_ids = list(leaf_regions.keys())
    if not leaf_ids:
        raise ValueError("No leaf regions available for probe generation.")

    supports = np.array([leaf_regions[lid].support for lid in leaf_ids], dtype=float)
    if prioritize_small_leaves:
        weights = 1.0 / np.maximum(supports, 1.0)
    else:
        weights = supports
    weights = weights / weights.sum()

    # allocate counts per leaf
    in_counts = rng.multinomial(n_in, weights)
    out_counts = rng.multinomial(n_out, weights)

    X_in_like = []
    X_out_like = []
    meta_in = []
    meta_out = []

    for i, leaf_id in enumerate(leaf_ids):
        region = leaf_regions[leaf_id]
        if in_counts[i] > 0:
            samples = sample_region(
                region, feature_data, int(in_counts[i]), rng,
                categorical_features=categorical_features,
            )
            X_in_like.append(samples)
            for _ in range(samples.shape[0]):
                meta_in.append({"leaf_id": leaf_id, "support": region.support})

        if out_counts[i] > 0:
            samples, meta = _sample_out_like(
                region, feature_data, int(out_counts[i]), rng,
                categorical_features=categorical_features,
            )
            X_out_like.append(samples)
            meta_out.extend(meta)

    X_in_like = pd.concat(X_in_like, ignore_index=True) if X_in_like else pd.DataFrame()
    X_out_like = pd.concat(X_out_like, ignore_index=True) if X_out_like else pd.DataFrame()

    if verbose:
        print("Generated %d in-like and %d out-like probes." % (len(X_in_like), len(X_out_like)))
    return X_in_like, X_out_like, meta_in, meta_out


def _sample_out_like(region, feature_data, n_samples, rng, categorical_features=None):
    categorical_features = categorical_features or {}
    samples = []
    meta = []
    features = list(region.ranges.keys())

    for _ in range(n_samples):
        # start from a valid in-region sample
        base = sample_region(region, feature_data, 1, rng, categorical_features=categorical_features).iloc[0].to_dict()
        boundary_features = [f for f, b in region.ranges.items()
                             if b.get('start') is not None or b.get('end') is not None]

        if boundary_features:
            f = rng.choice(boundary_features)
            bounds = region.ranges[f]
            global_min = feature_data[f]['min']
            global_max = feature_data[f]['max']
            eps = 1e-3 * (global_max - global_min) if global_max > global_min else 1e-3

            if bounds.get('end') is not None:
                # nudge just above end boundary
                base[f] = _clamp(bounds['end'] + eps, global_min, global_max)
                direction = "above_end"
            else:
                # nudge just below start boundary
                base[f] = _clamp(bounds['start'] - eps, global_min, global_max)
                direction = "below_start"
        else:
            # if no boundaries, sample from global bounds as out-like
            for f in features:
                if f in categorical_features:
                    allowed = categorical_features[f]
                    base[f] = rng.choice(allowed)
                else:
                    global_min = feature_data[f]['min']
                    global_max = feature_data[f]['max']
                    base[f] = rng.uniform(global_min, global_max)
            direction = "global"
            f = None

        samples.append(base)
        meta.append({"leaf_id": region.leaf_id, "support": region.support, "boundary_feature": f, "boundary_mode": direction})

    return pd.DataFrame(samples), meta


def query_label(predictor, X):
    """Query a deployed predictor, returning labels only."""
    preds = predictor(X)
    return np.asarray(preds)


def _perturb_point(x, feature_data, scale, rng, categorical_features=None):
    categorical_features = categorical_features or {}
    x_new = x.copy()
    for feature, value in x.items():
        if feature in categorical_features:
            allowed = list(categorical_features[feature])
            if len(allowed) > 1 and rng.rand() < 0.5:
                choices = [v for v in allowed if v != value]
                x_new[feature] = rng.choice(choices)
        else:
            global_min = feature_data[feature]['min']
            global_max = feature_data[feature]['max']
            span = global_max - global_min
            if span == 0:
                continue
            noise = rng.normal(0.0, scale * span)
            x_new[feature] = _clamp(value + noise, global_min, global_max)
    return x_new


def score_flip_rate(predictor, X, feature_data, perturb_rounds=20,
                    noise_schedule=None, seed=0, categorical_features=None):
    """Compute flip-rate scores (lower = more robust)."""
    rng = np.random.RandomState(seed)
    noise_schedule = _noise_schedule_from_arg(noise_schedule)
    scores = []

    for _, row in X.iterrows():
        x = row.to_dict()
        y0 = query_label(predictor, pd.DataFrame([x]))[0]
        flips = 0
        for i in range(perturb_rounds):
            scale = noise_schedule[i % len(noise_schedule)]
            x_p = _perturb_point(x, feature_data, scale, rng, categorical_features=categorical_features)
            y_p = query_label(predictor, pd.DataFrame([x_p]))[0]
            if y_p != y0:
                flips += 1
        scores.append(flips / float(perturb_rounds))

    return np.asarray(scores)


def score_min_delta(predictor, X, feature_data, noise_schedule=None,
                    seed=0, categorical_features=None):
    """Compute min-delta-to-flip scores (higher = more robust)."""
    rng = np.random.RandomState(seed)
    noise_schedule = _noise_schedule_from_arg(noise_schedule)
    scores = []

    for _, row in X.iterrows():
        x = row.to_dict()
        y0 = query_label(predictor, pd.DataFrame([x]))[0]
        flipped = False
        for scale in noise_schedule:
            x_p = _perturb_point(x, feature_data, scale, rng, categorical_features=categorical_features)
            y_p = query_label(predictor, pd.DataFrame([x_p]))[0]
            if y_p != y0:
                scores.append(scale)
                flipped = True
                break
        if not flipped:
            scores.append(max(noise_schedule) * 2.0)

    return np.asarray(scores)


def evaluate_attack(scores_in, scores_out):
    """Compute AUC, best-threshold accuracy, and TPR@FPR=1%."""
    y_true = np.concatenate([np.ones_like(scores_in), np.zeros_like(scores_out)])
    y_scores = np.concatenate([scores_in, scores_out])

    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # best threshold by accuracy
    best_acc = 0.0
    best_thr = thresholds[0] if thresholds.size > 0 else 0.0
    for thr in thresholds:
        preds = (y_scores >= thr).astype(int)
        acc = (preds == y_true).mean()
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    # TPR at FPR=1%
    target_fpr = 0.01
    tpr_at_fpr = 0.0
    for i in range(len(fpr)):
        if fpr[i] <= target_fpr:
            tpr_at_fpr = max(tpr_at_fpr, tpr[i])

    return {
        "auc": float(auc),
        "best_accuracy": float(best_acc),
        "best_threshold": float(best_thr),
        "tpr_at_fpr_1pct": float(tpr_at_fpr),
    }


def audit_report_to_json(report, path):
    """Persist audit report to json."""
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


class DeployedPredictor:
    """Label-only predictor wrapper for auditing."""

    def __init__(self, generalizer, teacher=None, student=None, encoder=None,
                 features=None, feature_data=None, categorical_features=None):
        self.generalizer = generalizer
        self.teacher = teacher
        self.student = student
        self.encoder = encoder
        self.features = features
        self.feature_data = feature_data
        self.categorical_features = categorical_features

    def __call__(self, X):
        if self.student is not None:
            return predict_with_student(
                self.generalizer,
                self.student,
                X,
                self.features,
                self.feature_data,
                categorical_features=self.categorical_features,
                encoder=self.encoder,
                return_proba=False,
            )
        # fallback: teacher over generalized inputs
        X_gen = self.generalizer.transform(X)
        return self.teacher.predict(X_gen)
