"""
This module implements data distillation on top of the generalizer.
"""
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
import joblib


@dataclass
class Region:
    """Container for a leaf region definition.

    Parameters
    ----------
    leaf_id : int
        Node id of the leaf (or internal node when generalization merges levels).
    ranges : dict
        Per-feature numeric constraints of the form
        {feature: {'start': float or None, 'end': float or None}}.
    categories : dict
        Per-feature categorical allowed sets of the form
        {feature: [allowed_values]}.
    support : int
        Number of training samples that reach this node.
    """
    leaf_id: int
    ranges: dict
    categories: dict
    support: int


def _softmax_with_temperature(probs, temperature):
    if temperature is None or temperature == 1.0:
        return probs
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    # apply temperature by scaling log-probabilities
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-12, 1.0)
    logp = np.log(probs) / temperature
    logp = logp - np.max(logp, axis=1, keepdims=True)
    exp = np.exp(logp)
    return exp / exp.sum(axis=1, keepdims=True)


def extract_leaf_regions(cells, features, feature_data, node_support,
                         categorical_features=None, min_leaf_support=1):
    """Extract a Region per leaf from the learned cells.

    Parameters
    ----------
    cells : list of dict
        Cells learned by the generalizer, each with 'id' and 'ranges'.
    features : list
        Feature names in order.
    feature_data : dict
        Per-feature stats with keys 'min' and 'max'.
    node_support : ndarray or dict
        Support counts per node id.
    categorical_features : dict, optional
        Mapping {feature: [allowed_values]} used to define categorical sets.
    min_leaf_support : int, optional
        Minimum support required to keep a leaf.

    Returns
    -------
    leaf_regions : dict
        Mapping {leaf_id: Region}.
    """
    categorical_features = categorical_features or {}
    leaf_regions = {}
    for cell in cells:
        leaf_id = cell['id']
        support = int(node_support[leaf_id]) if node_support is not None else 0
        if support < min_leaf_support:
            continue

        ranges = {}
        categories = {}
        for feature in features:
            if feature in cell['ranges']:
                ranges[feature] = {
                    'start': cell['ranges'][feature].get('start', None),
                    'end': cell['ranges'][feature].get('end', None),
                }
            else:
                ranges[feature] = {'start': None, 'end': None}

            if feature in categorical_features:
                categories[feature] = list(categorical_features[feature])

        leaf_regions[leaf_id] = Region(
            leaf_id=leaf_id,
            ranges=ranges,
            categories=categories,
            support=support,
        )
    return leaf_regions


def sample_region(region, feature_data, n_samples, rng, categorical_features=None):
    """Generate synthetic samples within a region.

    Parameters
    ----------
    region : Region
        Region containing per-feature constraints.
    feature_data : dict
        Per-feature numeric bounds.
    n_samples : int
        Number of synthetic points to generate.
    rng : numpy.random.RandomState
        Random state used for sampling.
    categorical_features : dict, optional
        Mapping {feature: [allowed_values]} with global categorical bounds.

    Returns
    -------
    samples : pandas.DataFrame
        Synthetic samples within the region, with feature names.
    """
    categorical_features = categorical_features or {}
    rows = {}
    for feature, bounds in region.ranges.items():
        if feature in categorical_features:
            allowed = region.categories.get(feature, categorical_features[feature])
            rows[feature] = rng.choice(allowed, size=n_samples, replace=True)
            continue

        if feature not in feature_data:
            raise ValueError("Missing feature_data for numeric feature: %s" % feature)

        global_min = feature_data[feature]['min']
        global_max = feature_data[feature]['max']
        start = bounds.get('start', None)
        end = bounds.get('end', None)

        low = global_min if start is None else max(global_min, start)
        high = global_max if end is None else min(global_max, end)
        if low == high:
            rows[feature] = np.full(n_samples, low)
        else:
            rows[feature] = rng.uniform(low=low, high=high, size=n_samples)

    return pd.DataFrame(rows)


def build_leaf_dataset(leaf_regions, leaf_representatives, teacher,
                       feature_data, samples_per_leaf=50, temperature=1.0,
                       seed=0, categorical_features=None, verbose=True):
    """Generate a distilled dataset for training a student.

    Parameters
    ----------
    leaf_regions : dict
        Mapping {leaf_id: Region}.
    leaf_representatives : dict
        Mapping {leaf_id: representative dict of feature -> value}.
    teacher : estimator
        Teacher model with predict_proba or predict.
    feature_data : dict
        Per-feature numeric bounds.
    samples_per_leaf : int, optional
        Number of synthetic samples per leaf.
    temperature : float, optional
        Temperature for probability smoothing.
    seed : int, optional
        Random seed.
    categorical_features : dict, optional
        Global categorical bounds.
    verbose : bool, optional
        If True, prints progress.

    Returns
    -------
    leaf_dataset : list
        List of tuples (leaf_repr, p_leaf, leaf_id, support).
    """
    rng = np.random.RandomState(seed)
    leaf_dataset = []
    for leaf_id, region in leaf_regions.items():
        if verbose:
            print("Sampling leaf %d (support=%d)..." % (leaf_id, region.support))

        X_syn = sample_region(
            region, feature_data, samples_per_leaf, rng,
            categorical_features=categorical_features,
        )

        if hasattr(teacher, "predict_proba"):
            probs = teacher.predict_proba(X_syn)
        else:
            labels = teacher.predict(X_syn)
            n_classes = int(np.max(labels)) + 1
            probs = np.eye(n_classes)[labels]

        probs = _softmax_with_temperature(probs, temperature)
        p_leaf = probs.mean(axis=0)

        if leaf_id not in leaf_representatives:
            raise ValueError("Missing representative for leaf_id: %s" % leaf_id)
        leaf_repr = leaf_representatives[leaf_id]

        leaf_dataset.append((leaf_repr, p_leaf, leaf_id, region.support))

    if verbose:
        print("Built leaf dataset with %d leaves." % len(leaf_dataset))
    return leaf_dataset


def _build_feature_matrix(leaf_dataset, features, feature_data,
                          categorical_features=None, encoder=None, fit_encoder=False):
    categorical_features = categorical_features or {}
    rows = []
    for leaf_repr, _, _, _ in leaf_dataset:
        row = {}
        for feature in features:
            value = leaf_repr.get(feature, None)
            if value is None and feature in feature_data:
                value = (feature_data[feature]['min'] + feature_data[feature]['max']) / 2.0
            row[feature] = value
        rows.append(row)

    df = pd.DataFrame(rows, columns=features)
    cat_cols = [f for f in features if f in categorical_features]
    num_cols = [f for f in features if f not in categorical_features]

    X_num = df[num_cols].astype(float).to_numpy() if num_cols else None

    if cat_cols:
        if encoder is None:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        if fit_encoder:
            X_cat = encoder.fit_transform(df[cat_cols])
        else:
            X_cat = encoder.transform(df[cat_cols])
    else:
        X_cat = None

    if X_num is None:
        X = X_cat
    elif X_cat is None:
        X = X_num
    else:
        X = np.concatenate([X_num, X_cat], axis=1)
    return X, encoder


def train_student(leaf_dataset, features, feature_data, student_type="logreg",
                  categorical_features=None, seed=0, verbose=True):
    """Train a student model to mimic per-leaf teacher distributions.

    Parameters
    ----------
    leaf_dataset : list
        List of tuples (leaf_repr, p_leaf, leaf_id, support).
    features : list
        Feature names in order.
    feature_data : dict
        Per-feature numeric bounds used for filling missing values.
    student_type : {'logreg', 'mlp'}, optional
        Student model type.
    categorical_features : dict, optional
        Global categorical bounds.
    seed : int, optional
        Random seed.
    verbose : bool, optional
        If True, prints training status.

    Returns
    -------
    student : estimator
        Trained student model.
    encoder : OneHotEncoder or None
        Encoder used for categorical features.
    metrics : dict
        Agreement metrics between student and teacher distributions.
    """
    X, encoder = _build_feature_matrix(
        leaf_dataset, features, feature_data,
        categorical_features=categorical_features,
        encoder=None, fit_encoder=True,
    )
    y_probs = np.vstack([p for _, p, _, _ in leaf_dataset])
    y_labels = np.argmax(y_probs, axis=1)
    rng = np.random.RandomState(seed)

    unique_labels = np.unique(y_labels)
    if unique_labels.size < 2:
        # If the argmax collapses to one class but probabilities have mass
        # on multiple classes, create pseudo-labels by sampling from p_leaf.
        class_mass = y_probs.sum(axis=0)
        if (class_mass > 0).sum() >= 2:
            if verbose:
                print("Only one argmax class in leaf dataset; generating pseudo-labels.")
            expanded_reprs = []
            expanded_labels = []
            n_classes = y_probs.shape[1]
            for (leaf_repr, p_leaf, _, support) in leaf_dataset:
                n = int(min(200, max(5, support)))
                labels = rng.choice(n_classes, size=n, p=p_leaf)
                expanded_reprs.extend([leaf_repr] * n)
                expanded_labels.extend(labels.tolist())
            expanded_dataset = [(r, None, None, None) for r in expanded_reprs]
            X, encoder = _build_feature_matrix(
                expanded_dataset, features, feature_data,
                categorical_features=categorical_features,
                encoder=encoder, fit_encoder=False,
            )
            y_labels = np.asarray(expanded_labels, dtype=int)
            unique_labels = np.unique(y_labels)
        else:
            if verbose:
                print("Only one class in leaf dataset; using DummyClassifier.")
            student = DummyClassifier(strategy="constant", constant=unique_labels[0])
            student.fit(X, y_labels)
            if hasattr(student, "predict_proba"):
                student_probs = student.predict_proba(X)
            else:
                pred_labels = student.predict(X)
                n_classes = y_probs.shape[1]
                student_probs = np.eye(n_classes)[pred_labels]
            acc = accuracy_score(y_labels, np.argmax(student_probs, axis=1))
            l1 = np.mean(np.abs(student_probs - y_probs[:student_probs.shape[0]]))
            metrics = {
                "agreement_accuracy": acc,
                "mean_l1_distance": l1,
            }
            if verbose:
                print("Student agreement accuracy: %.4f" % acc)
                print("Student mean L1 distance: %.4f" % l1)
            return student, encoder, metrics

    if student_type == "logreg":
        student = LogisticRegression(max_iter=1000, random_state=seed)
    elif student_type == "mlp":
        student = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=seed)
    else:
        raise ValueError("Unknown student_type: %s" % student_type)

    student.fit(X, y_labels)

    n_classes = y_probs.shape[1]
    if hasattr(student, "predict_proba"):
        raw_probs = student.predict_proba(X)
        # align to full class set in case some classes are missing
        student_probs = np.zeros((raw_probs.shape[0], n_classes), dtype=float)
        if hasattr(student, "classes_"):
            for i, cls in enumerate(student.classes_):
                if cls < n_classes:
                    student_probs[:, int(cls)] = raw_probs[:, i]
        else:
            # fallback assumes raw_probs already matches class order
            student_probs[:, :raw_probs.shape[1]] = raw_probs
    else:
        pred_labels = student.predict(X)
        student_probs = np.eye(n_classes)[pred_labels]

    acc = accuracy_score(y_labels, np.argmax(student_probs, axis=1))
    if student_probs.shape[0] == y_probs.shape[0]:
        l1 = np.mean(np.abs(student_probs - y_probs))
    else:
        l1 = float("nan")
        if verbose:
            print("Mean L1 distance skipped (expanded dataset for pseudo-labels).")
    metrics = {
        "agreement_accuracy": acc,
        "mean_l1_distance": l1,
    }

    if verbose:
        print("Student agreement accuracy: %.4f" % acc)
        print("Student mean L1 distance: %.4f" % l1)

    return student, encoder, metrics


def save_student(student, encoder, student_path, encoder_path):
    """Persist student and encoder to disk."""
    joblib.dump(student, student_path)
    joblib.dump(encoder, encoder_path)


def load_student(student_path, encoder_path):
    """Load student and encoder from disk."""
    student = joblib.load(student_path)
    encoder = joblib.load(encoder_path)
    return student, encoder


def predict_with_student(generalizer, student, X, features, feature_data,
                         categorical_features=None, encoder=None, return_proba=False):
    """Run the inference swap: generalize -> student -> prediction.

    Parameters
    ----------
    generalizer : GeneralizeToRepresentative
        Fitted generalizer.
    student : estimator
        Trained student model.
    X : array-like or DataFrame
        Raw input data.
    features : list
        Feature names in order.
    feature_data : dict
        Per-feature numeric bounds.
    categorical_features : dict, optional
        Global categorical bounds.
    encoder : OneHotEncoder, optional
        Encoder used during training.
    return_proba : bool, optional
        If True, returns probabilities if available.

    Returns
    -------
    preds : ndarray
        Predictions or probabilities.
    """
    X_gen = generalizer.transform(X)
    # X_gen is DataFrame if X is DataFrame
    if isinstance(X_gen, pd.DataFrame):
        leaf_dataset = [({f: X_gen.iloc[i][f] for f in features}, None, None, None)
                        for i in range(X_gen.shape[0])]
    else:
        X_gen = np.asarray(X_gen)
        leaf_dataset = [({features[j]: X_gen[i, j] for j in range(len(features))},
                         None, None, None)
                        for i in range(X_gen.shape[0])]

    X_student, _ = _build_feature_matrix(
        leaf_dataset, features, feature_data,
        categorical_features=categorical_features,
        encoder=encoder, fit_encoder=False,
    )

    if return_proba and hasattr(student, "predict_proba"):
        return student.predict_proba(X_student)
    return student.predict(X_student)
