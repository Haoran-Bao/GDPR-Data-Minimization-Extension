"""
Experiment script for GeneralizeToRepresentative.

Runs a simple end-to-end flow and prints:
1) base model accuracy on original test data
2) base model accuracy on generalized test data
3) sample of original vs generalized records with feature names
"""
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from minimization import GeneralizeToRepresentative


def main():
    # Load dataset with feature names
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Train a base estimator
    base_est = DecisionTreeClassifier(random_state=0)
    base_est.fit(X_train, y_train)

    # Fit the generalizer on training data + model predictions
    train_preds = base_est.predict(X_train)
    gen = GeneralizeToRepresentative(
        estimator=base_est,
        target_accuracy=0.9,
        features=list(X.columns),
    )
    gen.fit(X_train, train_preds)

    # Evaluate: base model on original vs generalized data
    base_acc = base_est.score(X_test, y_test)
    X_test_gen = gen.transform(X_test)
    gen_acc = base_est.score(X_test_gen, y_test)

    print("Base accuracy on original X_test: %.4f" % base_acc)
    print("Base accuracy on generalized X_test: %.4f" % gen_acc)

    # Show a small before/after sample
    sample_idx = np.random.RandomState(0).choice(X_test.shape[0], size=5, replace=False)
    original_sample = X_test.iloc[sample_idx]
    generalized_sample = X_test_gen.iloc[sample_idx]

    print("\nOriginal sample:")
    print(original_sample)
    print("\nGeneralized sample:")
    print(generalized_sample)


if __name__ == "__main__":
    main()
