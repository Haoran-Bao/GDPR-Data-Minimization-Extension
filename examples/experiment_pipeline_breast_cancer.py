"""
End-to-end pipeline experiment on the breast cancer dataset:
generalize -> distill -> audit.
"""
import argparse

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from minimization import GeneralizeToRepresentative
from minimization.audit import (
    DeployedPredictor,
    generate_probes,
    score_flip_rate,
    score_min_delta,
    evaluate_attack,
    audit_report_to_json,
)


def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline on breast cancer.")
    parser.add_argument("--samples_per_leaf", type=int, default=50)
    parser.add_argument("--student_type", choices=["logreg", "mlp"], default="logreg")
    parser.add_argument("--min_leaf_support", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n_in", type=int, default=200)
    parser.add_argument("--n_out", type=int, default=200)
    parser.add_argument("--perturb_rounds", type=int, default=20)
    parser.add_argument("--noise_schedule", type=str, default="0.01,0.02,0.05,0.1")
    parser.add_argument("--score", choices=["flip_rate", "min_delta"], default="flip_rate")
    parser.add_argument("--pass_auc", type=float, default=0.55)
    parser.add_argument("--pass_tpr_at_fpr", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    base_est = DecisionTreeClassifier(random_state=0)
    base_est.fit(X_train, y_train)
    train_preds = base_est.predict(X_train)

    gen = GeneralizeToRepresentative(
        estimator=base_est,
        target_accuracy=0.9,
        features=list(X.columns),
    )
    gen.fit(X_train, train_preds)

    student, encoder, metrics = gen.distill_teacher(
        samples_per_leaf=args.samples_per_leaf,
        student_type=args.student_type,
        min_leaf_support=args.min_leaf_support,
        temperature=args.temperature,
        seed=args.seed,
        categorical_features=None,
        verbose=True,
    )

    leaf_regions = gen.get_leaf_regions(min_leaf_support=args.min_leaf_support)
    X_in_like, X_out_like, meta_in, meta_out = generate_probes(
        leaf_regions,
        gen.feature_data_,
        n_in=args.n_in,
        n_out=args.n_out,
        seed=args.seed,
        categorical_features=None,
        prioritize_small_leaves=True,
        verbose=True,
    )

    predictor = DeployedPredictor(
        generalizer=gen,
        teacher=base_est,
        student=student,
        encoder=encoder,
        features=list(X.columns),
        feature_data=gen.feature_data_,
        categorical_features=None,
    )

    if args.score == "flip_rate":
        scores_in = 1.0 - score_flip_rate(
            predictor,
            X_in_like,
            gen.feature_data_,
            perturb_rounds=args.perturb_rounds,
            noise_schedule=args.noise_schedule,
            seed=args.seed,
            categorical_features=None,
        )
        scores_out = 1.0 - score_flip_rate(
            predictor,
            X_out_like,
            gen.feature_data_,
            perturb_rounds=args.perturb_rounds,
            noise_schedule=args.noise_schedule,
            seed=args.seed + 1,
            categorical_features=None,
        )
    else:
        scores_in = score_min_delta(
            predictor,
            X_in_like,
            gen.feature_data_,
            noise_schedule=args.noise_schedule,
            seed=args.seed,
            categorical_features=None,
        )
        scores_out = score_min_delta(
            predictor,
            X_out_like,
            gen.feature_data_,
            noise_schedule=args.noise_schedule,
            seed=args.seed + 1,
            categorical_features=None,
        )

    metrics = evaluate_attack(scores_in, scores_out)
    passed = (metrics["auc"] <= args.pass_auc) and (metrics["tpr_at_fpr_1pct"] <= args.pass_tpr_at_fpr)

    report = {
        "dataset": "breast_cancer",
        "config": vars(args),
        "metrics": metrics,
        "pass": bool(passed),
        "n_in": int(len(scores_in)),
        "n_out": int(len(scores_out)),
    }
    audit_report_to_json(report, "audit_report_breast_cancer.json")

    print("AUC: %.4f" % metrics["auc"])
    print("Best attack accuracy: %.4f" % metrics["best_accuracy"])
    print("TPR@FPR=1%%: %.4f" % metrics["tpr_at_fpr_1pct"])
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
