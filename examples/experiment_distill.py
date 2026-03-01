"""
Experiment script for data distillation on top of GeneralizeToRepresentative.
"""
import argparse
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from minimization import GeneralizeToRepresentative
from minimization.distillation import save_student, predict_with_student


def main():
    parser = argparse.ArgumentParser(description="Data distillation experiment.")
    parser.add_argument("--samples_per_leaf", type=int, default=50)
    parser.add_argument("--student_type", choices=["logreg", "mlp"], default="logreg")
    parser.add_argument("--min_leaf_support", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
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

    save_student(student, encoder, "student.pkl", "encoder.pkl")

    # inference swap evaluation
    student_preds = predict_with_student(
        gen, student, X_test, list(X.columns), gen.feature_data_,
        categorical_features=None, encoder=encoder, return_proba=False,
    )

    teacher_preds = base_est.predict(X_test)
    agreement = (student_preds == teacher_preds).mean()
    print("Teacher-student agreement on X_test: %.4f" % agreement)


if __name__ == "__main__":
    main()
