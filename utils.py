from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer

import numpy as np

def rw_rmse(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    w = 1 + y_true / 0.1
    return np.sqrt(np.sum(w * (y_true - y_pred)**2) / np.sum(w))


def evaluate_feature_selection_cv(X, y, model, selected_features, print_results=False):

    X_selected = X[selected_features]

    scorer = make_scorer(rw_rmse, greater_is_better=False)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
        model,
        X_selected,
        y.values.ravel(),
        cv=kf,
        scoring=scorer
    )
    if print_results:
        print("Selected features:")
        for feature in selected_features:
            print("-", feature)

        print("----------------------------------")
        print("RW-RMSE per fold:", -scores)
        print(f"Mean RW-RMSE: {-scores.mean():.5f}")

    return -scores.mean()


def evaluate_sfs(X, y, model, direction="forward", n_features=6):

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(rw_rmse, greater_is_better=False)

    selector = SequentialFeatureSelector(
        model,
        n_features_to_select=n_features,
        direction=direction,
        scoring=scorer,
        cv=kf
    )

    pipeline = make_pipeline(
        selector,
        model
    )

    # Nested cross-validation
    scores = cross_val_score(
        pipeline,
        X,
        np.asarray(y).ravel(),
        cv=kf,
        scoring=scorer
    )

    pipeline.fit(X, np.asarray(y).ravel())

    selected_features = X.columns[
        pipeline.named_steps["sequentialfeatureselector"].get_support()
    ].tolist()

    print("Direction:", direction)
    print("Selected features:", selected_features)
    print("RW-RMSE per fold:", -scores)
    print(f"Mean RW-RMSE: {-scores.mean():.5f}")
    print("----------------------------------")

    return -scores.mean(), selected_features, pipeline