import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config
from .data_loader import load_logs
from .model import build_pipeline


def train(save: bool = True):
    df = load_logs()
    X = df[config.TEXT_COL].values
    y = df[config.LABEL_COL].values

    pipe = build_pipeline()

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    oof_pred = cross_val_predict(pipe, X, y, cv=cv)

    pipe.fit(X, y)

    if save:
        joblib.dump(pipe, config.MODEL_PATH)
        print(f"saved model -> {config.MODEL_PATH}")

    return pipe, y, oof_pred


if __name__ == "__main__":
    train()
