import pandas as pd

from . import config


def load_logs() -> pd.DataFrame:
    df = pd.read_csv(config.LOGS_CSV)
    required = {config.TEXT_COL, config.LABEL_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"logs.csv missing columns: {missing}")

    df = df.dropna(subset=[config.TEXT_COL, config.LABEL_COL]).reset_index(drop=True)

    # `service` is dropped deliberately: in 90% of rows it does not match the
    # component named in the message, so it is noise for root-cause prediction.
    return df[[config.TEXT_COL, config.LABEL_COL]]


def load_label_catalog() -> dict:
    df = pd.read_csv(config.LABELS_CSV)
    return {
        row["id"]: {
            "label": row["label"],
            "description": row["description"],
            "severity": row["severity"],
            "typical_resolution": row["typical_resolution"],
        }
        for _, row in df.iterrows()
    }
