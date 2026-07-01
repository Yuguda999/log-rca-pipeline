import joblib
import numpy as np

from . import config


class RootCauseClassifier:
    def __init__(self, model_path=config.MODEL_PATH):
        self.pipe = joblib.load(model_path)
        self.classes_ = self.pipe.classes_

    def predict(self, messages):
        if isinstance(messages, str):
            messages = [messages]
        probs = self.pipe.predict_proba(messages)
        idx = probs.argmax(axis=1)
        results = []
        for i, j in enumerate(idx):
            conf = float(probs[i, j])
            results.append(
                {
                    "log_message": messages[i],
                    "predicted_label": str(self.classes_[j]),
                    "confidence": round(conf, 4),
                    "needs_review": conf < config.CONFIDENCE_THRESHOLD,
                }
            )
        return results


if __name__ == "__main__":
    clf = RootCauseClassifier()
    for r in clf.predict("ERROR [db-pool] all 15 connections exhausted. timeout after 9000ms."):
        print(r)
