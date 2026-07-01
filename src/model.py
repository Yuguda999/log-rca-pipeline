from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .preprocess import normalize


def build_pipeline() -> Pipeline:
    word = TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    char = TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    features = FeatureUnion([("word", word), ("char", char)])
    clf = LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])
