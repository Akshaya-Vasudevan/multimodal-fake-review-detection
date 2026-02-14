from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from scipy.sparse import hstack
from scipy.sparse import csr_matrix

class FusionModel:
    def __init__(self):
        self.model = SGDClassifier(
            loss="log_loss",   # logistic regression behavior
            max_iter=1000,
            n_jobs=-1          # use all CPU cores
        )

    def fuse_features(self, text_features, metadata_features):
        meta_sparse = csr_matrix(metadata_features)
        return hstack([text_features, meta_sparse])

    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        predictions = self.model.predict(X)
        print(classification_report(y, predictions))
