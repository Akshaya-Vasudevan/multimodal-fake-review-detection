from sklearn.feature_extraction.text import TfidfVectorizer

class TextPipeline:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english"
        )

    def preprocess(self, df):
        texts = df["reviewText"].fillna("")
        return texts.values

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)
