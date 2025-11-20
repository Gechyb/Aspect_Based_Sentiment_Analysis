from typing import List
import sklearn_crfsuite


class CRFTagger:
    def __init__(self, c1=0.1, c2=0.1, max_iterations=200):
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X) -> List[List[str]]:
        return self.model.predict(X)
