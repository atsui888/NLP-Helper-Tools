"""
Software Dev: Richard Chai

"""

from abc import (ABC, abstractmethod)
from typing import List, Iterable
from dataclasses import dataclass
# https://www.freecodecamp.org/news/python-property-decorator/
# https://pypi.org/project/jaro-winkler/


@dataclass
class WordPrediction:
    _algorithm: str
    _word: str
    _score: float

    @property
    def algorithm(self):
        return self._algorithm

    @property
    def word(self):
        return self._word

    @property
    def score(self):
        return self._score

    def __eq__(self, other):
        return round(self._score, 3) == round(other.score, 3)

    def __lt__(self, other):
        return round(self._score, 3) < round(other.score, 3)

    def __str__(self):
        return f"algorithm: '{self._algorithm}' | predict: '{self._word}' | score: {self._score:0.3f}"


class StringSimilarity(ABC):
    def __init__(self, targets: Iterable[str], search_text: str, threshold: float, top_n: int):
        """
            It is the responsibility of the calling function to lowercase the targets and search_text
            if required
        :param targets: an iterable of strings
        :param search_text: a string to match against the targets
        :param threshold: If scores > threshold, we append the score to the _predictions.
        :param top_n: e.g. if top_n is -3, then we want the last 3 results sorted by score. if top_n is 3,
                            we want the top 3 results (by score). If 0, return all available predictions.
        """

        self._targets: Iterable[str] = targets
        self._search_text: str = search_text
        self._threshold: float = threshold
        self._top_n: int = top_n
        self._predictions: List[WordPrediction] = []

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, new_targets: Iterable[str]):
        if isinstance(new_targets, (list, tuple)):
            self._targets = new_targets

    @property
    def search_text(self):
        return self._search_text

    @search_text.setter
    def search_text(self, new_search_text: str):
        if isinstance(new_search_text, str):
            self._search_text = new_search_text

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, new_threshold: float):
        if isinstance(new_threshold, float) and new_threshold >= 0:
            # if new_threshold is 0, all predictions are accepted subject to top_n (sorted by score)
            self._threshold = new_threshold

    @property
    def top_n(self):
        return self._top_n

    @top_n.setter
    def top_n(self, new_top_n: int):
        if isinstance(new_top_n, int):
            self._top_n = new_top_n

    # @top_n.deleter
    # def top_n(self):
    #     del self._top_n

    @property
    def predictions(self):
        # no setter is allowed
        return self._predictions

    def predict(self, algorithm: str) -> List[WordPrediction]:
        if round(self._threshold, 1) == 0.0:
            self._predictions = \
                [WordPrediction(algorithm, tgt, self._prediction_fn(tgt)) for tgt in self._targets]
        else:
            for tgt in self._targets:
                score = self._prediction_fn(tgt)
                if score > self._threshold:
                    self._predictions.append(WordPrediction(algorithm, tgt, score))

        self._predictions = sorted(self._predictions, reverse=True)

        if self._top_n > 0:
            return self._predictions[:self._top_n]
        elif self._top_n < 0:
            return self._predictions[self._top_n:]
        else:
            return self._predictions

    @abstractmethod
    def _prediction_fn(self, tgt) -> float:
        """
        Your code for your prediction algorithm goes here.

        :param tgt:
        :return: a score (data type: float)
        """

        pass


if __name__ == "__main__":
    # notes: it is the responsibility of the calling function to lowercase the targets and search_text
    # if required. This is a deliberate design choice because in some industries/fields, certain terms
    # are mixed case, and it is important to match the exact casing.
    JOB_ROLES = [
        "application developer",
        "application developer - back end",
        "application developer - front end",
        "application developer - full stack",
        "manager",
        "marketing manager",
        "managing director",
        "SALES Manager"
    ]
    SEARCH_TEXT = 'application dev full stack'

    import jaro

    class JaroMetric(StringSimilarity):
        algorithm = 'Jaro Metric'

        def __init__(self, targets: Iterable[str], search_text: str, threshold: float = 0.8, top_n: int = 0):
            super().__init__(targets, search_text, threshold, top_n)

        def _prediction_fn(self, tgt) -> float:
            return jaro.jaro_metric(tgt, self._search_text)


    jm = JaroMetric(JOB_ROLES, SEARCH_TEXT, threshold=0.7, top_n=-2)
    predictions = jm.predict(JaroMetric.algorithm)
    for p in predictions:
        print(p)
