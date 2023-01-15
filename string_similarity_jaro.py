"""
Software Dev: Richard Chai

"""

from typing import Iterable
from abc_StringSimilarity import StringSimilarity
import jaro


class JaroMetric(StringSimilarity):
    algorithm = 'jaro standard'

    def __init__(self, targets: Iterable[str], search_text: str, threshold: float = 0.8, top_n: int = 0):
        super().__init__(targets, search_text, threshold, top_n)

    def _prediction_fn(self, tgt):
        return jaro.jaro_metric(tgt, self._search_text)


class JaroWinklerMetric(StringSimilarity):
    algorithm = 'jaro winkler'

    def __init__(self, targets: Iterable[str], search_text: str, threshold: float = 0.8, top_n: int = 0):
        super().__init__(targets, search_text, threshold, top_n)

    def _prediction_fn(self, tgt):
        return jaro.jaro_winkler_metric(tgt, self._search_text)


class JaroOriginalMetric(StringSimilarity):
    algorithm = 'jaro original'

    def __init__(self, targets: Iterable[str], search_text: str, threshold: float = 0.8, top_n: int = 0):
        super().__init__(targets, search_text, threshold, top_n)

    def _prediction_fn(self, tgt):
        return jaro.original_metric(tgt, self._search_text)


class JaroCustomMetric(StringSimilarity):
    # https://pypi.org/project/jaro-winkler/
    algorithm = 'jaro custom'

    def __init__(self, targets: Iterable[str], search_text: str, threshold: float, top_n: int,
                 typo_table, typo_scale, boost_threshold, pre_len, pre_scale, longer_prob):
        super().__init__(targets, search_text, threshold, top_n)
        self._typo_table = typo_table
        self._typo_scale = typo_scale
        self._boost_threshold = boost_threshold
        self._pre_len = pre_len
        self._pre_scale = pre_scale
        self._longer_prob = longer_prob

    def _prediction_fn(self, tgt) -> float:
        raise NotImplementedError


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
    SEARCH_TEXT = 'application'

    # single algorithm
    jm = JaroMetric(JOB_ROLES, SEARCH_TEXT, threshold=0.7, top_n=3)
    predictions = jm.predict(JaroMetric.algorithm)
    for p in predictions:
        print(p)

    print('\nEnsemble')
    print('*' * 50)
    str_sim = [
        JaroMetric(JOB_ROLES, SEARCH_TEXT, threshold=0.8, top_n=1),
        JaroOriginalMetric(JOB_ROLES, SEARCH_TEXT, threshold=0.8, top_n=1),
        JaroWinklerMetric(JOB_ROLES, SEARCH_TEXT, threshold=0.8, top_n=1)
    ]

    predictions = [algo.predict(algo.algorithm) for algo in str_sim]
    predictions = sorted([p[0] for p in predictions if len(p) > 0], reverse=True)

    total_score = 0.0
    total_predictions = 0

    for p in predictions:
        print(p)
