from typing import Iterable, List
from string_similarity_jaro import JaroOriginalMetric, JaroMetric, JaroWinklerMetric


class PredictedWordFreq:
    def __init__(self, word: str, count: int, score: float):
        self._word = word
        self._count = count
        self._total_score = score
        self._mean_score: float = 0.0
        self._calc_mean_score()

    @property
    def word(self):
        return self._word

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, value: int):
        if not isinstance(value, int):
            raise ValueError("PredictedWordFreq object - value to incr Count must be an integer.")
        if value < 1:
            raise ValueError("PredictedWordFreq object - value to incr Count must be a positive integer.")

        # print(f"**************************** {value}")
        self._count = value

    @property
    def total_score(self):
        return self._total_score

    @total_score.setter
    def total_score(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("PredictedWordFreq object - value to incr Score must be a number.")
        self._total_score = value
        self._calc_mean_score()

    @property
    def mean_score(self):
        return self._mean_score

    def _calc_mean_score(self):
        self._mean_score = self._total_score / self._count

    def __eq__(self, other):
        return round(self._count, 3) == round(other.count, 3)

    def __lt__(self, other):
        return round(self._count, 3) < round(other.count, 3)

    def __str__(self):
        return f"word: '{self._word}' | count: '{self._count}' | mean_score: {self._mean_score:0.3f}"


class StringSimilarityEnsemble:
    def __init__(self, algorithms: list, targets: Iterable[str], search_text: str, threshold: float, top_n: int = 1):

        self._algorithms = algorithms
        self._targets = targets
        self._search_text = search_text
        self._threshold = threshold
        self._top_n = top_n

        # Each algo will give only their best prediction.
        # If the ensemble predicts > 1 word, and if top_n == 1, only the word with the highest counts is returned
        # if the ensemble predicts > 1 word, and if top_n == 2, two words with the highest and second counts
        # if the ensemble predicts > 1 word, and if top_n == 0, all words predicted by algo are returned
        # if top_n is negative, same rules as python negative indexing
        # mean score is returned
        self._prediction_objects = \
            [algo(self._targets, self._search_text, self._threshold, top_n=1) for algo in algorithms]
        self._predictions = []  # holds predictions by individual algorithm
        self._majority_predictions = []  # holds the results of the majority voting by each algorithm

    @property
    def predictions(self):
        return self._predictions

    @property
    def majority_predictions(self):
        return self._majority_predictions

    def _predict(self):
        self._predictions = [p_obj.predict(p_obj.algorithm) for p_obj in self._prediction_objects]
        self._predictions = \
            sorted([p_obj_lst[0] for p_obj_lst in self._predictions if len(p_obj_lst) > 0], reverse=True)

    def predict_by_majority_voting(self) -> List[tuple[str, int, float]]:
        """

        :return:
            top_n:
                (word, count, mean_score)
        """
        self._predict()
        for p in self._predictions:
            print(p)
        print('\n', '*'*80, '\n')

        for p in self._predictions:
            # word_found = False
            word = p.word.strip().lower()
            score = p.score
            for mp in self._majority_predictions:
                if mp.word == word:
                    # print("word already exist in mp")
                    mp.count += 1
                    mp.total_score += score
                    word_found = True
                    break
            else:
                # if above 'break', this section will not run.
                # print("word NOT exist in mp, append new word to mp")
                self._majority_predictions.append(PredictedWordFreq(word, 1, score))

        for mp in self._majority_predictions:
            print(mp)
            print('\t\t\t ------------------')

        # predictions = sorted([p[0] for p in predictions if len(p) > 0], reverse=True)
        self._majority_predictions = sorted(self._majority_predictions, reverse=True)
        if self._top_n > 0:
            return self._majority_predictions[:self._top_n]
        elif self._top_n < 0:
            return self._majority_predictions[self._top_n:]
        else:
            return self._majority_predictions


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
        "sales manager",
        "sales executive"
    ]
    SEARCH_TEXT = 'sales director '

    ALGORITHMS = [JaroOriginalMetric, JaroMetric, JaroWinklerMetric]
    sse = StringSimilarityEnsemble(ALGORITHMS, JOB_ROLES, SEARCH_TEXT, threshold=0.5, top_n=1)

    preds = sse.predict_by_majority_voting()
    print(f"\n~~~~~~~ For the word: {SEARCH_TEXT},")
    for p in preds:
        print(f"the majority voted for: {p.word}, count: {p.count}, mean_score: {p.mean_score:0.3f}")

