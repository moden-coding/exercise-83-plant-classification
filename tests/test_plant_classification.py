#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock, patch

import sklearn
from sklearn.model_selection import train_test_split

from src.plant_classification import plant_classification


def spy_decorator(method_to_decorate, name):
    """
    Wrap a method so calls to it are recorded on a MagicMock while the
    original implementation still runs.

    This solution to wrap a patched method without obstructing its
    implementation comes originally from
    https://stackoverflow.com/questions/25608107/
    """
    mock = MagicMock(name="%s method" % name)

    def wrapper(self, *args, **kwargs):
        mock(*args, **kwargs)
        return method_to_decorate(self, *args, **kwargs)
    wrapper.mock = mock
    return wrapper


class TestPlantClassification(unittest.TestCase):

    def test_correctness(self):
        acc = plant_classification()
        self.assertAlmostEqual(
            acc, 0.966667, places=5,
            msg="Incorrect accuracy score! Expected approximately "
                "0.966667 for the iris dataset. Got %r." % (acc,))

    def test_accuracy_called(self):
        score_method = spy_decorator(
            sklearn.naive_bayes.GaussianNB.score, "score")
        with patch("src.plant_classification.accuracy_score",
                   side_effect=sklearn.metrics.accuracy_score) as accuracy, \
             patch.object(sklearn.naive_bayes.GaussianNB, "score",
                          new=score_method):
            plant_classification()
            try:
                score_method.assert_called_once()
            except AssertionError:
                accuracy.assert_called_once()

    def test_third(self):
        with patch("src.plant_classification.train_test_split",
                   side_effect=train_test_split) as split:
            plant_classification()
            split.assert_called_once()
            args, kwargs = split.call_args
            self.assertIn(
                'random_state', kwargs,
                msg="You did not give the random_state argument to "
                    "train_test_split!")
            self.assertEqual(
                kwargs['random_state'], 0,
                msg="Incorrect random_state argument passed to "
                    "train_test_split! Expected 0.")
            if "test_size" in kwargs:
                self.assertEqual(
                    kwargs["test_size"], 0.2,
                    msg="Incorrect test_size argument passed to "
                        "train_test_split! Expected 0.2.")
            else:
                self.assertIn(
                    'train_size', kwargs,
                    msg="You did not give the train_size argument to "
                        "train_test_split!")
                self.assertEqual(
                    kwargs['train_size'], 0.8,
                    msg="Incorrect train_size argument passed to "
                        "train_test_split! Expected 0.8.")

    def test_gaussian(self):
        predict_method = spy_decorator(
            sklearn.naive_bayes.GaussianNB.predict, "predict")
        fit_method = spy_decorator(sklearn.naive_bayes.GaussianNB.fit, "fit")

        with patch.object(sklearn.naive_bayes.GaussianNB, "fit",
                          new=fit_method), \
             patch.object(sklearn.naive_bayes.GaussianNB, "predict",
                          new=predict_method), \
             patch("src.plant_classification.naive_bayes.GaussianNB",
                   wraps=sklearn.naive_bayes.GaussianNB) as mock_gaussian:
            plant_classification()
            mock_gaussian.assert_called_once()

            # Check that fit and predict methods of GaussianNB object are
            # called
            predict_method.mock.assert_called()
            fit_method.mock.assert_called()


if __name__ == '__main__':
    unittest.main()
