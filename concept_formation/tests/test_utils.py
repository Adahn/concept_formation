from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

import random

from concept_formation import utils
from concept_formation.cobweb3 import ContinuousValue


def test_cv_mean():
    for i in range(10):
        values = [random.normalvariate(0, 1) for i in range(100)]
        cv = ContinuousValue()
        cv.update_batch(values)
        assert cv.mean - utils.mean(values) < 0.00000000001


def test_cv_std():
    for i in range(10):
        values = [random.normalvariate(0, 1) for i in range(100)]
        cv = ContinuousValue()
        cv.update_batch(values)
        assert cv.biased_std() - utils.std(values) < 0.00000000001


def test_cv_unbiased_std():
    for i in range(10):
        values = [random.normalvariate(0, 1) for i in range(10)]
        cv = ContinuousValue()
        cv.update_batch(values)
        assert (cv.unbiased_std() -
                ((len(values) * utils.std(values) / (len(values) - 1)) /
                 utils.c4(len(values))) < 0.00000000001)


def test_cv_update():
    for i in range(10):
        values = []
        cv = ContinuousValue()
        for i in range(20):
            x = random.normalvariate(0, 1)
            values.append(x)
            cv.update(x)
            assert cv.biased_std() - utils.std(values) < 0.00000000001


def test_cv_combine():
    for i in range(10):
        values1 = [random.normalvariate(0, 1) for i in range(50)]
        values2 = [random.normalvariate(0, 1) for i in range(50)]
        values = values1 + values2
        cv = ContinuousValue()
        cv2 = ContinuousValue()

        cv.update_batch(values1)
        assert cv.biased_std() - utils.std(values1) < 0.00000000001

        cv2.update_batch(values2)
        assert cv2.biased_std() - utils.std(values2) < 0.00000000001

        cv.combine(cv2)
        assert cv.biased_std() - utils.std(values) < 0.00000000001
