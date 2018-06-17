Evaluations
===========

For each test instance, we will expect you to return a set of 10 choices (candidate ids) from the set of possible follow-up sentences and a probability distribution over those 10 choices.
For the competition metric we will consider the choices that cover 90% of the distribution, and compute an F-score as the harmonic mean of precision and recall:

.. math::

    Precision &= \frac {\text{number of correct sentences selected}} {\text{total number of sentences selected}}

    \\

    Recall &= \frac {\text{number of correct sentences selected}} {\text{total number of correct sentences in all sets}}

We are also considering a number of other metrics for the purpose of analyzing system behavior, but those will not be the official metric used for ranking.
