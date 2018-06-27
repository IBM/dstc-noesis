Evaluations
===========

For each test instance, we will expect you to return a set of 100 choices (candidate ids) from the set of possible follow-up sentences and a probability distribution over those 100 choices.
As competition metrics we will compute range of scores, including F-score as the harmonic mean of precision and recall, recall@k, MRR(mean reciprocal rank) and MAP(mean average precision):

.. math::

    Precision &= \frac {\text{number of correct sentences selected}} {\text{total number of sentences selected}}

    \\

    Recall &= \frac {\text{number of correct sentences selected}} {\text{total number of correct sentences in all sets}}

The final set of competition metrics will be announced soon.