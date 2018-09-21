Evaluations
===========

For each test instance, we will expect you to return a set of 100 choices (candidate ids) from the set of possible follow-up sentences and a probability distribution over those 100 choices.
As competition metrics we will compute range of scores, including recall@k, MRR(mean reciprocal rank) and MAP(mean average precision).

Following evaluation metrics will be used to evaluate your submissions.

    +----------+------------------------------------------+-----------------------------------------------+
    | Sub-Task | Ubuntu                                   | Advising                                      |
    +==========+==========================================+===============================================+
    | 1        | Recall @1, Recall @10, Recall @50, MRR   |   Recall @1, Recall @10, Recall @50, MRR]     |
    +----------+------------------------------------------+-----------------------------------------------+
    | 2        | Recall @1, Recall @10, Recall @50, MRR   |                                               |
    +----------+------------------------------------------+-----------------------------------------------+
    | 3        |                                          |   Recall @1, Recall @10, Recall @50, MRR, MAP |
    +----------+------------------------------------------+-----------------------------------------------+
    | 4        | Recall @1, Recall @10, Recall @50, MRR   |   Recall @1, Recall @10, Recall @50, MRR      |
    +----------+------------------------------------------+-----------------------------------------------+
    | 5        | Recall @1, Recall @10, Recall @50, MRR   |   Recall @1, Recall @10, Recall @50, MRR      |
    +----------+------------------------------------------+-----------------------------------------------+

**Note:**
We will evaluate MAP for sub-task 3 with Advising data as the you are supposed to return the correct response and all the paraphrases associated with it.

Rankings
--------

We will be announcing **two winners**; one for each dataset. The winners will be decided by a score which considers the average of MRR and Recall @10.