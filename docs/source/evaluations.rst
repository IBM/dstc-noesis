Evaluations
===========

Metrics
-------

For each test instance, we will expect you to return a set of 100 choices (candidate ids) from the set of possible follow-up sentences and a probability distribution over those 100 choices.
As competition metrics we will compute range of scores, including recall@k, MRR(mean reciprocal rank) and MAP(mean average precision).

Following evaluation metrics will be used to evaluate your submissions.

    +----------+------------------------------------------+-----------------------------------------------+
    | Sub-Task | Ubuntu                                   | Advising                                      |
    +==========+==========================================+===============================================+
    | 1        | Recall @1, Recall @10, Recall @50, MRR   |   Recall @1, Recall @10, Recall @50, MRR      |
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

Best Scores
-----------

The ranking considers the average of Recall\@10 and MRR. Best Recall\@10 and MRR scores for each subtask is is shown in the below table.

Recall\@10

    +----------+---------+-----------------+-----------------+
    | Sub-Task | Ubuntu  | Advising-Case-1 | Advising-Case-2 |
    +==========+=========+=================+=================+
    | 1        | 0.902   |    0.85         |   0.63          |
    +----------+---------+-----------------+-----------------+
    | 2        | 0.361   |        NA       |      NA         |
    +----------+---------+-----------------+-----------------+
    | 3        |   NA    |    0.906        |   0.75          |
    +----------+---------+-----------------+-----------------+
    | 4        | 0.739   |    0.652        |  0.508          |
    +----------+---------+-----------------+-----------------+
    | 5        | 0.905   |  0.864          |  0.63           |
    +----------+---------+-----------------+-----------------+


MRR

    +----------+---------+-----------------+-----------------+
    | Sub-Task | Ubuntu  | Advising-Case-1 | Advising-Case-2 |
    +==========+=========+=================+=================+
    | 1        | 0.7350  |    0.6078       |   0.3390        |
    +----------+---------+-----------------+-----------------+
    | 2        | 0.2528  |        NA       |      NA         |
    +----------+---------+-----------------+-----------------+
    | 3        |   NA    |    0.6238       |   0.4341        |
    +----------+---------+-----------------+-----------------+
    | 4        | 0.5891  |    0.3495       |  0.2422         |
    +----------+---------+-----------------+-----------------+
    | 5        | 0.7399  |  0.6455         |  0.3390         |
    +----------+---------+-----------------+-----------------+