FAQs
====

1. **What is the timeline of the competition**?

    +-----------------------------------+-----------------------+
    |       Task                        |        Dates          |
    +===================================+=======================+
    | Development phase (14 weeks)      | Jun 1 – Sep 9, 2018   |
    +-----------------------------------+-----------------------+
    | Evaluation phase (2 weeks)        | Sep 10 – Sep 24, 2018 |
    +-----------------------------------+-----------------------+
    | Release of the results            | 1 Oct 2018            |
    +-----------------------------------+-----------------------+
    | Paper submission deadline         |  Oct-Nov 2018         |
    +-----------------------------------+-----------------------+
    | DSTC7 special session or workshop | Spring 2019           |
    +-----------------------------------+-----------------------+

|

2. **What should we submit**?

    You are required to submit the responses to the test dataset that will be released on the 10th of September. The format of the responses to be submitted will be announced later.

3. **Do we need to work on both datasets**?

    Not necessary. You can select one dataset and work on all or a subset of the subtasks. But, submitting results for both datasets and all subtasks will increase your chance of winning the competition.

4. **How are we evaluated**?

    For each test instance, we expect you to return a set of 10 choices (candidate ids) from the set candidates and a probability distribution over those 10 choices. For the competition metric we will consider the choices that cover 90% of the distribution, and compute an F-score as the harmonic mean of precision and recall.