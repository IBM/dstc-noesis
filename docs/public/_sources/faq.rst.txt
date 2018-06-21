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

4. **What do you mean by end-to-end models**?

    We don't need the whole system to be end-to-end trainable. You can have separate components, which are not trained with back-propagation. However, we expect the functionality of each of the components in your system to be learned from the given dataset. We discourage the use of hand-coded features for any component in your system, as one of the focus points of the challenge is automation.

5. **Can we use pre-trained word embeddings**?

    You can use any pre-trained embeddings that was publicly available before the 1st of June.