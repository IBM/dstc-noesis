Submission
==========

Your submissions should be emailed to chulaka.gunasekara@ibm.com, with the subject line **DSTC7_Track1_Submission**. The results should be submitted from an email address that is registered for Track 1.

You need to submit a single zipped directory containing the result files for each of the subtasks that you need to be evaluated on. The files should be named in the following format.
``<dataset>_subtask_<subtask_number>.json``

The <dataset> should be replaced by either ‘Ubuntu’ or ‘Advising’, and the <subtask_number> should be replaced by the subtask number(1-5).
For example, the results file for subtask 1 on Ubuntu dataset should be named as Ubuntu_subtask_1.json

Each results file should follow the following json format.

.. code-block:: json

    [
        {
            "example-id": xxxxxxx,
            "candidate-ranking":[
                {
                    "candidate-id": aaaaaa,
                    "confidence": b.bbb
                },
                {
                    "candidate-id": cccccc,
                    "confidence": d.ddd
                },
                ...
            ]
        },
        ...
    ]


The value for the field "example-id" should contain the corresponding example-id of the test dataset. The candidate ranking field should ONLY include 100 candidates in the order of confidence.

For subtask 2, where the selection is made from a global list of candidates, candidate-ranking fields should **only include the top 100 candidates** from the global list.

For subtask 4, when the correct candidate not available in the candidate set, return ``"candidate-id": NONE`` with the confidence score as an item in the candidate-ranking list.