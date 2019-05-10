Data Description
================

The datasets will be available to the contestants upon `registration <https://ibm.biz/BdZ6E3>`_ and selecting **Sentence Selection** as an interested track.


Ubuntu
------

A new set of disentangled Ubuntu IRC dialogs representing two party conversations extracted from the Ubuntu IRC channel.
A typical dialog starts with a question that was asked by *participant_1*, and then someone else, *participant_2*, responds with either an answer or follow-up questions that then lead to a back-and-forth conversation.
In this challenge, the context of each dialog contains more than 3 turns which occurred between the two participants and the next turn of *participant_2* should be selected from the given set of candidate utterances.
We focus on *participant_2* to set the task up as creating a bot that could help provide answers.
Relevant external information of the form of Linux manual pages is also provided.

Note: this data is NOT the same as the previous Ubuntu datasets from `Lowe et. al (2015) <https://arxiv.org/abs/1506.08909>`. It is a new resource, described in the following paper::

  @Article{arxiv18disentangle,
    author    = {Jonathan K. Kummerfeld, Sai R. Gouravajhala, Joseph Peper, Vignesh Athreya, Chulaka Gunasekara, Jatin Ganhotra, Siva Sankalp Patel, Lazaros Polymenakos, and Walter S. Lasecki},
    title     = {Analyzing Assumptions in Conversation Disentanglement Research Through the Lens of a New Dataset and Model},
    journal   = {ArXiv e-prints},
    archivePrefix = {arXiv},
    eprint    = {1810.11118},
    primaryClass = {cs.CL},
    year      = {2018},
    month     = {October},
    url       = {https://arxiv.org/pdf/1810.11118.pdf},
  }

Advising
--------

A two party dialogs dataset that simulate a discussion between a *student* and an academic *advisor*.
The purpose of the dialogs is to guide the student to pick courses that fit not only their curriculum, but also personal preferences about time, difficulty, areas of interest, etc.
These conversations were collected by having students at the University of Michigan act as the two roles using provided personas.
Structured information in the form of a database of course information will be provided, as well as the personas (though at test time only information available to the advisor will be provided, i.e. not the explicit student preferences).
The data also includes paraphrases of the sentences and of the target responses.

Sub-Tasks
---------
We are considered several tasks that have similar structure, but vary in the output space and available context. In the table below, [x] indicates that the task is evaluated on the marked dataset.

    +----------+-----------------------------------------------------------------------------------------------------------+--------+----------+
    | Sub-Task | Description                                                                                               | Ubuntu | Advising |
    +==========+===========================================================================================================+========+==========+
    | 1        | Select the next utterance from a candidate pool of 100 sentences                                          |   [x]  |    [x]   |
    +----------+-----------------------------------------------------------------------------------------------------------+--------+----------+
    | 2        | Select the next utterance from a candidate pool of 120000                                                 |   [x]  |          |
    +----------+-----------------------------------------------------------------------------------------------------------+--------+----------+
    | 3        | Select the next utterance and its paraphrases from candidate pool of 100                                  |        |    [x]   |
    +----------+-----------------------------------------------------------------------------------------------------------+--------+----------+
    | 4        | Select the next utterance from a candidate pool of 100 which might not contain the correct next utterance |   [x]  |    [x]   |
    +----------+-----------------------------------------------------------------------------------------------------------+--------+----------+
    | 5        | Select the next utterance from a candidate pool of 100 incorporating the provided external knowledge      |   [x]  |    [x]   |
    +----------+-----------------------------------------------------------------------------------------------------------+--------+----------+


In **sub-task 1**, for each partial dialog and a candidate pool of 100 is given and the contestants are expected to select the best next utterance from the given pool.

In **sub-task 2**, one large candidate pool of 120000 utterances is shared by training and validation datasets. The next best utterance should be selected from this large pool of candidate utterances.

For **sub-task 3**, in addition to the training and validation dialog datasets, and extra dataset which includes paraphrases for utterances is provided. The contestants are required to use the paraphrase information to select the next utterance as well as its paraphrases from the given set of 100 candidates.

The candidate sets that are provided for some dialogs in **sub-task 4** does not include the correct next utterance. The contestants are expected to train their models in a way that during testing they can identify such cases.

In **sub-task 5**, additional external information which will be important for dialog modeling will be provided. For Ubuntu dataset, this external information comes in the form of Linux manual pages and for Advising dataset, extra information about courses will be given. The same training, validation and test data files in task 1 will be reused for this task. The contestants can use the provided knowledge sources as is, or transform them to appropriate representations (e.g. knowledge graphs, continuous embeddings, etc.) that can be integrated with end-to-end dialog systems to improve accuracy.


Data Format
-----------

Each dialog contains in training, validation and test datasets follows the JSON format which is similar to the below example.

.. code-block:: json

    {
        "data-split": "train",
        "example-id": 1100001,
        "messages-so-far": [
            {
                "speaker": "participant_1",
                "utterance": "hey guys, does your livecd have chroot installed? and bash?"
            },
            {
                "speaker": "participant_2",
                "utterance": "sure"
            },
            ...
        ],
        "options-for-correct-answers": [
            {
                "candidate-id": "TLSHF16Y4J4L",
                "utterance": "what are you missing in apt ?"
            }
        ],
        "options-for-next": [
            {
                "candidate-id": "YWOA49156J9P",
                "utterance": "issues with msn?. I'm experiencing them on windows atm, current msn version"
            },
            {
                "candidate-id": "RYBI7QRD9QZN",
                "utterance": "<> AmaroqWolf: alias='sudo admincommand'.  <AmaroqWolf>  aw, can't make myself type sudo? I like it better that way."
            },
            ...
        ],
        "scenario": 1
    }


The field `messages-so-far` contains the context of the dialog and `options-for-next` contains the candidates to select the next utterance from. The correct next utterance is given in the field `options-for-correct-answers`. The field `scenario` refers to the subtask.

For each dialog in `Advising` dataset, we provide a profile that contains information used during the creation of the dialog. It has the following fields:

- `Aggregated` - contains student preferences, with each field matching up with a field in the `course-info.json` file.
- `Courses` - contains two lists, first is a list of courses this student has taken ("Prior") and second is a list of suggestions that the advisor had access to ("Suggested").
- `Term` - specifies the simulated year and semester for the conversation
- `Standing` - specifies how far through their degree the student is.



External Data
-------------

Each course offering record found in the external dataset provided for Advising domain contains the following fields.

    +---------------------+---------------------------------------------------------------------------------------------+
    | Field               | Description                                                                                 |
    +=====================+=============================================================================================+
    | Area                | Six general areas in computer science (otherwise NA)                                        |
    +---------------------+---------------------------------------------------------------------------------------------+
    | Category            | Five general types of classes in computer science (otherwise NA)                            |
    +---------------------+---------------------------------------------------------------------------------------------+
    | ClarityRating       | A number in [1.0, 5.0] indicating course clarity, or NA (78% of cases)                      |
    +---------------------+---------------------------------------------------------------------------------------------+
    | ClassSize           | A number in [1.0, 250.0] indicating average class size, or NA (71% of cases)                |
    +---------------------+---------------------------------------------------------------------------------------------+
    | Course              | Course ID, a series of letters and numbers                                                  |
    +---------------------+---------------------------------------------------------------------------------------------+
    | CourseTitle         | Complete course name                                                                        |
    +---------------------+---------------------------------------------------------------------------------------------+
    | Credits             | A number or a range (for example, indicated as 1 - 3)                                       |
    +---------------------+---------------------------------------------------------------------------------------------+
    | Description         | Free text description of the class topic                                                    |
    +---------------------+---------------------------------------------------------------------------------------------+
    | EasinessRating      | A number in [1.0, 5.0] indicating course difficulty level, or NA (78% of cases)             |
    +---------------------+---------------------------------------------------------------------------------------------+
    | HasDiscussion       | Whether the course has a discussion section (Y, N or null)                                  |
    +---------------------+---------------------------------------------------------------------------------------------+
    | HasLab              | Whether the course has a lab (Y, N, null)                                                   |
    +---------------------+---------------------------------------------------------------------------------------------+
    | HelpfulnessRating   | A number in [1.0, 5.0] indicating how helpful course staff were, or NA (78% of cases)       |
    +---------------------+---------------------------------------------------------------------------------------------+
    | Semester            | Which semester the class was held in (Fall, Winter, Spring, Summer, or Spring-Summer)       |
    +---------------------+---------------------------------------------------------------------------------------------+
    | Workload            | One of {1, 2, 3, 4, null, NA}, where higher numbers indicate higher workload                |
    +---------------------+---------------------------------------------------------------------------------------------+
    | Year                | A four digit number                                                                         |
    +---------------------+---------------------------------------------------------------------------------------------+
    | Section             | Information about available sections. The key for each is the instructor name, or 'NA"      |
    +---------------------+---------------------------------------------------------------------------------------------+
    | DaysOfClass         | One per section, lists weekdays, or for unknown it has "" or []                             |
    +---------------------+---------------------------------------------------------------------------------------------+
    | StartTime / EndTime | When the class is held as a 24-hour time, or NA (2%) or 0:00:00 (71%) when unknown          |
    +---------------------+---------------------------------------------------------------------------------------------+




***All the datasets will be publicly available after the competition.***
