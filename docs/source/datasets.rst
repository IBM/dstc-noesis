Datasets
========

The datasets can be downloaded from the following links.

Note: the Ubuntu data is NOT the same as the previous Ubuntu dataset from `Lowe et. al (2015) <https://arxiv.org/abs/1506.08909>`. It is a new resource, described in the following paper::

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

Training and Validation
-----------------------

    +----------+----------------------+--------------------------+---------------------+
    | Sub-Task | Training             | Validation               | Other               |
    +==========+======================+==========================+=====================+
    | 1        | Ubuntu_st1_train_    | Ubuntu_st1_validation_   | None                |
    |          |                      |                          |                     |
    |          | Advising_st1_train_  | Advising_st1_validation_ |                     |
    +----------+----------------------+--------------------------+---------------------+
    | 2        | Ubuntu_st2_train_    | Ubuntu_st2_validation_   | Candidate_pool_     |
    |          |                      |                          |                     |
    +----------+----------------------+--------------------------+---------------------+
    | 3        |                      |                          | None                |
    |          | Advising_st3_train_  | Advising_st3_validation_ |                     |
    +----------+----------------------+--------------------------+---------------------+
    | 4        | Ubuntu_st4_train_    | Ubuntu_st4_validation_   | None                |
    |          |                      |                          |                     |
    |          | Advising_st4_train_  | Advising_st4_validation_ |                     |
    +----------+----------------------+--------------------------+---------------------+
    | 5        | Same as subtask 1    | Same as subtask 1        | Linux_manpages_     |
    |          |                      |                          |                     |
    |          |                      |                          | Course_information_ |
    +----------+----------------------+--------------------------+---------------------+

.. _Ubuntu_st1_train: https://ibm.box.com/s/fsk885se8ieoape46uzk7ylhx1097kk9
.. _Advising_st1_train: https://ibm.box.com/s/sb5wloejbsbhrpfws0yuj1wbb28you2w
.. _Ubuntu_st1_validation: https://ibm.box.com/s/rqb6bocovby1jau112y5wq99tz1fffp2
.. _Advising_st1_validation: https://ibm.box.com/s/f53kcojriaqrj5taevtw3doaatq3sfjv
.. _Ubuntu_st2_train: https://ibm.box.com/s/i9o9gz37leycvxfqgdabh7478ep1dqo7
.. _Ubuntu_st2_validation: https://ibm.box.com/s/ha4lcw6cjcwq6wseq5qv0t6ogxat2fhl
.. _Candidate_pool: https://ibm.box.com/s/uyzbhvt6zuowg120qzin099fbcijc2bp
.. _Advising_st3_train: https://ibm.box.com/s/kfev11bqpsvhwl8u2ko4fxb11kl9satq
.. _Advising_st3_validation: https://ibm.box.com/s/vhwmnt0kg1j1vx1j5wijez67mhjxjlnc
.. _Ubuntu_st4_train: https://ibm.box.com/s/ss7vaagg83qsycjv38bce6i8wsze8p9k
.. _Advising_st4_train: https://ibm.box.com/s/4p31ja8p83fehes0f6cuakr2wbdd4px9
.. _Ubuntu_st4_validation: https://ibm.box.com/s/6jmxiavc50achlr7k4g5i5lgyspcsqbg
.. _Advising_st4_validation: https://ibm.box.com/s/6jq99o1cz9m3env319s6e02ibtwksc1b
.. _Linux_manpages: https://ibm.box.com/s/7ro3t72tp0rcnggq5cgq9hq80fvh5pkh
.. _Course_information: https://ibm.box.com/s/lslz39r951fys52qqa3enl0ccods5lus

Additionally, for the Advising data, we are providing a form of the data with the original dialogs and their paraphrases before remixing. This can be used for training in any subtask, and can be downloaded here_. 
The global candidate pool for the sub-task 2, should be shared across training, validation and test datasets for sub-task 2.

.. _here: https://ibm.box.com/s/qh9gbkjo8pg8uph3vysv9fjhp18407fx

Test
----
    +----------+---------------------------+
    | Sub-Task | Test                      |
    +==========+===========================+
    | 1        | Ubuntu_st1_test_          |
    |          |                           |
    |          | Advising_st1_case1_test_  |
    |          |                           |
    |          | Advising_st1_case2_test_  |
    +----------+---------------------------+
    | 2        | Ubuntu_st2_test_          |
    |          |                           |
    |          |                           |
    +----------+---------------------------+
    | 3        |                           |
    |          | Advising_st3_case1_test_  |
    |          |                           |
    |          | Advising_st3_case2_test_  |
    +----------+---------------------------+
    | 4        | Ubuntu_st4_test_          |
    |          |                           |
    |          | Advising_st4_case1_test_  |
    |          |                           |
    |          | Advising_st4_case2_test_  |
    +----------+---------------------------+
    | 5        | Same as subtask 1         |
    |          |                           |
    |          |                           |
    +----------+---------------------------+

.. _Ubuntu_st1_test: https://ibm.box.com/s/lerplhwcm7n6nbsnywhku5m8kckxq90n
.. _Advising_st1_case1_test: https://ibm.box.com/s/bw6wj2lbt2g9alarnsoj8myckexij1s2
.. _Advising_st1_case2_test: https://ibm.box.com/s/9vkmus89gn459th9l8mtiarj1yj3jtyz
.. _Ubuntu_st2_test: https://ibm.box.com/s/pw3v5nz152yr75d9dfcvsldpo5kfpvuu
.. _Advising_st3_case1_test: https://ibm.box.com/s/cip5j31lptl8ih2cy0o1kj6mtljdihp0
.. _Advising_st3_case2_test: https://ibm.box.com/s/tteqjsflzm5venqezv1ba6hwv6qh5msv
.. _Ubuntu_st4_test: https://ibm.box.com/s/2socex1jk1h9vw6ni1l8v868vbp6og1j
.. _Advising_st4_case1_test: https://ibm.box.com/s/mdznqlga1g6i4j7knq0opkmf7m7m7plr
.. _Advising_st4_case2_test: https://ibm.box.com/s/5tqmwio1j59i04emix83y6ro4dwvgd0g


Ground truth for test datasets
------------------------------
    +----------+-----------------------------------+
    | Sub-Task | Test                              |
    +==========+===================================+
    | 1        | Ubuntu_st1_ground_truth_          |
    |          |                                   |
    |          | Advising_st1_case1_ground_truth_  |
    |          |                                   |
    |          | Advising_st1_case2_ground_truth_  |
    +----------+-----------------------------------+
    | 2        | Ubuntu_st2_ground_truth_          |
    |          |                                   |
    |          |                                   |
    +----------+-----------------------------------+
    | 3        |                                   |
    |          | Advising_st3_case1_ground_truth_  |
    |          |                                   |
    |          | Advising_st3_case2_ground_truth_  |
    +----------+-----------------------------------+
    | 4        | Ubuntu_st4_ground_truth_          |
    |          |                                   |
    |          | Advising_st4_case1_ground_truth_  |
    |          |                                   |
    |          | Advising_st4_case2_ground_truth_  |
    +----------+-----------------------------------+
    | 5        | Same as subtask 1                 |
    |          |                                   |
    |          |                                   |
    +----------+-----------------------------------+

.. _Ubuntu_st1_ground_truth: https://ibm.box.com/s/xjf30dirjql3t5y2zuhaytr6x9pr6soa
.. _Advising_st1_case1_ground_truth: https://ibm.box.com/s/gtogma9n6torzupv1g6g01c9kmuzkc4f
.. _Advising_st1_case2_ground_truth: https://ibm.box.com/s/7ay1kjeqp25laspho7egiwhea61r2xky
.. _Ubuntu_st2_ground_truth: https://ibm.box.com/s/f7so3abgcdt7afvr17mmyzswotkxa3my
.. _Advising_st3_case1_ground_truth: https://ibm.box.com/s/5f0rh7vnqpgyq3oa7kwstin8nlhv80qa
.. _Advising_st3_case2_ground_truth: https://ibm.box.com/s/3yqc61kkmxjid0cg4fw8uz1s0a226voo
.. _Ubuntu_st4_ground_truth: https://ibm.box.com/s/w6gs5g5j0ea2069pq9p1ipqs2imchfxt
.. _Advising_st4_case1_ground_truth: https://ibm.box.com/s/s4cd0et6bx20eusn20ko5azvpn9utp68
.. _Advising_st4_case2_ground_truth: https://ibm.box.com/s/zrzlvpds1ekfsq4oznmxry0yfwyjuhcl
