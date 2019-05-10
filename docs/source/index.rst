.. dstc7-noesis documentation master file, created by
   sphinx-quickstart on Wed Jun  6 01:03:45 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Noetic End-to-End Response Selection Challenge
==============================================

**Update** - A tensorflow based baseline for subtask 1 of the track is available `here <https://github.com/IBM/dstc7-noesis/tree/models/noesis-tf>`_.

This challenge is part of dialog state tracking challenge (DSTC 7) series. It provides a partial conversation, and requires participants to select the correct next utterances from a set of candidates.
Unlike previous similar challenges, this task tries to push towards real world problems by introducing:

- A large number of candidates
- Cases where no candidate is correct
- External data

This challenge is offered with two goal oriented dialog datasets, used in 5 subtasks.
A participant may participate in one, several, or all the subtasks.
A full description of the track is available `here <http://workshop.colips.org/dstc7/proposals/Track%201%20Merged%20Challenge%20Extended%20Desscription_v2.pdf>`_.

If you use this data or code in your work, please cite the task description paper::

  @InProceedings{dstc19task1,
    title     = {DSTC7 Task 1: Noetic End-to-End Response Selection},
    author    = {Chulaka Gunasekara, Jonathan K. Kummerfeld, Lazaros Polymenakos, and Walter S. Lasecki},
    year      = {2019},
    booktitle = {7th Edition of the Dialog System Technology Challenges at AAAI 2019},
    url       = {http://workshop.colips.org/dstc7/papers/dstc7_task1_final_report.pdf},
    month     = {January},
  }

If you use the Ubuntu data, please also cite the paper in which we describe its creation::

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

Organizers
----------
* `Lazaros Polymenako <mailto:lcpolyme@us.ibm.com>`_, `Chulaka Gunasekara <mailto:chulaka.gunasekara@ibm.com>`_ – IBM Research AI
* `Walter Lasecki <mailto:wlasecki@umich.edu>`_, `Jonathan K. Kummerfeld <mailto:jkummerf@umich.edu>`_ – University of Michigan


Maintainers
-----------
* `Chulaka Gunasekara <https://researcher.watson.ibm.com/researcher/view.php?person=ibm-chulaka.gunasekara>`_ 

To get a guaranteed support you are kindly requested to open an issue.

Thank you for understanding!

.. toctree::
   :hidden:
   :glob:

   *
