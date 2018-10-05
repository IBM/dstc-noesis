import argparse
import logging
import ijson

import numpy as np


def read_predictions(filename):
    predictions = {}
    with open(filename, 'r') as fp:
        for item in ijson.items(fp, 'item'):
            predictions[item['dialog-id']] = [candidate['candidate-id'] for candidate in item['candidate-ranking']]
    return predictions


def read_targets(filename):
    targets = {}
    with open(filename, 'r') as fp:
        for line in fp:
            uid, answer_id = line.rstrip('\n').split(sep='\t')
            targets[uid] = answer_id
    return targets


def rank(src, tgt):
    """
    The function calculates rank for each prediction given target

    Args:
        src (dict): predictions by the model
        tgt (dict): ground truth/ targets

    Returns:
         ranks (list): rank of a correct responses (default = 0)
    """
    ranks = []
    for idx, target in tgt.items():
        ranks.append(0)
        try:
            predictions = src[idx]
            for i, entry in enumerate(predictions):
                if entry == target:
                    ranks[-1] = i + 1
                    break
        except KeyError:
            msg = "No matching entry found for test case with dialog-id {}".format(idx)
            logging.warning(msg)

    return ranks


def calculate_recall(ranks):
    """
    The function calculates recall at different cutoff points.

    Args:
        ranks (list): represents position of a correct response for given problem
    """
    ranks = np.array(ranks)
    nonzero = ranks[np.nonzero(ranks)]

    total = len(ranks)
    result = dict()
    result["R@1"] = len(np.flatnonzero(nonzero <= 1)) / total
    result["R@10"] = len(np.flatnonzero(nonzero <= 10)) / total
    result["R@50"] = len(np.flatnonzero(nonzero <= 50)) / total
    result["R@100"] = len(nonzero) / total
    logging.info(result)


def calculate_MRR(ranks):
    """
    The function calculate Mean Reciprocal Rank (MRR).
    Args:
        ranks (list): represents position of a correct response for given problem
    """
    ranks = np.array(ranks)
    idx = np.nonzero(ranks)
    msg = "Mean Reciprocal Rank (MRR): {}".format((1 / sum(ranks[idx])) / len(ranks))
    logging.info(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', action='store', dest='predictions',
                        help='Path to predictions file in the requested format.')
    parser.add_argument('--targets', action='store', dest='targets',
                        help='Path to ground truth/targets file.')
    parser.add_argument('--log-level', dest='log_level',
                        default='debug',
                        help='Logging level.')

    opt = parser.parse_args()

    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
    logging.info(opt)

    predictions = read_predictions(opt.predictions)
    targets = read_targets(opt.targets)

    ranks = rank(predictions, targets)
    calculate_recall(ranks)
    calculate_MRR(ranks)