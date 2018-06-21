from __future__ import print_function, division

import numpy as np
import torch
from torch.nn import CrossEntropyLoss


class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (torch.NN.CrossEntropyLoss, optional): loss for evaluator (default: torch.NN.CrossEntropyLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss_func=CrossEntropyLoss(), batch_size=64):
        self.loss_func = loss_func
        self.batch_size = batch_size

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (models.networks): model to evaluate
            data (dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        match = 0
        total = 0
        recall = {'@1': 0, '@2': 0, '@5': 0, '@10': 0, '@50': 0, '@100': 0}
        loss = 0

        # device = None if torch.cuda.is_available() else -1

        with torch.no_grad():
            for batch in data.make_batches(self.batch_size):
                if torch.cuda.is_available():
                    context_variables = torch.tensor(batch[0]).cuda()
                    responses_variables = torch.tensor(batch[1]).cuda()
                    target_variables = torch.tensor(batch[2]).cuda()
                else:
                    context_variables = torch.tensor(batch[0])
                    responses_variables = torch.tensor(batch[1])
                    target_variables = torch.tensor(batch[2])

                outputs = model(context_variables, responses_variables)

                # Get loss
                if len(outputs.size()) == 1:
                    outputs = outputs.unsqueeze(0)
                loss += self.loss_func(outputs, target_variables)

                # Evaluation
                predictions = np.argsort(outputs.numpy(), axis=1)
                num_samples = predictions.shape[0]

                ranks = predictions[np.arange(num_samples), target_variables]
                match += sum(ranks == 0)
                recall['@1'] = match
                recall['@2'] += sum(ranks <= 2)
                recall['@5'] += sum(ranks <= 5)
                recall['@10'] += sum(ranks <= 10)
                recall['@50'] += sum(ranks <= 50)
                recall['@100'] += sum(ranks <= 100)
                total += num_samples

        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return loss, accuracy, {k: v/total for k, v in recall.items()}
