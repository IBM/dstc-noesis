from __future__ import division
import logging
import os
import random

import torch
from torch.nn import CrossEntropyLoss
from torch import optim
from noesis.util.checkpoint import Checkpoint
from noesis.evaluator.evaluator import Evaluator


class SupervisedTrainer(object):
    r""" The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss_func (torch.nn.CrossEntropyLoss, optional): loss for training, (default: torch.nn.CrossEntropyLoss)
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
    """
    def __init__(self, expt_dir='experiment', loss_func=CrossEntropyLoss(), batch_size=64,
                 random_seed=None,
                 checkpoint_every=5000, print_every=1000):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss_func = loss_func
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.evaluator = Evaluator(batch_size=self.batch_size)

        self.logger = logging.getLogger(__name__)

    def _train_batch(self, context_variable, responses_variable, target_variable, model):
        loss_func = self.loss_func
        # Forward propagation
        outputs = model(context_variable, responses_variable)
        # Get loss
        if len(outputs.size()) == 1:
            outputs = outputs.unsqueeze(0)
        loss = loss_func(outputs, target_variable)

        # Backward propagation
        model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def _train_epoches(self, data, model, n_epochs, start_epoch, start_step, batch_size=2,
                       dev_data=None):
        log = self.logger

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch

        # device = None if torch.cuda.is_available() else -1
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        steps_per_epoch = data.num_batches(batch_size)
        total_steps = steps_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0
        for epoch in range(start_epoch, n_epochs + 1):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))

            model.train(True)
            for batch in data.make_batches(batch_size):
                step += 1
                step_elapsed += 1

                # context_variables = torch.tensor(batch[0])
                # responses_variables = torch.tensor(batch[1])
                # target_variables = torch.tensor(batch[2])
                # context_variables = torch.tensor(batch[0], device=device)
                # responses_variables = torch.tensor(batch[1], device=device)
                # target_variables = torch.tensor(batch[2], device=device)

                if torch.cuda.is_available():
                    context_variables = torch.tensor(batch[0]).cuda()
                    responses_variables = torch.tensor(batch[1]).cuda()
                    target_variables = torch.tensor(batch[2]).cuda()
                else:
                    context_variables = torch.tensor(batch[0])
                    responses_variables = torch.tensor(batch[1])
                    target_variables = torch.tensor(batch[2])

                loss = self._train_batch(context_variables, responses_variables, target_variables, model)

                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    log_msg = 'Progress: %d%%, Train %s: %.4f' % (
                        step / total_steps * 100,
                        'CrossEntropyLoss',
                        print_loss_avg)
                    log.info(log_msg)

                # Checkpoint
                if step % self.checkpoint_every == 0 or step == total_steps:
                    Checkpoint(model=model,
                               optimizer=self.optimizer,
                               epoch=epoch, step=step,
                               vocab=data.vocab).save(self.expt_dir)

            if step_elapsed == 0:
                continue

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0
            log_msg = "Finished epoch %d: Train %s: %.4f" % (epoch, 'CrossEntropyLoss', epoch_loss_avg)
            if dev_data is not None:
                dev_loss, accuracy, recall = self.evaluator.evaluate(model, dev_data)
                # self.optimizer.update(dev_loss, epoch)
                log_msg += ", Dev CrossEntropyLoss: %.4f, Accuracy: %.4f" % (dev_loss, accuracy)
                log_msg += ", \n Recall: {}".format(recall)
                model.train(mode=True)
            else:
                self.optimizer.update(epoch_loss_avg, epoch)

            log.info(log_msg)

    def train(self, model, data, num_epochs=5,
              resume=False, dev_data=None,
              optimizer=None):
        r""" Run training for a given model.

        Args:
            model (models.networks): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (models.dataset.dataset.Dataset): dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data (models.dataset.dataset.Dataset, optional): dev Dataset (default None)
            optimizer (pytorch.optim, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))

        """
        # If training is set to resume
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = optim.Adam(model.parameters())
            self.optimizer = optimizer

        self.logger.info("Optimizer: %s" % self.optimizer)

        self._train_epoches(data, model, num_epochs,
                            start_epoch, step, dev_data=dev_data)
