import os
import argparse
import logging

import torch
from torch.nn import CrossEntropyLoss

from networks.dual_encoder import Encoder, DualEncoder
from trainers.supervised_trainer import SupervisedTrainer
from util.checkpoint import Checkpoint
from dataset.dataset import Dataset
from evaluator.evaluator import Evaluator

# Sample usage:
#     # training
#     python sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#      python sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data')
parser.add_argument('--test_path', action='store', dest='test_path',
                    help='Path to test data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    dual_encoder = checkpoint.model
    vocab = checkpoint.vocab
else:
    # Prepare dataset
    train = Dataset.from_file(opt.train_path)
    dev = Dataset.from_file(opt.dev_path)
    vocab = train.vocab
    max_len = 500

    # Prepare loss
    loss_func = CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_func.cuda()

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size = 128
        bidirectional = True
        context_encoder = Encoder(vocab.get_vocab_size(), max_len, hidden_size,
                             bidirectional=bidirectional, variable_lengths=False)
        response_encoder = Encoder(vocab.get_vocab_size(), max_len, hidden_size,
                             bidirectional=bidirectional, variable_lengths=False)

        dual_encoder = DualEncoder(context_encoder, response_encoder)
        if torch.cuda.is_available():
            dual_encoder.cuda()

        for param in dual_encoder.parameters():
            param.data.uniform_(-0.08, 0.08)

    # train
    t = SupervisedTrainer(loss_func=loss_func, batch_size=64,
                          checkpoint_every=5000, print_every=1000, expt_dir=opt.expt_dir)

    t.train(dual_encoder, train, num_epochs=10, dev_data=dev, optimizer=optimizer, resume=opt.resume)

    evaluator = Evaluator(batch_size=64)
    l, precision, recall = evaluator.evaluate(dual_encoder, dev)
    print("Precision: {}, Recall: {}".format(precision, recall))



