from torch import nn
import torch
import logging


class Base(nn.Module):
    r"""
    Applies a multi-layer RNN to an input sequence.

    Warning:
        Do not use this class directly, use one of the sub classes.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): maximum allowed length for the sequence to be processed
        hidden_size (int): number of features in the hidden state `h`
        input_dropout_p (float): dropout probability for the input sequence
        dropout_p (float): dropout probability for the output sequence
        n_layers (int): number of recurrent layers
        rnn_cell (str): type of RNN cell (Eg. 'LSTM' , 'GRU')

    Inputs: ``*args``, ``**kwargs``
        - ``*args``: variable length argument list.
        - ``**kwargs``: arbitrary keyword arguments.

    """

    def __init__(self, vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell_type):
        super(Base, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.rnn_cell_type = rnn_cell_type.lower()
        if self.rnn_cell_type == 'lstm':
            self.rnn_cell = nn.LSTM
        elif self.rnn_cell_type == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell_type))

        self.dropout_p = dropout_p

        self.logger = logging.getLogger(__name__)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class Encoder(Base):
    r"""
    Applies a multi-layer RNN to an input sequence.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encodr (defulat False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)
        embedding (torch.Tensor, optional): Pre-trained embedding.  The size of the tensor has to match
            the size of the embedding parameter: (vocab_size, hidden_size).  The embedding layer would be initialized
            with the tensor if provided (default: None).
        update_embedding (bool, optional): If the embedding should be updated during training (default: False).

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences in the mini-batch, it must be provided when using variable length RNN (default: `None`)

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`

    Examples::
         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)
    """

    def __init__(self, vocab_size, max_len, hidden_size,
                 input_dropout_p=0, dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell_type='gru', variable_lengths=False,
                 embedding=None, update_embedding=True):
        super(Encoder, self).__init__(vocab_size, max_len, hidden_size,
                                      input_dropout_p, dropout_p, n_layers, rnn_cell_type)

        self.bidirectional = bidirectional
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=self.bidirectional, dropout=dropout_p)

    def forward(self, input_var, input_lengths=None):
        r"""
        Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden


class DualEncoder(nn.Module):
    r""""
    Applies dual encoder architecture.

    Args:
        context (noesis.networks.dual_encoder.Encoder): encoder RNN for context information
        response (noesis.networks.dual_encoder.Encoder): encoder RNN for responses information

    Inputs: context_var, responses_var
        - **context_var** : a tensor containing context information
        - **responses_var** : a tensor containing responses per context information

    Outputs: output
        - **output** (batch, num_responses): tensor containing scaled probabilities of responses

    Examples::
         >>> dual_encoder = DualEncoder(ctx_encoder, resp_encoder)
         >>> output = dual_encoder(ctx_variable, resp_variable)
    """
    def __init__(self, context, response, use_output=False):
        super(DualEncoder, self).__init__()
        self.context = context
        self.response = response
        self.use_output = use_output

        c_h = context.hidden_size
        r_h = response.hidden_size

        if self.context.bidirectional:
            c_h = 2 * c_h

        if self.response.bidirectional:
            r_h = 2 * r_h

        self.M = torch.randn([c_h, r_h], requires_grad=True)
        self.final_layer = nn.Softmax()

    def forward(self, context_var, responses_var, context_lengths_var, responses_lengths_var):
        r"""
        Applies a multi-layer RNN to an input sequence.

        Args:
            context_var (batch, seq_len): tensor containing the features of the context sequence.
            responses_var (batch, num_responses, seq_len): tensor containing the features of the responses sequence.

        Returns: output
            - **output** (batch, num_responses): variable containing the scaled probabilities over responses
        """
        batch, num_resp, seq_len = responses_var.size()

        if self.context.rnn_cell_type == 'gru':
            c, h_c = self.context(context_var, context_lengths_var)
        elif self.context.rnn_cell_type == 'lstm':
            c, (h_c, _) = self.context(context_var, context_lengths_var)

        if self.response.rnn_cell_type == 'gru':
            r, h_r = self.response(responses_var.reshape([-1, seq_len]), responses_lengths_var.reshape([-1]))
        elif self.response.rnn_cell_type == 'lstm':
            r, (h_r, _) = self.response(responses_var.reshape([-1, seq_len]), responses_lengths_var.reshape([-1]))

        # unscaled log probabilities
        if self.use_output:
            f_c = c.gather(1, context_lengths_var.view(-1, 1, 1).expand(c.size(0), 1, c.size(2)) - 1)
            f_r = r.gather(1, responses_lengths_var.view(-1, 1, 1).expand(r.size(0), 1, r.size(2)) - 1).squeeze(1)
            logits = torch.matmul(torch.matmul(f_c, self.M), f_r.reshape([batch, num_resp, -1]).transpose(1, 2)).squeeze(1)
        else:
            logits = torch.matmul(torch.matmul(h_c.view(self.context.n_layers, 1, -1), self.M),
                                  h_r.view(self.response.n_layers, num_resp, -1).transpose(1, 2)).squeeze()

        output = self.final_layer(logits)
        return output
