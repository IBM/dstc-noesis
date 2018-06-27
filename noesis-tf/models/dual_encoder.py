import tensorflow as tf
from models import helpers

FLAGS = tf.flags.FLAGS


def get_embeddings(hparams):
    if hparams.glove_path and hparams.vocab_path:
        tf.logging.info("Loading Glove embeddings...")
        vocab_array, vocab_dict = helpers.load_vocab(hparams.vocab_path)
        glove_vectors, glove_dict = helpers.load_glove_vectors(hparams.glove_path, vocab=set(vocab_array))
        initializer = helpers.build_initial_embedding_matrix(vocab_dict, glove_dict, glove_vectors,
                                                             hparams.embedding_dim)
    else:
        tf.logging.info("No glove/vocab path specificed, starting with random embeddings.")
        initializer = tf.random_uniform_initializer(-0.25, 0.25)

    if hparams.glove_path and hparams.vocab_path:
        return tf.get_variable(
            "word_embeddings",
            initializer=initializer)
    elif hparams.vocab_path:
        vocab_array, vocab_dict = helpers.load_vocab(hparams.vocab_path)
        return tf.get_variable(
            "word_embeddings",
            shape=[len(vocab_dict), hparams.embedding_dim],
            initializer=initializer)
    else:
        return tf.get_variable(
            "word_embeddings",
            shape=[hparams.vocab_size, hparams.embedding_dim],
            initializer=initializer)


def dual_encoder_model(
        hparams,
        mode,
        context,
        context_len,
        utterances,
        utterances_len,
        targets,
        batch_size):
    # Initialize embeddings randomly or with pre-trained vectors if available
    embeddings_W = get_embeddings(hparams)

    # Embed the context and the utterance
    context_embedded = tf.nn.embedding_lookup(
        embeddings_W, context, name="embed_context")
    utterances_embedded = tf.nn.embedding_lookup(
        embeddings_W, utterances, name="embed_utterance")


    # Build the Context Encoder RNN
    with tf.variable_scope("encoder-rnn") as vs:
        # We use an LSTM Cell
        cell_context = tf.nn.rnn_cell.LSTMCell(
            hparams.rnn_dim,
            forget_bias=2.0,
            use_peepholes=True,
            state_is_tuple=True)

        # Run context through the RNN
        context_encoded_outputs, context_encoded = tf.nn.dynamic_rnn(cell_context, context_embedded,
                                                                            context_len, dtype=tf.float32)

    # Build the Utterance Encoder RNN
    with tf.variable_scope("decoder-rnn") as vs:
        # We use an LSTM Cell
        cell_utterance = tf.nn.rnn_cell.LSTMCell(
            hparams.rnn_dim,
            forget_bias=2.0,
            use_peepholes=True,
            state_is_tuple=True)
        # Run all utterances through the RNN batch by batch
        # TODO: Needs to be parallelized
        all_utterances_encoded = []
        for i in range(batch_size):
            temp_outputs, temp_states = tf.nn.dynamic_rnn(cell_utterance, utterances_embedded[:,i],
                                                          utterances_len[i], dtype=tf.float32)
            all_utterances_encoded.append(temp_states[1]) # since it's a tuple, use the hidden states

        all_utterances_encoded = tf.stack(all_utterances_encoded, axis=0)

    with tf.variable_scope("prediction") as vs:
        M = tf.get_variable("M",
                            shape=[hparams.rnn_dim, hparams.rnn_dim],
                            initializer=tf.truncated_normal_initializer())

        # "Predict" a  response: c * M
        generated_response = tf.matmul(context_encoded[1], M) # using the hidden states
        generated_response = tf.expand_dims(generated_response, 1)
        all_utterances_encoded = tf.transpose(all_utterances_encoded, perm=[0, 2, 1]) # transpose last two dimensions

        # Dot product between generated response and actual response
        # (c * M) * r
        logits = tf.matmul(generated_response, all_utterances_encoded)
        logits = tf.squeeze(logits, [1])

        # Apply sigmoid to convert logits to probabilities
        probs = tf.nn.softmax(logits)

        if mode == tf.contrib.learn.ModeKeys.INFER:
            return probs, None

        # Calculate the binary cross-entropy loss
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(targets))

    # Mean loss across the batch of examples
    mean_loss = tf.reduce_mean(losses, name="mean_loss")
    return probs, mean_loss
