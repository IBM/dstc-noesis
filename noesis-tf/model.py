import tensorflow as tf


def get_id_feature(features, key, len_key, max_len):
    ids = features[key]
    ids_len = tf.squeeze(features[len_key], [1])
    ids_len = tf.minimum(ids_len, tf.constant(max_len, dtype=tf.int64))
    return ids, ids_len


def create_train_op(loss, hparams):
    def exp_decay(learning_rate, global_step):
        return tf.train.exponential_decay(learning_rate, global_step, decay_steps=hparams.decay_steps, decay_rate=hparams.decay_rate,
                                          staircase=hparams.staircase, name="lr_decay")
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=hparams.learning_rate,
        clip_gradients=10.0,
        optimizer=hparams.optimizer,
        learning_rate_decay_fn=exp_decay
    )
    return train_op


def create_model_fn(hparams, model_impl):
    def model_fn(features, targets, mode):
        context, context_len = get_id_feature(
            features, "context", "context_len", hparams.max_context_len)

        all_utterances = []
        all_utterances_lens = []

        for i in range(100):
            option, option_len = get_id_feature(features,
                                                "option_{}".format(i),
                                                "option_{}_len".format(i),
                                                hparams.max_utterance_len)
            all_utterances.append(option)
            all_utterances_lens.append(option_len)

        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            probs, loss = model_impl(
                hparams,
                mode,
                context,
                context_len,
                all_utterances,
                tf.transpose(tf.stack(all_utterances_lens, axis=0)),
                targets,
                hparams.batch_size)
            train_op = create_train_op(loss, hparams)
            return probs, loss, train_op

        if mode == tf.contrib.learn.ModeKeys.INFER:

            probs, loss = model_impl(
                hparams,
                mode,
                tf.concat(0, context),
                tf.concat(0, context_len),
                tf.concat(0, all_utterances),
                tf.concat(0, all_utterances_lens),
                None,
                hparams.eval_batch_size)

            split_probs = tf.split(0, features["len"], probs)
            probs = tf.concat(1, split_probs)

            return probs, 0.0, None

        if mode == tf.contrib.learn.ModeKeys.EVAL:
            probs, loss = model_impl(
                hparams,
                mode,
                context,
                context_len,
                all_utterances,
                tf.transpose(tf.stack(all_utterances_lens, axis=0)),
                targets,
                hparams.eval_batch_size)

            shaped_probs = probs

            return shaped_probs, loss, None

    return model_fn
