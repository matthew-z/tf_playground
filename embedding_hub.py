import tensorflow as tf
import tensorflow_hub as hub


def _tokenize(seq):
    return seq.split()


def data_gen(data):
    for seq, label in data:
        seq_tokens = _tokenize(seq)
        yield (seq_tokens, len(seq_tokens)), label


def input_fn(data, batch_size, train=False):
    dataset = tf.data.Dataset.from_generator(lambda: data_gen(data), output_types=((tf.string, tf.int32), tf.int32))
    if train:
        dataset = dataset.shuffle(100).repeat()
    dataset = dataset.padded_batch(batch_size, padded_shapes=(([None], []), []), padding_values=(("", 0), 0))
    iterator = dataset.make_one_shot_iterator()
    (seq, length), label = iterator.get_next()
    feature_dict = {"seq": seq, "length": length}
    return feature_dict, label


def main():
    batch_size = 2
    dataset = [("a hello world", 0),
               ("are you aaaaaaaaaaaa ffffffffffff", 1)]
    feat, label = input_fn(dataset, batch_size)
    # embed = hub.Module("/Users/zhaohaozeng/.models/glove.6B.50d.2d", trainable=True)
    # inputs = embed(feat["seq"])

    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
    inputs = elmo(
        inputs={
            "tokens": feat["seq"],
            "sequence_len": feat["length"]
        },
        signature="tokens",
        as_dict=True)["elmo"]

    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=10)
    outputs, state = tf.nn.dynamic_rnn(
        rnn_cell, inputs=inputs, sequence_length=feat["length"], dtype=tf.float32)

    logits = tf.layers.dense(state, 1)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=logits)
    opt = tf.train.AdamOptimizer()
    opt.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        print((sess.run(inputs)))


main()
