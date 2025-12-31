"""OFDM deep learning pipeline with a more structured training entrypoint."""

from __future__ import annotations
from __future__ import division

import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


@dataclass
class OFDMSystemConfig:
    """Configuration of the OFDM system and modulation parameters."""

    num_subcarriers: int = 64
    pilot_count: int = 64
    bits_per_symbol: int = 2
    snr_db: int = 20
    cyclic_prefix: int | None = None

    def __post_init__(self) -> None:
        if self.cyclic_prefix is None:
            self.cyclic_prefix = self.num_subcarriers // 4

    @property
    def payload_bits_per_symbol(self) -> int:
        return self.num_subcarriers * self.bits_per_symbol


@dataclass
class TrainingConfig:
    """Configuration for the training loop and dataset handling."""

    train_dataset_path: str = "../H_dataset/"
    test_dataset_path: str = "../H_dataset/"
    train_channel_start: int = 1
    train_channel_end: int = 301
    test_channel_start: int = 301
    test_channel_end: int = 401
    pilot_cache_prefix: str = "Pilot_"
    pilot_cache_dir: str = "."
    learning_rate: float = 0.001
    learning_rate_decay_step: int = 2000
    learning_rate_decay_factor: float = 5.0
    training_epochs: int = 20000
    display_step: int = 5
    evaluation_step: int = 1000
    batch_iterations: int = 50
    batch_examples: int = 1000
    test_batch_examples: int = 1000
    extended_test_examples: int = 10000
    hidden_units: Tuple[int, int, int] = (500, 250, 120)
    output_bits: int = 16


def modulation(bits: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    """Map input bits to QAM constellation points."""
    reshaped_bits = bits.reshape((int(len(bits) / bits_per_symbol), bits_per_symbol))
    return (2 * reshaped_bits[:, 0] - 1) + 1j * (2 * reshaped_bits[:, 1] - 1)


def idft(ofdm_data: np.ndarray) -> np.ndarray:
    """Inverse discrete Fourier transform."""
    return np.fft.ifft(ofdm_data)


def add_cyclic_prefix(ofdm_time: np.ndarray, cyclic_prefix: int) -> np.ndarray:
    """Prepend a cyclic prefix to the time domain signal."""
    cp = ofdm_time[-cyclic_prefix:]
    return np.hstack([cp, ofdm_time])


def channel(signal: np.ndarray, channel_response: np.ndarray, snr_db: int) -> np.ndarray:
    """Pass the signal through the channel and add white noise."""
    convolved = np.convolve(signal, channel_response)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10 ** (-snr_db / 10)
    noise = np.sqrt(sigma2 / 2) * (
        np.random.randn(*convolved.shape) + 1j * np.random.randn(*convolved.shape)
    )
    return convolved + noise


def remove_cyclic_prefix(signal: np.ndarray, cyclic_prefix: int, num_subcarriers: int) -> np.ndarray:
    """Remove the cyclic prefix from the received OFDM block."""
    return signal[cyclic_prefix : cyclic_prefix + num_subcarriers]


def simulate_ofdm(
    codeword: np.ndarray,
    channel_response: np.ndarray,
    system_config: OFDMSystemConfig,
    pilot_value: np.ndarray,
    all_carriers: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate pilot and data transmission through the channel."""
    ofdm_data = np.zeros(system_config.num_subcarriers, dtype=complex)
    ofdm_data[all_carriers] = pilot_value
    ofdm_time = idft(ofdm_data)
    ofdm_with_cp = add_cyclic_prefix(ofdm_time, system_config.cyclic_prefix)
    ofdm_rx = channel(ofdm_with_cp, channel_response, system_config.snr_db)
    ofdm_rx_no_cp = remove_cyclic_prefix(ofdm_rx, system_config.cyclic_prefix, system_config.num_subcarriers)

    symbol = np.zeros(system_config.num_subcarriers, dtype=complex)
    codeword_qam = modulation(codeword, system_config.bits_per_symbol)
    symbol[np.arange(system_config.num_subcarriers)] = codeword_qam
    ofdm_time_codeword = np.fft.ifft(symbol)
    ofdm_with_cp_codeword = add_cyclic_prefix(ofdm_time_codeword, system_config.cyclic_prefix)
    ofdm_rx_codeword = channel(ofdm_with_cp_codeword, channel_response, system_config.snr_db)
    ofdm_rx_no_cp_codeword = remove_cyclic_prefix(
        ofdm_rx_codeword, system_config.cyclic_prefix, system_config.num_subcarriers
    )
    combined = np.concatenate(
        (
            np.real(ofdm_rx_no_cp),
            np.imag(ofdm_rx_no_cp),
            np.real(ofdm_rx_no_cp_codeword),
            np.imag(ofdm_rx_no_cp_codeword),
        )
    )
    return combined, abs(channel_response)


def load_or_create_pilot(system_config: OFDMSystemConfig, training_config: TrainingConfig) -> np.ndarray:
    """Load pilot bits from disk or generate and persist them."""
    pilot_file_name = os.path.join(
        training_config.pilot_cache_dir, f"{training_config.pilot_cache_prefix}{system_config.pilot_count}"
    )
    if os.path.isfile(pilot_file_name):
        print(f"Loading cached pilots from {pilot_file_name}")
        bits = np.loadtxt(pilot_file_name, delimiter=",")
    else:
        bits = np.random.binomial(n=1, p=0.5, size=(system_config.num_subcarriers * system_config.bits_per_symbol,))
        np.savetxt(pilot_file_name, bits, delimiter=",")
        print(f"Saved new pilot sequence to {pilot_file_name}")
    return modulation(bits, system_config.bits_per_symbol)


def load_channel_responses(path: str, start: int, end: int) -> List[np.ndarray]:
    """Load channel response vectors from text files in the dataset path."""
    channel_responses: List[np.ndarray] = []
    for idx in range(start, end):
        print(f"Processing the {idx}th document")
        file_path = os.path.join(path, f"{idx}.txt")
        with open(file_path, encoding="utf-8") as file_handle:
            for line in file_handle:
                values = [float(x) for x in line.split()]
                midpoint = int(len(values) / 2)
                real = np.asarray(values[:midpoint])
                imag = np.asarray(values[midpoint:])
                channel_responses.append(real + 1j * imag)
    return channel_responses


def build_autoencoder_graph(input_dimension: int, output_dimension: int, hidden_units: Sequence[int]):
    """Construct the TensorFlow graph for the autoencoder."""
    X = tf.compat.v1.placeholder(tf.float32, [None, input_dimension], name="input_samples")
    Y = tf.compat.v1.placeholder(tf.float32, [None, output_dimension], name="target_bits")
    learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[], name="learning_rate")

    weights = {
        "encoder_h1": tf.Variable(tf.random.truncated_normal([input_dimension, hidden_units[0]], stddev=0.1)),
        "encoder_h2": tf.Variable(tf.random.truncated_normal([hidden_units[0], hidden_units[1]], stddev=0.1)),
        "encoder_h3": tf.Variable(tf.random.truncated_normal([hidden_units[1], hidden_units[2]], stddev=0.1)),
        "encoder_h4": tf.Variable(tf.random.truncated_normal([hidden_units[2], output_dimension], stddev=0.1)),
    }
    biases = {
        "encoder_b1": tf.Variable(tf.random.truncated_normal([hidden_units[0]], stddev=0.1)),
        "encoder_b2": tf.Variable(tf.random.truncated_normal([hidden_units[1]], stddev=0.1)),
        "encoder_b3": tf.Variable(tf.random.truncated_normal([hidden_units[2]], stddev=0.1)),
        "encoder_b4": tf.Variable(tf.random.truncated_normal([output_dimension], stddev=0.1)),
    }

    layer_1 = tf.nn.relu(tf.add(tf.matmul(X, weights["encoder_h1"]), biases["encoder_b1"]))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights["encoder_h2"]), biases["encoder_b2"]))
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights["encoder_h3"]), biases["encoder_b3"]))
    y_pred = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights["encoder_h4"]), biases["encoder_b4"]))

    cost = tf.reduce_mean(tf.pow(Y - y_pred, 2))
    optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    return X, Y, y_pred, learning_rate, optimizer, cost


def train_model(
    system_config: OFDMSystemConfig | None = None, training_config: TrainingConfig | None = None
) -> None:
    """Run the end-to-end training loop."""
    system_config = system_config or OFDMSystemConfig()
    training_config = training_config or TrainingConfig()
    pilot_value = load_or_create_pilot(system_config, training_config)
    all_carriers = np.arange(system_config.num_subcarriers)

    print("Loading channel responses ...")
    channel_response_train = load_channel_responses(
        training_config.train_dataset_path, training_config.train_channel_start, training_config.train_channel_end
    )
    channel_response_test = load_channel_responses(
        training_config.test_dataset_path, training_config.test_channel_start, training_config.test_channel_end
    )
    print(
        f"Channel responses loaded - train: {len(channel_response_train)}, "
        f"test: {len(channel_response_test)}"
    )

    input_dimension = system_config.num_subcarriers * 4
    X, Y, y_pred, learning_rate, optimizer, cost = build_autoencoder_graph(
        input_dimension=input_dimension,
        output_dimension=training_config.output_bits,
        hidden_units=training_config.hidden_units,
    )

    tf.compat.v1.set_random_seed(42)
    init = tf.compat.v1.global_variables_initializer()
    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=session_config) as sess:
        sess.run(init)
        learning_rate_current = training_config.learning_rate

        for epoch in range(training_config.training_epochs):
            if epoch > 0 and epoch % training_config.learning_rate_decay_step == 0:
                learning_rate_current /= training_config.learning_rate_decay_factor

            average_cost = 0.0
            for _ in range(training_config.batch_iterations):
                input_samples: List[np.ndarray] = []
                input_labels: List[np.ndarray] = []
                for _ in range(training_config.batch_examples):
                    bits = np.random.binomial(
                        n=1, p=0.5, size=(system_config.payload_bits_per_symbol,)
                    )
                    channel_response = channel_response_train[
                        np.random.randint(0, len(channel_response_train))
                    ]
                    signal_output, _ = simulate_ofdm(
                        bits, channel_response, system_config, pilot_value, all_carriers
                    )
                    input_samples.append(signal_output)
                    input_labels.append(bits[16:32])
                batch_x = np.asarray(input_samples)
                batch_y = np.asarray(input_labels)
                _, batch_cost = sess.run(
                    [optimizer, cost],
                    feed_dict={X: batch_x, Y: batch_y, learning_rate: learning_rate_current},
                )
                average_cost += batch_cost / training_config.batch_iterations

            if epoch % training_config.display_step == 0:
                print(f"Epoch: {epoch:04d} | cost = {average_cost:.9f}")
                test_number = training_config.test_batch_examples
                if epoch % training_config.evaluation_step == 0:
                    print("Using extended evaluation set for this epoch.")
                    test_number = training_config.extended_test_examples

                input_samples_test: List[np.ndarray] = []
                input_labels_test: List[np.ndarray] = []
                for _ in range(test_number):
                    bits = np.random.binomial(
                        n=1, p=0.5, size=(system_config.payload_bits_per_symbol,)
                    )
                    channel_response = channel_response_test[
                        np.random.randint(0, len(channel_response_test))
                    ]
                    signal_output, _ = simulate_ofdm(
                        bits, channel_response, system_config, pilot_value, all_carriers
                    )
                    input_samples_test.append(signal_output)
                    input_labels_test.append(bits[16:32])

                batch_x = np.asarray(input_samples_test)
                batch_y = np.asarray(input_labels_test)
                mean_error = tf.reduce_mean(tf.abs(y_pred - batch_y))
                mean_error_rate = 1 - tf.reduce_mean(
                    tf.reduce_mean(
                        tf.cast(tf.equal(tf.sign(y_pred - 0.5), tf.sign(batch_y - 0.5)), tf.float32), 1
                    )
                )
                test_error, test_error_rate = sess.run(
                    [mean_error, mean_error_rate], feed_dict={X: batch_x, Y: batch_y}
                )
                print(
                    f"Validation: output_bits={training_config.output_bits}, "
                    f"SNR={system_config.snr_db}, pilots={system_config.pilot_count}, "
                    f"mean error={test_error:.6f}, bit error rate={test_error_rate:.6f}"
                )

                train_eval_samples: List[np.ndarray] = []
                train_eval_labels: List[np.ndarray] = []
                for _ in range(training_config.test_batch_examples):
                    bits = np.random.binomial(
                        n=1, p=0.5, size=(system_config.payload_bits_per_symbol,)
                    )
                    channel_response = channel_response_train[
                        np.random.randint(0, len(channel_response_train))
                    ]
                    signal_output, _ = simulate_ofdm(
                        bits, channel_response, system_config, pilot_value, all_carriers
                    )
                    train_eval_samples.append(signal_output)
                    train_eval_labels.append(bits[16:32])

                train_batch_x = np.asarray(train_eval_samples)
                train_batch_y = np.asarray(train_eval_labels)
                mean_error_train = tf.reduce_mean(tf.abs(y_pred - train_batch_y))
                mean_error_rate_train = 1 - tf.reduce_mean(
                    tf.reduce_mean(
                        tf.cast(tf.equal(tf.sign(y_pred - 0.5), tf.sign(train_batch_y - 0.5)), tf.float32), 1
                    )
                )
                train_error, train_error_rate = sess.run(
                    [mean_error_train, mean_error_rate_train], feed_dict={X: train_batch_x, Y: train_batch_y}
                )
                print(
                    f"Training sample preview: mean error={train_error:.6f}, "
                    f"bit error rate={train_error_rate:.6f}"
                )

        print("Optimization finished.")


ofdm_config = OFDMSystemConfig()
training_config = TrainingConfig()

if __name__ == "__main__":
    train_model(ofdm_config, training_config)
