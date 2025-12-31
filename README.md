# OFDM Channel Estimation with Deep Learning

This repository demonstrates how to train a deep neural network to perform channel estimation and symbol detection for an OFDM system. The training loop simulates OFDM transmissions with pilots and random payloads, applies a learned estimator, and reports reconstruction metrics throughout training.

## Project layout

- `main.py`: Command-line entrypoint for running training with common configuration overrides.
- `OFDM_Deep_neural_network.py`: Core simulation utilities and TensorFlow 1.x training loop.
- `OFDM_ChannelEstimation_DeepLearning_QAM_random_pilot.py`: Alternative experimental script for QAM-based setups.

## Prerequisites

- Python 3.10+
- TensorFlow 1.x (the code uses the TensorFlow 1.x graph API)
- NumPy

Install the dependencies with pip (adjust the TensorFlow version based on your environment and hardware):

```bash
pip install "tensorflow<2" numpy
```

## Dataset

The training loop expects channel response files named `1.txt`, `2.txt`, … in the `H_dataset` directory. You can download the dataset from the original source:

```
https://github.com/haoyye/OFDM_DNN/blob/423775dd6e57f52ad4bc9511b60b60f99a24fdac/H_dataset/H_dataset.zip.001
```

Unzip the archive and place the `H_dataset` directory next to the repository root or update the paths via command-line arguments.

## Running training

With the dataset available, start training with the default configuration:

```bash
python main.py
```

Common options can be overridden from the CLI:

- `--train-dataset`: Path to the folder containing training channel response files (default: `../H_dataset/`).
- `--test-dataset`: Path to the folder containing test channel response files (default: `../H_dataset/`).
- `--snr-db`: Signal-to-noise ratio applied during simulation (default: `20`).
- `--pilot-count`: Number of pilot subcarriers per OFDM block (default: `64`).
- `--training-epochs`: Total number of training epochs (default: `20000`).
- `--batch-examples`: Number of random examples generated for each mini-batch iteration (default: `1000`).

Example with custom SNR and dataset locations:

```bash
python main.py --snr-db 15 --pilot-count 32 --train-dataset ./data/H_dataset --test-dataset ./data/H_dataset
```

## Notes

- Pilots are generated once and cached in the repository root (e.g., `Pilot_64`); delete the file to regenerate pilots.
- The training loop prints both validation and training preview metrics at configurable intervals so you can monitor convergence.
- The TensorFlow 1.x session configuration enables GPU memory growth to better utilize available hardware.
