"""Command line entrypoint for OFDM channel estimation training."""

import argparse

from OFDM_Deep_neural_network import OFDMSystemConfig, TrainingConfig, train_model


def parse_args() -> argparse.Namespace:
    """Create the argument parser for basic training overrides."""
    parser = argparse.ArgumentParser(description="Train OFDM channel estimation network.")
    parser.add_argument("--train-dataset", default="../H_dataset/", help="Path to the training dataset directory.")
    parser.add_argument("--test-dataset", default="../H_dataset/", help="Path to the test dataset directory.")
    parser.add_argument("--snr-db", type=int, default=20, help="Signal-to-noise ratio used during training.")
    parser.add_argument("--pilot-count", type=int, default=64, help="Number of pilots per OFDM block.")
    parser.add_argument(
        "--training-epochs", type=int, default=20000, help="Number of training epochs for the autoencoder."
    )
    parser.add_argument(
        "--batch-examples",
        type=int,
        default=1000,
        help="Number of randomly generated examples per mini-batch iteration.",
    )
    return parser.parse_args()


def main() -> None:
    """Build configuration objects and kick off training."""
    args = parse_args()
    system_config = OFDMSystemConfig(snr_db=args.snr_db, pilot_count=args.pilot_count)
    training_config = TrainingConfig(
        train_dataset_path=args.train_dataset,
        test_dataset_path=args.test_dataset,
        training_epochs=args.training_epochs,
        batch_examples=args.batch_examples,
    )
    train_model(system_config, training_config)


if __name__ == "__main__":
    main()
