import argparse
import ast
import torch
import pytorch_lightning as pl

from datasets.bids import BIDSEEGData  
from models.seizureclassifier import SeizureClassifier  

def main(args):
    
    try:
        data_patients = ast.literal_eval(args.data_patients)
    except Exception as e:
        raise ValueError(
            "Invalid format for data_patients. It should be a string representation of a list-of-lists, e.g. \"[['01', '02']]\""
        ) from e

    # Wrap data_folders in a list 
    data_folder = [args.data_folder] if isinstance(args.data_folder, str) else args.data_folder

    train_patients = data_patients
    val_patients = data_patients
    test_patients = data_patients

    datamodule = BIDSEEGData(
        folders=data_folder,
        batch_size=args.batch_size,
        train_patients=train_patients,
        val_patients=val_patients,
        test_patients=test_patients,
        segment_size=args.segment_size,
        stride=args.stride,
        num_workers=args.num_workers
    )

    model = SeizureClassifier(
        encoder_ckpt=args.encoder_ckpt,
        lr=args.lr,
        num_channels=args.num_channels
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu", 
        devices=1, 
    )

    if args.command == "train":
        trainer.fit(model, datamodule=datamodule)
    elif args.command == "test":
        trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training or testing for the SeizureClassifier"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Pipeline to run: train or test")
    
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--data_folder", type=str, required=True, 
                               help="Path to the BIDS dataset folder")
    common_parser.add_argument("--encoder_ckpt", type=str, required=True, 
                               help="Path to the pre-trained encoder checkpoint")
    common_parser.add_argument("--data_patients", type=str, required=True, 
                               help="Subset of patients to use, e.g. \"[['01', '02']]\"")
    common_parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    common_parser.add_argument("--segment_size", type=int, default=4000, help="Segment size (in ms)")
    common_parser.add_argument("--stride", type=int, default=1000, help="Stride (in ms)")
    common_parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs")
    common_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    common_parser.add_argument("--num_channels", type=int, default=18, help="Number of EEG channels")
    common_parser.add_argument("--num_workers", type=int, default=1, help="Number of data loader workers")

    # Subparser for training
    train_parser = subparsers.add_parser("train", parents=[common_parser], help="Run training pipeline")
    # Subparser for testing only
    test_parser = subparsers.add_parser("test", parents=[common_parser], help="Run testing pipeline")

    args = parser.parse_args()
    main(args)