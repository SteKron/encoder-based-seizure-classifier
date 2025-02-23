import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from ancpbids.pybids_compat import BIDSLayout

from datasets.bids import BIDSEEGData  
from models.seizureclassifier import SeizureClassifier  

def run_fold(data_folder: str,
             encoder_ckpt: str,
             batch_size: int,
             segment_size: int,
             stride: int,
             max_epochs: int,
             lr: float,
             num_channels: int,
             test_subject: str,
             all_subjects: list) -> dict:
    
    train_subjects = [s for s in all_subjects if s != test_subject]
    
    if len(train_subjects) > 1:
        val_subjects = [train_subjects[0]]

        train_subjects = train_subjects[1:]
    else:
        val_subjects = train_subjects

    train_patients = [train_subjects]
    val_patients   = [val_subjects]
    test_patients  = [[test_subject]]
    
    datamodule = BIDSEEGData(
        folders=[data_folder],
        batch_size=batch_size,
        train_patients=train_patients,
        val_patients=val_patients,
        test_patients=test_patients,
        segment_size=segment_size,
        stride=stride,
        num_workers=4 
    )
    
    model = SeizureClassifier(
        encoder_ckpt=encoder_ckpt,
        lr=lr,
        num_channels=num_channels
    )

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=False  # Could use wandb here
    )
    
    # Train the model
    trainer.fit(model, datamodule=datamodule)
    
    # Evaluate on the test set
    test_results = trainer.test(model, datamodule=datamodule)

    return test_results[0]

def main(args):
    # Use BIDSLayout to list all subjects in the dataset
    layout = BIDSLayout(args.data_folder)
    all_subjects = layout.get_subjects()
    print(f"Found subjects: {all_subjects}")
    
    fold_metrics = []
    # Loop over each subject to use it as the test set
    for test_subject in all_subjects:
        print(f"\n===== Running LOOCV fold with test subject: {test_subject} =====")
        fold_result = run_fold(
            data_folder=args.data_folder,
            encoder_ckpt=args.encoder_ckpt,
            batch_size=args.batch_size,
            segment_size=args.segment_size,
            stride=args.stride,
            max_epochs=args.max_epochs,
            lr=args.lr,
            num_channels=args.num_channels,
            test_subject=test_subject,
            all_subjects=all_subjects
        )
        print(f"Results for subject {test_subject}: {fold_result}")
        fold_metrics.append(fold_result)
    
    # Aggregate metrics across all folds
    avg_metrics = {}
    keys = fold_metrics[0].keys()
    for key in keys:
        avg_metrics[key] = sum(fold[key] for fold in fold_metrics) / len(fold_metrics)
    
    print("\n===== LOOCV Average Metrics =====")
    for key, value in avg_metrics.items():
        print(f"{key}: {value}")
    
    # Write the aggregated metrics to a file
    with open("loocv_metrics.txt", "w") as f:
        for key, value in avg_metrics.items():
            f.write(f"{key}: {value}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LOOCV Training for SeizureClassifier")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the BIDS dataset folder")
    parser.add_argument("--encoder_ckpt", type=str, required=True, help="Path to pre-trained encoder checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--segment_size", type=int, default=4000, help="Segment size (in ms)")
    parser.add_argument("--stride", type=int, default=1000, help="Stride (in ms)")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_channels", type=int, default=18, help="Number of EEG channels")
    
    args = parser.parse_args()
    main(args)