import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from .encodec import SEANetEncoder


class SpatialFusionBlock(nn.Module):
    def __init__(self, eeg_channels, embedding_dim, num_embeddings, mlp_hidden_dim):
        super().__init__()
        
        # 2D Convolution for spatial fusion
        self.conv2d = nn.Conv2d(
            in_channels=64,  # Input is treated as 64 2D channels of dimensions: 18 spatial x 128 temporal
            out_channels=64,   # Output is 64 channels of 1D spatio-temporal features
            kernel_size=(eeg_channels, 1),  # Cover all spatial channels and slides along time
            stride=(1, 1),  # Slide only along the time dimension
            padding="valid"  # No padding needed since we cover full channels
        )
        
        # MLP for hierarchical spatiotemporal features representation
        self.mlp = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Linear(int(num_embeddings * embedding_dim / 2), mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim), 
            nn.ReLU(),
        )

    def forward(self, x):  
        x = self.conv2d(x) 
        x = torch.flatten(x, start_dim=1)
        x = self.mlp(x)
        return x

class LinearClassifier(nn.Module):
    
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

class SeizureClassifier(pl.LightningModule):
    def __init__(
        self,
        encoder_ckpt: str,
        lr: float = 1e-3,
        num_channels: int = 18,
        embedding_dim: int = 64,
        num_embeddings: int = 128,  # output length from the encoder per channel
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["encoder_ckpt"])

        # Load pre-trained encoder
        self.encoder = SEANetEncoder(
            dimension=64,
            n_filters=16,
            max_filters=256,
            ratios=[2, 2, 2, 2],
            norm="none",
            kernel_size=3,
            last_kernel_size=3,
            causal=False,
            true_skip=True,
            lstm=0,
        )
        self._load_encoder_weights(encoder_ckpt)

        self.channel_fusion = SpatialFusionBlock(
            eeg_channels=num_channels,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            mlp_hidden_dim=num_embeddings,
        )

        # Linear classifier
        self.classifier = LinearClassifier(input_dim=num_embeddings, output_dim=1)
        
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()

        # Logging metrics
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()

        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_f1 = BinaryF1Score()

    def _load_encoder_weights(self, encoder_ckpt):

        encoder_dict = self.encoder.state_dict()
        pt = torch.load(encoder_ckpt)
        pretrained_dict = {k: v for k, v in pt["state_dict"].items() if k in encoder_dict}
        encoder_dict.update(pretrained_dict)
        self.encoder.load_state_dict(encoder_dict)
        self.encoder.eval()  # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _common_step(self, batch, mode="train"):

        x, y = batch.data, batch.labels
        logits = self(x)
        loss = self.criterion(logits, y.float())

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        return loss, preds, y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = x.shape
        # Merge batch and channel dimensions
        x = x.view(-1, 1, length)  # (batch * num_channels, 1, length)
        # Use the frozen encoder without computing gradients
        with torch.no_grad():
            embeddings = self.encoder(x)
        # Reshape back to (batch, num_channels, embedding_length)
        embeddings = embeddings.view(
            batch_size,
            self.hparams.embedding_dim,
            self.hparams.num_channels,
            self.hparams.num_embeddings,
        )  # (batch, embedding_length, num_channels, num_embeddings)
        fused_features = self.channel_fusion(embeddings)  # (batch, embedding_length)
        logits = self.classifier(fused_features)
        return logits

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._common_step(batch)
        self.train_acc(preds, y)
        self.log_dict({
            'train_loss': loss,
            'train_acc': self.train_acc
        }, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._common_step(batch)
        self.val_acc(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_f1(preds, y)
        self.log_dict({
            'val_loss': loss,
            'val_acc': self.val_acc,
            'val_precision': self.val_precision,
            'val_recall': self.val_recall,
            'val_f1': self.val_f1
        }, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._common_step(batch)
        self.test_acc(preds, y)
        self.test_precision(preds, y)
        self.test_recall(preds, y)
        self.test_f1(preds, y)
        self.log_dict({
            'test_loss': loss,
            'test_acc': self.test_acc,
            'test_precision': self.test_precision,
            'test_recall': self.test_recall,
            'test_f1': self.test_f1
        }, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
