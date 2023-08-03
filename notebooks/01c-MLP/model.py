import torch
from torch import nn, optim
import pytorch_lightning as pl
import torchvision
import torchmetrics

class MLP(pl.LightningModule):
    def __init__(self, input_feature_size, layer_num, hidden_dim, learning_rate, dropout_rate):
        super().__init__()
        self.lr = learning_rate
        self.mlp = torchvision.ops.MLP(input_feature_size,layer_num*[hidden_dim],dropout=dropout_rate,inplace=False)
        self.linear = nn.Linear(hidden_dim, 1)
        self.loss_fn = nn.MSELoss()
        self.mse_metric = torchmetrics.MeanSquaredError()

    def forward(self, x):
        y = self.mlp(x)
        y = self.linear(y)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)

        self.log_dict(
            {
                "train_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss, "scores": scores, "y": y}
    
    def training_epoch_end(self, outputs):
        scores = torch.cat([x["scores"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])
        self.log_dict(
            {
                "train_MSE_loss": self.mse_metric(scores, y)
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def validation_step(self, batch, batch_idx, dataset_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        if dataset_idx==0:
            postfix='CEX'
        elif dataset_idx==1:
            postfix='GEX/DEX'
        elif dataset_idx==2:
            postfix='AEX'
        elif dataset_idx==3:
            postfix='RND'
        self.log(f"val_{postfix}_MSE_loss", loss,sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("test_mse_loss", loss,sync_dist=True)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
