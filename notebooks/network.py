import pytorch_lightning
import torch

import monai
from monai.data import list_data_collate
from monai.networks.layers import Norm

from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism


class MRILiverSegmentation(pytorch_lightning.LightningModule):

    def __init__(self):
        super().__init__()
        
        self._model = monai.networks.nets.UNet(
            dimensions=3, 
            in_channels=1, 
            out_channels=2, 
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2), 
            num_res_units=2, 
            norm=Norm.BATCH)
        
        self.loss_function = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        
        self.val_metric = DiceMetric(include_background=True, reduction="mean")

        self.best_val_dice = None
        self.best_val_epoch = None
        
        
    def forward(self, x):
        return self._model(x)

    
    def prepare_data(self):
        # set deterministic training for reproducibility
        set_determinism(seed=0)
    
    
    def training_step(self, batch, batch_idx):
        images, labels = batch["IMAGE"], batch["SEGM"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        return {"loss": loss}
    
    
    def training_epoch_end(self, outputs):
        # Only add the graph at the first epoch
        if self.current_epoch==1:
            sample_input = torch.rand((1,1,64,64,32))
            self.logger.experiment.add_graph(
                MRILiverSegmentation(),
                [sample_input]
            )
        
        # Calculate the average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # Logging at the end of every epoch
        self.logger.experiment.add_scalar('Loss/Train', avg_loss, self.current_epoch)
        return {'loss': avg_loss}
    
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch["IMAGE"], batch["SEGM"]
        roi_size = (224, 224, 128)
        sw_batch_size = 4
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)

        loss = self.loss_function(outputs, labels)
        value = self.val_metric(y_pred=outputs, y=labels)

        return {"val_loss": loss, "val_dice": value}


    
    def validation_epoch_end(self, outputs):
        # Calculate the average loss
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_dice = torch.stack([x['val_dice'] for x in outputs]).mean()
        # Logging at the end of every epoch
        self.logger.experiment.add_scalar('Loss/Val', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('Dice/Val', avg_dice, self.current_epoch)

        # Handle the first time
        if self.best_val_dice is None:
            self.best_val_dice = avg_dice
            self.best_val_epoch = self.current_epoch
        if avg_loss < self.best_val_dice:
            self.best_val_dice = avg_dice
            self.best_val_epoch = self.current_epoch
        return {"val_loss": avg_loss, "val_dice": avg_dice}


#     def test_step(self, batch, batch_idx):
#         images, labels = batch["IMAGE"], batch["SEGM"]
#         roi_size = (224, 224, 128)
#         sw_batch_size = 4
#         outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)

#         value = self.val_metric(y_pred=outputs, y=labels)
#         return {"test_dice": value}

    
#     def test_epoch_end(self, outputs):
#         test_dice, num_items = 0, 0
#         for output in outputs:
#             test_dice += output["test_dice"].sum().item()
#             num_items += len(output["test_dice"])
#         mean_test_dice = torch.tensor(test_dice / num_items)
#         tensorboard_logs = {
#             "test_dice": mean_test_dice,
#         }
#         print(
#             f"current epoch: {self.current_epoch} current test mean dice: {mean_test_dice:.4f}"
#         )
#         return {"log": tensorboard_logs}

    
    def configure_optimizers(self):
        return torch.optim.Adam(self._model.parameters(), lr=1e-4)
    
    