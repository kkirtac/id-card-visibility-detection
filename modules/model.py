"""Class definition for Resnet convolutional neural networks. 
18 layered Resnet is suported in this implementation.
"""
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import precision_recall_fscore_support


class Resnet(nn.Module):
    """Resnet class definition.
    
    Attributes:
        resnet: An instance of torchvision.models.resnet18.
        head: Creates the head network on top of the resnet backbone. 
         The purpose of creating this head is to adapt the outputs of the network to the given dataset.
    """

    def __init__(self, pretrained: bool = False, num_outputs: int = 3):
        """Initializes the Resnet instance.

        Args:
            pretrained (bool, optional): True if the network weights should be initialized by ImageNet.
             Defaults to False.
            num_outputs (int, optional): number of output units. Defaults to 3 (Visibility labels).
        """
        super().__init__()

        # base network
        self.resnet = models.resnet18(pretrained=pretrained)

        self.head = nn.Linear(self.resnet.fc.in_features, out_features=num_outputs)

        #change the behavior of old fc layer
        self.resnet.fc = Identity()

    def forward(self, x):
        """ Computation performed during the forward pass """
        x = self.resnet(x)
        x = self.head(x)

        return x


class Identity(nn.Module):
    """Identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResnetLightning(pl.LightningModule):
    """ResnetLightning class definition.
    
    Attributes:
        loss_fn: An instance of nn.CrossEntropyLoss. Computes per frame phase classification error.
        network: An instance of Resnet. (See self.create_network())
    """

    def __init__(self, learning_rate: float, out_features=3, **kwargs) -> None:
        """Instantiates the ResnetLightning class.

        Args:
            learning_rate (float): learning rate of the optimization.
            out_features (int): number of output units (num classes).
        """
        super().__init__(**kwargs)

        # call this to save num_resnet_layers, out_features, learning_rate and class_weights to the checkpoint.
        self.save_hyperparameters()
        # Now possible to access out_features from self.hparams

        self.loss_fn = nn.CrossEntropyLoss()
        self.network = self.create_network()

    def create_network(self) -> Resnet:
        """Creates the deep net model.

        Returns:
            Resnet: The created instance of the Resnet model.
        """
        network = Resnet(pretrained=True, num_outputs=self.hparams.out_features)

        return network

    def get_finetuning_params(self) -> List:
        """Returns finetuning parameters with corresponding learning rates.

        Returns:
            List: list of fine tuning parameters.
             Each list element is a dict of parameter and its learning rate.
        """
        params = []

        for idx, module in enumerate(self.network.named_modules()):
            print(idx, '->', module)

        base_lr = self.hparams.learning_rate

        for name, module in self.network.named_modules():
            if name == 'head':
                params.append({"params": module.parameters(), "name": name, "lr": base_lr})

        return params

    def forward(self, X: torch.tensor) -> torch.tensor:
        """Forward pass of the deep network. Overrides pl.LightningModule.forward().

        Args:
            X (torch.tensor): Input tensor.
                X -> (N=batch_size, C=num_channels, H=height, W=width)

        Returns:
            torch.tensor: network outputs.
        """
        return self.network(X)

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], Dict]:
        """Configures optimizers. Overrides pl.LightningModule.configure_optimizers().

        Returns:
            Tuple[List[optim.Optimizer], Dict]: List of Optimizer objects and a corresponding dictionary for the scheduler.
        """
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.learning_rate, weight_decay=5e-4, momentum=0.9)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=5, mode='min')
        scheduler_dict = [{'scheduler': scheduler, 'monitor': 'val_loss', 'interval': 'epoch', 'frequency': 1}]

        return [optimizer], scheduler_dict

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.tensor:
        """Overrides pl.LightningModule.training_step().
        A single step of training optimization.

        Args:
            batch (Tuple): (X,y), i.e., tuple of input and target tensors.
                X -> (N=batch_size, C=num_channels, H=height, W=width)
                y -> (N=batch_size, )
            batch_idx (int): batch index in the current epoch.

        Returns:
            torch.tensor: the loss tensor.
        """
        x, y = batch
        y_hat = self.network(x)
        loss = self.loss_fn(y_hat, y)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict:
        """Overrides pl.LightningModule.validation_step().
        Runs the single step of validation with current data batch.
        Computes validation loss and predictions for each input image in the batch.

        Args:
            batch (Tuple): (X,y), i.e., tuple of input and target tensors.
            batch_idx (int): batch index in the current epoch.

        Returns:
            Dict: dict of predictions, scores, targets, video name and frame indices.
        """
        y_pred, y_true, loss = self.compute_prediction_for_batch(batch, batch_idx)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'val_y_pred': y_pred, 'val_y_true': y_true, 'val_loss': loss}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """Overrides pl.LightningModule.validation_epoch_end().
        Called at the end of the validation epoch with the outputs of all validation steps.
        Logs the computed metrics to the logger.

        Args:
            outputs (List[Any]): List of outputs defined in :meth:`validation_step`.
        """
        y_pred, y_true = self.aggregate_prediction_from_batches(outputs, 'val')

        agg_loss = torch.cat([o['val_loss'].view(-1, 1) for o in outputs])

        accuracy = (y_pred == y_true).mean()

        precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2])

        results = {
            'val_precision_mean': torch.tensor(precision.mean()),
            'val_recall_mean': torch.tensor(recall.mean()),
            'val_fscore_mean': torch.tensor(fscore.mean()),
            'val_accuracy': torch.tensor(accuracy),
            'val_loss_mean': torch.mean(agg_loss),
            'val_precision_full_visibility': torch.tensor(precision[0]),
            'val_recall_full_visibility': torch.tensor(recall[0]),
            'val_fscore_full_visibility': torch.tensor(fscore[0]),
            'val_precision_partial_visibility': torch.tensor(precision[1]),
            'val_recall_partial_visibility': torch.tensor(recall[1]),
            'val_fscore_partial_visibility': torch.tensor(fscore[1]),
            'val_precision_no_visibility': torch.tensor(precision[2]),
            'val_recall_no_visibility': torch.tensor(recall[2]),
            'val_fscore_no_visibility': torch.tensor(fscore[2]),
        }

        self.log_dict(results, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch: Tuple, batch_idx: int) -> Dict:
        """Overrides pl.LightningModule.test_step(). Operates on a single batch of data from the test set.

        Args:
            batch (Tuple): (X,y), i.e., tuple of input and target tensors.
            batch_idx (int): batch index in the current epoch.

        Returns:
            Dict: dict of predictions, scores, targets, video name and frame indices.
        """
        y_pred, y_true, loss = self.compute_prediction_for_batch(batch, batch_idx)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'test_y_pred': y_pred, 'test_y_true': y_true, 'test_loss': loss}

    def test_epoch_end(self, outputs: List[Any]) -> None:
        """Overrides pl.LightningModule.test_epoch_end().
        Called at the end of a test epoch with the output of all test steps. 
        Aggregate the outputs from all test steps.
        Logs test performance metrics.

        Args:
            outputs: List of outputs you defined in :meth:`test_step_end`.

        Note:
            If you didn't define a :meth:`test_step`, this won't be called.
        """
        y_pred, y_true = self.aggregate_prediction_from_batches(outputs, 'test')

        agg_loss = torch.cat([o['test_loss'].view(-1, 1) for o in outputs])

        accuracy = (y_pred == y_true).mean()

        precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2])

        results = {
            'test_precision_mean': torch.tensor(precision.mean()),
            'test_recall_mean': torch.tensor(recall.mean()),
            'test_fscore_mean': torch.tensor(fscore.mean()),
            'test_accuracy': torch.tensor(accuracy),
            'test_loss_mean': torch.mean(agg_loss),
            'test_precision_full_visibility': torch.tensor(precision[0]),
            'test_recall_full_visibility': torch.tensor(recall[0]),
            'test_fscore_full_visibility': torch.tensor(fscore[0]),
            'test_precision_partial_visibility': torch.tensor(precision[1]),
            'test_recall_partial_visibility': torch.tensor(recall[1]),
            'test_fscore_partial_visibility': torch.tensor(fscore[1]),
            'test_precision_no_visibility': torch.tensor(precision[2]),
            'test_recall_no_visibility': torch.tensor(recall[2]),
            'test_fscore_no_visibility': torch.tensor(fscore[2]),
        }

        self.log_dict(results, on_epoch=True, prog_bar=True, logger=True)

    def compute_prediction_for_batch(self, batch: Tuple, batch_idx: int) -> Tuple:
        """Compute predictions and softmax scores for the validation and test batches.

        Args:
            batch (Tuple): (X,y), i.e., tuple of input and target tensors.
            batch_idx (int): batch index in the current epoch.

        Returns:
            Tuple: a tuple of the following tensors, plus the loss tensor,
             top1_class_idx: (n_samples, 1) -> predictions
             target: (n_samples,) -> targets
        """
        x, y = batch
        y_hat = self.network(x)

        loss = self.loss_fn(y_hat, y)

        # get the index of the top-1 class
        _, top1_class_idx = y_hat.topk(1)

        return top1_class_idx.cpu(), y.cpu(), loss

    def aggregate_prediction_from_batches(self, outputs: List, prefix: str) -> Tuple:
        """Aggregate the output tensors generated from several batches.

        Args:
            outputs (List): List of batch predictions, scores, targets, frame indices and video names
             as tensors generated by validation_step or test_step.
            prefix (str): 'val' or 'test' for respectively 'val' and 'test'

        Returns:
            Tuple: tuple of aggregated outputs. Each output is returned as a numpy array. 
        """
        y_true_all = []
        y_pred_all = []

        for o in outputs:
            y_true_all.append(o[f'{prefix}_y_true'])
            y_pred_all.append(o[f'{prefix}_y_pred'])

        y_pred = torch.cat(y_pred_all).cpu().numpy().ravel()
        y_true = torch.cat(y_true_all).cpu().numpy().ravel()
        # y_pred -> (num_samples,)
        # y_true -> (num_samples,)

        return y_pred, y_true
