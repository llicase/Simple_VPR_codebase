
import torch
import numpy as np
import torchvision.models
import pytorch_lightning as pl
from torchvision import transforms as tfm
from pytorch_metric_learning import losses
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import logging
from os.path import join
from pytorch_metric_learning.losses import SelfSupervisedLoss
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.distances import LpDistance

import utils
import parser
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset


class GeM(torch.nn.Module):
    def __init__(self, p=2.5, eps=1e-6):
        super(GeM,self).__init__()
        self.p = torch.nn.Parameter(torch.ones(1)*p)
        self.eps = eps
    
    def forward(self, x):
        return torch.nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

class LightningModel(pl.LightningModule):
    def __init__(self, val_dataset, test_dataset, descriptors_dim=512, num_preds_to_save=0, save_only_wrong_preds=True, 
                 alpha_param=1, beta_param=50, base_param=0.0, eps_param=0.1, opt_param="sgd", loss_param="cl", 
                 pool_param="None", miner_param="None", lr_adam_param=0.0001):
        super().__init__()
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_preds_to_save = num_preds_to_save
        self.save_only_wrong_preds = save_only_wrong_preds
        self.opt_param = opt_param
        self.loss_param = loss_param
        self.pool_param = pool_param
        self.miner_param = miner_param
        self.lr_adam_param = lr_adam_param
        
        #set the miner
        if self.miner_param == "ms":
            self.miner = miners.MultiSimilarityMiner(epsilon=eps_param, distance=CosineSimilarity())
        
        # Use a pretrained model
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        
        #set the kind of pooling
        if self.pool_param == "gem":
            self.model.avgpool= GeM()
        
        # Change the output of the FC layer to the desired descriptors dimension
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, descriptors_dim)
        
        # Set the loss function
        if self.loss_param == "cl":
            self.loss_fn = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
        elif self.loss_param == "ms":
            self.loss_fn = losses.MultiSimilarityLoss(alpha=alpha_param, beta=beta_param, base=base_param)


    def forward(self, images):
        descriptors = self.model(images)
        return descriptors

    def configure_optimizers(self):
        if self.opt_param == "sgd":
            optimizers = torch.optim.SGD(self.parameters(), lr=0.001, weight_decay=0.001, momentum=0.9)
        elif self.opt_param == "adam":
            optimizers = torch.optim.Adam(self.parameters(), lr=self.lr_adam_param, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        elif self.opt_param == "adamw":
            optimizers = torch.optim.AdamW(self.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        elif self.opt_param == "adamax":
            optimizers = torch.optim.Adamax(self.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        elif self.opt_param == "amsgrad":
            optimizers = torch.optim.Adam(self.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
        return optimizers

    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels, triplets):
        loss = self.loss_fn(descriptors, labels, triplets)
        return loss

    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        images, labels = batch
        num_places, num_images_per_place, C, H, W = images.shape
        images = images.view(num_places * num_images_per_place, C, H, W)
        labels = labels.view(num_places * num_images_per_place)

        # Feed forward the batch to the model
        descriptors = self(images)  # Here we are calling the method forward that we defined above
        triplets = self.miner(descriptors, labels)
        loss = self.loss_function(descriptors, labels)  # Call the loss_function we defined above
        
        self.log('loss', loss.item(), logger=True)
        return {'loss': loss}

    # For validation and test, we iterate step by step over the validation set
    def inference_step(self, batch):
        images, _ = batch
        descriptors = self(images)
        return descriptors.cpu().numpy().astype(np.float32)

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def test_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def validation_epoch_end(self, all_descriptors):
        return self.inference_epoch_end(all_descriptors, self.val_dataset, 'val')

    def test_epoch_end(self, all_descriptors):
        return self.inference_epoch_end(all_descriptors, self.test_dataset, 'test', self.num_preds_to_save)

    def inference_epoch_end(self, all_descriptors, inference_dataset, split, num_preds_to_save=0):
        """all_descriptors contains database then queries descriptors"""
        all_descriptors = np.concatenate(all_descriptors)
        queries_descriptors = all_descriptors[inference_dataset.database_num : ]
        database_descriptors = all_descriptors[ : inference_dataset.database_num]

        recalls, recalls_str = utils.compute_recalls(
            inference_dataset, queries_descriptors, database_descriptors,
            self.logger.log_dir, num_preds_to_save, self.save_only_wrong_preds
        )
        # print(recalls_str)
        logging.info(f"Epoch[{self.current_epoch:02d}]): " +
                      f"recalls: {recalls_str}")
    
        self.log(f'{split}/R@1', recalls[0], prog_bar=False, logger=True)
        self.log(f'{split}/R@5', recalls[1], prog_bar=False, logger=True)

def get_datasets_and_dataloaders(args):
    train_transform = tfm.Compose([
        tfm.RandAugment(num_ops=3),
        tfm.ToTensor(),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = TrainDataset(
        dataset_folder=args.train_path,
        img_per_place=args.img_per_place,
        min_img_per_place=args.min_img_per_place,
        transform=train_transform
    )
    val_dataset = TestDataset(dataset_folder=args.val_path)
    test_dataset = TestDataset(dataset_folder=args.test_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


if __name__ == '__main__':
    args = parser.parse_arguments()
    utils.setup_logging(join('logs', 'lightning_logs', args.exp_name), console='info')
    

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_datasets_and_dataloaders(args)
    
    model = LightningModel(val_dataset, test_dataset, args.descriptors_dim, args.num_preds_to_save, args.save_only_wrong_preds, 
                           alpha_param=args.alpha, beta_param=args.beta, base_param=args.base, eps_param=args.eps, opt_param=args.opt, 
                           loss_param=args.loss, pool_param=args.pool, miner_param=args.miner, lr_adam_param=args.lr_adam)
    
    # Model params saving using Pytorch Lightning. Save the best 3 models according to Recall@1
    checkpoint_cb = ModelCheckpoint(
        monitor='val/R@1',
        filename='_epoch({epoch:02d})_R@1[{val/R@1:.4f}]_R@5[{val/R@5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_top_k=1,
        save_last=True,
        mode='max'
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/", version=args.exp_name)

    # Instantiate a trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        default_root_dir='./logs',  # Tensorflow can be used to viz
        num_sanity_val_steps=0,  # runs a validation step before stating training
        precision=16,  # we use half precision to reduce  memory usage
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,  # run validation every epoch
        logger=tb_logger, # log through tensorboard
        callbacks=[checkpoint_cb],  # we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
        log_every_n_steps=20,
    )
    trainer.validate(model=model, dataloaders=val_loader, ckpt_path=args.checkpoint)
    trainer.fit(model=model, ckpt_path=args.checkpoint, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model=model, dataloaders=test_loader, ckpt_path='best')

