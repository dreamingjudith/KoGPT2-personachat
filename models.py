import argparse

import torch
from pytorch_lightning.core.lightning import LightningModule
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from utils import get_kogpt2_model


class CMPersonaChat(LightningModule):
    def __init__(self, **hparams):  # should get hparams with ** if you want pass args
    # def __init__(self, hparams):  # not like this
        super(CMPersonaChat, self).__init__()
        self.save_hyperparameters()
        self.kogpt2 = get_kogpt2_model()  # for inference. but why kogpt2 model isn't applied device option?

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        return parser

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices.
        https://github.com/PyTorchLightning/pytorch-lightning/issues/5449#issuecomment-757863689
        https://github.com/Zasder3/train-CLIP/issues/29#issuecomment-1056339940
        """
        dataset = self.trainer._data_connector._train_dataloader_source.dataloader()

        if self.trainer.max_steps:
            return self.trainer.max_steps

        dataset_size = (
            self.trainer.limit_train_batches
            if self.trainer.limit_train_batches != 0
            else len(dataset)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = dataset.batch_size * self.trainer.accumulate_grad_batches * num_devices
        return (dataset_size // effective_batch_size) * self.trainer.max_epochs

    def forward(self, inputs, token_type_ids):
        output, *_ = self.kogpt2(inputs, token_type_ids=token_type_ids)
        return output

    def training_step(self, batch, batch_idx):
        token_ids, label, mask = batch
        # forward: input(batch,max_sentence_length) -> output(batch_size, max_sentence_length,vocab)
        # e.g. (4,768) -> (4,768,50000)
        outputs = self.kogpt2(token_ids, token_type_ids=mask, labels=label)
        self.log("loss/train_loss", outputs.loss)

        return outputs.loss

    def validation_step(self, batch, batch_idx):
        # batch = tuple(input_tensor.to(self.hparams.device) for input_tensor in batch)
        token_ids, label, mask = batch
        outputs = self.kogpt2(token_ids, token_type_ids=mask, labels=label)
        self.log("loss/val_loss", outputs.loss)

        return outputs.loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log("loss/avg_val_loss", avg_loss)

    def configure_optimizers(self):
        # TODO: num_training_step을 구하기 위해 dataloder 없이 manual optimization을 이용해 warmup 하게 고치기
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)

        # Prepare learning rate scheduler
        num_train_steps = self.num_training_steps
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]
