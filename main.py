import argparse
import logging
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from utils import get_dataloaders, get_kogpt2_model, get_kogpt2_tokenizer

logger = logging.getLogger(__file__)


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


def train(dataloader, args):
    tb_logger = TensorBoardLogger("logs", name=args.name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{tb_logger.log_dir}/checkpoints',
        filename='model_{epoch:02d}-{loss/avg_val_loss:.4f}',
        auto_insert_metric_name=True,
        verbose=True,
        save_last=True,
        save_top_k=10,
        mode='min',
        monitor='loss/avg_val_loss'
    )

    # Fine-tune from pretrained KoGPT2
    if args.ckpt_path is None:
        trainer = Trainer.from_argparse_args(
            args,
            callbacks=[checkpoint_callback],
            gradient_clip_val=1.0,
            logger=tb_logger)

        # model = CMPersonaChat(args)  # 그냥 args로 전달하면 멍청한 코드가 model init 때 namespace를 파싱하지 못 하고 dict로 인식하고 자빠짐
        model = CMPersonaChat(**vars(args))  # 따라서 dict화 시켜서 던져주고 model_init 때도 kwargs 형태로 받아야 한다 https://github.com/PyTorchLightning/pytorch-lightning/issues/5944#issuecomment-778620122
    # Fine-tune from saved checkpoint
    else:
        trainer = Trainer(
            resume_from_checkpoint=args.ckpt_path,
            callbacks=[checkpoint_callback],
            gradient_clip_val=1.0,
            logger=tb_logger)

        model = CMPersonaChat.load_from_checkpoint(args.ckpt_path)

    model.train()
    trainer.fit(model, dataloader['train'], dataloader['valid'])
    logging.info('Best model path: {}'.format(checkpoint_callback.best_model_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str,
                        default="dataset/Ko_persona_merged.json",
                        help="Path of the dataset.")
    parser.add_argument("--dataset_cache", type=str,
                        default='./dataset_cache',
                        help="Path or url of the dataset cache")
    parser.add_argument("--num_candidates", type=int, default=1,
                        help="Number of candidates for training")
    parser.add_argument("--personality_permutations", type=int, default=1,
                        help="Number of permutations of personality sentences")
    parser.add_argument("--max_history", type=int, default=2,
                        help="Number of previous exchanges to keep in history")
    parser.add_argument("--name", type=str,
                        default="cm_kogpt2",
                        help="Model name for logging")
    parser.add_argument("--ckpt_path", type=str,
                        help="Checkpoint path for training or evaluation")

    # Shared arguments for dataloader and training
    parser.add_argument('--max_len',
                        type=int,
                        default=768,
                        help='max sentence length on input (default: 768)')
    parser.add_argument("--train_batch_size", type=int,
                        default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int,
                        default=4, help="Batch size for validation")
    parser.add_argument("--num_workers", type=int,
                        default=min(os.cpu_count(), 8), help="Number of workers for DataLoader")

    # Select train/inference
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'chat'],
                        required=True,
                        help='Script mode to execute (train, eval, chat)')

    # TODO: 로컬 테스트 중 tokenizer 로딩하다 죽는 현상 회피할 방법 찾기
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Model configuration augments
    parser = CMPersonaChat.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Load dataset
    tokenizer = get_kogpt2_tokenizer()
    dataloader = get_dataloaders(args, tokenizer)

    if args.mode == 'train':
        train(dataloader, args)

    elif args.mode == 'chat':
        raise ValueError("Not implemented yet!!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
