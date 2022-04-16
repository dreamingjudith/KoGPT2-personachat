import argparse
import logging
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from interact import chat

from utils import get_dataloaders, get_kogpt2_tokenizer
from models import CMPersonaChat

logger = logging.getLogger(__file__)


def train(args, tokenizer):
    dataloader = get_dataloaders(args, tokenizer)

    tb_logger = TensorBoardLogger("logs", name=args.name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{tb_logger.log_dir}/checkpoints',
        filename='model_epoch-{epoch:02d}_avg_val_loss-{loss/avg_val_loss:.4f}',
        auto_insert_metric_name=False,
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
            max_epochs=3,
            accumulate_grad_batches=8,
            logger=tb_logger)

        # model = CMPersonaChat(args)  # 그냥 args로 전달하면 멍청한 코드가 model init 때 namespace를 파싱하지 못 하고 dict로 인식하고 자빠짐
        model = CMPersonaChat(**vars(args))  # 따라서 dict화 시켜서 던져주고 model_init 때도 kwargs 형태로 받아야 한다 https://github.com/PyTorchLightning/pytorch-lightning/issues/5944#issuecomment-778620122
    # Fine-tune from saved checkpoint
    else:
        trainer = Trainer(
            resume_from_checkpoint=args.ckpt_path,
            callbacks=[checkpoint_callback],
            gradient_clip_val=1.0,
            max_epochs=3,
            accumulate_grad_batches=8,
            logger=tb_logger)

        model = CMPersonaChat.load_from_checkpoint(args.ckpt_path)

    model.train()
    trainer.fit(model, dataloader['train'], dataloader['valid'])
    logging.info('Best model path: {}'.format(checkpoint_callback.best_model_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str,
                        default="dataset/Ko_persona_merged.json",
                        help="Path of url of the dataset (JSON)")
    parser.add_argument("--dataset_cache", type=str,
                        default='./dataset_cache',
                        help="Path of the dataset cache")
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

    # Special arguments for inference
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")

    # Select train/inference
    parser.add_argument('--mode', type=str, choices=['train', 'chat'],
                        required=True,
                        help='Script mode to execute (train, eval, chat)')

    # Model configuration augments
    parser = CMPersonaChat.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Load dataset
    tokenizer = get_kogpt2_tokenizer()

    if args.mode == 'train':
        train(args, tokenizer)

    elif args.mode == 'chat':
        chat(args, tokenizer)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
