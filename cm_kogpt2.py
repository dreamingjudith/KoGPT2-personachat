import argparse
import logging
import os
import random
from collections import defaultdict
from itertools import chain

import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from utils import get_dataset, get_kogpt2_model, get_kogpt2_tokenizer


SPECIAL_TOKENS = ["<s>", "</s>", "<usr>", "<sys>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<usr>', '<sys>']}
MODEL_INPUTS = ["input_ids", "labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "labels", "token_type_ids"]

logger = logging.getLogger(__file__)


def pad_dataset(args, dataset, padding=0):
    """ Pad the dataset.
    This could be optimized by defining a Dataset class and padding at the batch level,
    but this is simpler. """
    # max_l = max(len(x) for x in dataset["input_ids"])
    max_l = args.max_len

    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "labels" else -100] * (max_l - len(x)) for x in dataset[name]]

    return dataset


def build_input_from_segments(persona, history, reply, vocab, labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = vocab[SPECIAL_TOKENS[:-1]]
    sequence = [[bos] + list(chain(*persona))] + \
        history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) %
                                 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i %
                                  2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["labels"] = [-100] * len(instance["input_ids"])
    if labels:
        instance["labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]

    return instance


def get_data_loaders(args, tokenizer, vocab):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset(tokenizer, vocab, args.dataset_path, args.dataset_cache)

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for dataset_name, dataset in personachat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(args.num_candidates, num_candidates)
        for dialog in dataset:
            persona = dialog["personality"].copy()
            for _ in range(args.personality_permutations):
                for utterance in dialog["utterances"]:
                    history = utterance["history"][-(2 * args.max_history + 1):]
                    for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                        labels = bool(j == num_candidates - 1)
                        instance = build_input_from_segments(
                            persona, history, candidate, vocab, labels)
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(
                                input_array)
                    datasets[dataset_name]["n_candidates"] = num_candidates
                # permuted personalities
                persona = [persona[-1]] + persona[:-1]

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(
            args, dataset, padding=vocab[SPECIAL_TOKENS[-1]])
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            tensor = tensor.view(
                (-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.valid_batch_size,
                              num_workers=args.num_workers,
                              shuffle=False)

    return train_loader, valid_loader


class CMPersonaChat(LightningModule):
    def __init__(self, hparams, *args):
        super(CMPersonaChat, self).__init__()
        self.hparams = hparams
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

    def forward(self, inputs, token_type_ids):
        output, *_ = self.kogpt2(inputs, token_type_ids=token_type_ids)
        return output

    def training_step(self, batch, batch_idx):
        batch = tuple(input_tensor.to(self.hparams.device) for input_tensor in batch)
        token_ids, label, mask = batch
        # forward: input(batch,max_sentence_length) -> output(batch_size, max_sentence_length,vocab)
        # e.g. (4,768) -> (4,768,50000)
        loss, *_ = self.kogpt2(token_ids, token_type_ids=mask, labels=label)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        batch = tuple(input_tensor.to(self.hparams.device) for input_tensor in batch)
        token_ids, label, mask = batch
        loss, *_ = self.kogpt2(token_ids, token_type_ids=mask, labels=label)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]


###################################
# Chat inference functions
def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history, vocab, model, args, current_output=None):
    special_tokens_ids = vocab[SPECIAL_TOKENS]
    if current_output is None:
        current_output = []

    for i in range(args.max_len):
        instance = build_input_from_segments(personality, history, current_output, vocab, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--dataset_path", type=str,
                        default="dataset/sample.json",
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

    # Shared arguments for dataloader and training
    parser.add_argument('--max_len',
                        type=int,
                        default=768,
                        help='max sentence length on input (default: 768)')
    parser.add_argument("--train_batch_size", type=int,
                        default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int,
                        default=2, help="Batch size for validation")
    parser.add_argument("--num_workers", type=int,
                        default=8, help="Number of workers for DataLoader")

    # Select train/inference
    parser.add_argument('--train',
                        action='store_true',
                        default=False,
                        help='eval train set (default: False)')
    parser.add_argument('--restore',
                        action='store_true',
                        default=False,
                        help='train using saved checkpoint (default: False)')
    parser.add_argument('--chat',
                        action='store_true',
                        default=False,
                        help='response generation on given user input')
    parser.add_argument('--model_params',
                        type=str,
                        default='cm_model_chp/model_last.ckpt',
                        help='model binary for starting chat')

    # Additional arguments for chatting
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")

    # Model configuration augments
    parser = CMPersonaChat.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Fine-tuning KoGPT2 for the PersonaChat
    if args.train:
        tokenizer, detokenizer, vocab = get_kogpt2_tokenizer()
        train_loader, val_loader = get_data_loaders(args, tokenizer, vocab)
        logger = TensorBoardLogger("logs", name=args.name)

        checkpoint_callback = ModelCheckpoint(
            filepath='{}/checkpoints/{}'.format(logger.log_dir, '{epoch:02d}-{val_loss:.4f}'),
            verbose=True,
            save_last=True,
            save_top_k=10,
            monitor='val_loss',
            mode='min',
            prefix='model_'
        )

        if args.restore:
            model = CMPersonaChat.load_from_checkpoint(args.model_params)
            model.to(args.device)
            model.train()
            trainer = Trainer(resume_from_checkpoint=args.model_params,
                              checkpoint_callback=checkpoint_callback,
                              gradient_clip_val=1.0,
                              logger=logger)
            trainer.fit(model, train_loader, val_loader)
        else:
            model = CMPersonaChat(args)
            model.to(args.device)
            model.train()
            trainer = Trainer.from_argparse_args(
                args,
                checkpoint_callback=checkpoint_callback,
                weights_save_path=os.getcwd(),
                gradient_clip_val=1.0,
                logger=logger)
            trainer.fit(model, train_loader, val_loader)
        logging.info('best model path {}'.format(checkpoint_callback.best_model_path))

    if args.chat:
        tokenizer, detokenizer, vocab = get_kogpt2_tokenizer()
        dataset = get_dataset(
            tokenizer, vocab, args.dataset_path, args.dataset_cache)
        model = CMPersonaChat.load_from_checkpoint(args.model_params)
        model.to(args.device)

        personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
        personality = random.choice(personalities)
        for sentence in personality:
            print("Selected personality: %s" % detokenizer(vocab.to_tokens(sentence)))
        history = []

        while True:
            raw_text = input(">>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input(">>> ")
            history.append(vocab[tokenizer(raw_text)])
            with torch.no_grad():
                out_ids = sample_sequence(personality, history, vocab, model, args)
            history.append(out_ids)
            history = history[-(2 * args.max_history + 1):]
            out_text = detokenizer(vocab.to_tokens(out_ids))
            print(out_text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
