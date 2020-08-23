import argparse
import logging
import os
from collections import defaultdict
from itertools import chain
from pprint import pformat

import torch
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import (TensorboardLogger,
                                                        OutputHandler,
                                                        OptimizerParamsHandler)
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Accuracy, Loss, RunningAverage
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, CONFIG_NAME, WEIGHTS_NAME

from utils import (get_dataset, get_kogpt2_model, get_kogpt2_tokenizer,
                   make_logdir)


SPECIAL_TOKENS = ["<s>", "</s>", "<usr>", "<sys>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<s>',
                         'eos_token': '</s>',
                         'pad_token': '<pad>',
                         'additional_special_tokens': ['<usr>', '<sys>']}
MODEL_INPUTS = ["input_ids", "labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "labels", "token_type_ids"]

logger = logging.getLogger(__file__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def pad_dataset(dataset, padding=0):
    """
    Pad the dataset.
    This could be optimized by defining a Dataset class
    and padding at the batch level,
    but this is simpler.
    """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "labels" else -100] * (max_l - len(x)) for x in dataset[name]]

    return dataset


def build_input_from_segments(persona, history, reply, vocab,
                              labels=False, with_eos=True):
    """
    Build a sequence of input from 3 segments:
    persona, history and last reply.
    """
    bos, eos, speaker1, speaker2 = vocab[SPECIAL_TOKENS[:-1]]
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["labels"] = [-100] * len(instance["input_ids"])
    if labels:
        instance["labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]

    return instance


def get_data_loaders(args, tokenizer, vocab):
    """
    Prepare the dataset for training and evaluation
    """
    personachat = get_dataset(tokenizer, vocab, args.dataset_path)

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
                        instance = build_input_from_segments(persona, history, candidate, vocab, labels)
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)
                    datasets[dataset_name]["n_candidates"] = num_candidates
                # permuted personalities
                persona = [persona[-1]] + persona[:-1]

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=vocab[SPECIAL_TOKENS[-1]])
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_batch_size,
                              shuffle=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.valid_batch_size,
                              shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader


def train(args, tokenizer, model, train_loader, val_loader):
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)

    def update(engine, batch):
        model.train()

        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input_ids, labels, token_type_ids = batch

        loss, *_ = model(input_ids,
                         token_type_ids=token_type_ids,
                         labels=labels)
        loss = loss / args.gradient_accumulation_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        return loss.item()

    trainer = Engine(update)

    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device)
                          for input_tensor in batch)
            input_ids, labels, token_type_ids = batch
            # logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
            # if we dont send labels to model, it doesnt return losses
            logits, *_ = model(input_ids, token_type_ids=token_type_ids)
            logits_flat_shifted = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            labels_flat_shifted = labels[..., 1:].contiguous().view(-1)
            return (logits_flat_shifted), (labels_flat_shifted)

    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100),
                           output_transform=lambda x: (x[0], x[1])),
               "accuracy": Accuracy(output_transform=lambda x: (x[0], x[1]))}
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model,
    # configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        log_dir = make_logdir("kogpt2_personachat")
        tb_logger = TensorboardLogger(log_dir)

        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(
            metrics.keys()), global_step_transform=global_step_from_engine(trainer)), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', save_interval=1, n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation

        torch.save(args, log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
        # tokenizer.save_pretrained(log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        # TODO: PR in ignite to have better access to saved file paths (cleaner)
        os.rename(os.path.join(log_dir, checkpoint_handler._saved[-1][1]), os.path.join(log_dir, WEIGHTS_NAME))
        tb_logger.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path of the dataset.")
    parser.add_argument("--dataset_cache", type=str,
                        default='./dataset_cache',
                        help="Path or url of the dataset cache")
    parser.add_argument("--num_candidates", type=int, default=2,
                        help="Number of candidates for training")
    parser.add_argument("--personality_permutations", type=int, default=1,
                        help="Number of permutations of personality sentences")
    parser.add_argument("--max_history", type=int, default=2,
                        help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int,
                        default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int,
                        default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float,
                        default=6.25e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float,
                        default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()

    tokenizer, _, vocab = get_kogpt2_tokenizer()
    model = get_kogpt2_model(ctx=args.device)

    train_loader, val_loader = get_data_loaders(args, tokenizer, vocab)
    train(args, tokenizer, model, train_loader, val_loader)

    print("DONE")


if __name__ == "__main__":
    main()
