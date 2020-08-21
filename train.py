import argparse
import logging
from collections import defaultdict
from itertools import chain

import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import get_dataset, get_kogpt2_model, get_kogpt2_tokenizer


SPECIAL_TOKENS = ["<s>", "</s>", "<usr>", "<sys>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<usr>', '<sys>']}

logger = logging.getLogger(__file__)


def pad_dataset(dataset, padding=0):
    """ Pad the dataset.
    This could be optimized by defining a Dataset class and padding at the batch level,
    but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -100]
                         * (max_l - len(x)) for x in dataset[name]]
    return dataset


def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(
        SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))] + \
        history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) %
                                 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i %
                                  2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-100] * sum(len(s)
                                              for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    return instance


def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)

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
                    history = utterance["history"][-(2*args.max_history+1):]
                    for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                        lm_labels = bool(j == num_candidates-1)
                        instance = build_input_from_segments(
                            persona, history, candidate, tokenizer, lm_labels)
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(
                                input_array)
                    datasets[dataset_name]["mc_labels"].append(
                        num_candidates - 1)
                    datasets[dataset_name]["n_candidates"] = num_candidates
                # permuted personalities
                persona = [persona[-1]] + persona[:-1]

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(
            dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                tensor = tensor.view(
                    (-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(
        *tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler,
                              batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler,
                              batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(
        train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(
        valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path of the dataset.")
    args = parser.parse_args()

    tokenizer, vocab = get_kogpt2_tokenizer()
    model = get_kogpt2_model(ctx=args.device)
    dataset = get_dataset(tokenizer, vocab, args.dataset_path)

    print("DONE")


if __name__ == "__main__":
    main()
