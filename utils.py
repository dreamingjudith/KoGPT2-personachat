import json
import logging
import os
from collections import defaultdict
from itertools import chain

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

logger = logging.getLogger(__file__)


SPECIAL_TOKENS = ["<s>", "</s>", "<usr>", "<sys>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<usr>', '<sys>']}
MODEL_INPUTS = ["input_ids", "labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "labels", "token_type_ids"]


def pad_dataset(dataset, padding=0):
    """ Pad the dataset.
    This could be optimized by defining a Dataset class and padding at the batch level,
    but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])

    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "labels" else -100] * (max_l - len(x)) for x in dataset[name]]

    return dataset


def build_input_from_segments(persona, history, reply, tokenizer, labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
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


def get_dataset(tokenizer, dataset_path, dataset_cache):
    """Read PersonaChat json file and return tokenized dataset"""

    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)

    else:
        dataset_basename = os.path.basename(dataset_path).split(".")[0]
        dataset_cache = "dataset_cache_{}".format(dataset_basename)

        logger.info("Reading {}".format(dataset_path))
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(dataset)
        torch.save(dataset, dataset_cache)

    return dataset


def get_dataloaders(args, tokenizer):
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
                    history = utterance["history"][-(2 * args.max_history + 1):]
                    for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                        labels = bool(j == num_candidates - 1)
                        instance = build_input_from_segments(
                            persona, history, candidate, tokenizer, labels)
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
            dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
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

    return {"train": train_loader, "valid": valid_loader}


def get_kogpt2_model():
    """Get KoGPT2 model after downloading"""

    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
    model.eval()

    return model


def get_kogpt2_tokenizer():
    """Get KoGPT2 Tokenizer after downloading"""

    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                        bos_token='</s>',
                                                        eos_token='</s>',
                                                        unk_token='<unk>',
                                                        pad_token='<pad>',
                                                        mask_token='<mask>')

    return tokenizer
