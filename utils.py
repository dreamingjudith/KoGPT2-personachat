import hashlib
import json
import logging
import os
import requests
import socket
import sys
from datetime import datetime

import gluonnlp as nlp
import torch
from gluonnlp.data import SentencepieceTokenizer, SentencepieceDetokenizer
from transformers import GPT2Config, GPT2LMHeadModel

from tools.example_entry import EXAMPLE_ENTRY


logger = logging.getLogger(__file__)


def _download(url, filename, chksum, cachedir='~/kogpt2/'):
    f_cachedir = os.path.expanduser(cachedir)
    os.makedirs(f_cachedir, exist_ok=True)
    file_path = os.path.join(f_cachedir, filename)
    if os.path.isfile(file_path):
        if hashlib.md5(open(file_path,
                            'rb').read()).hexdigest()[:10] == chksum:
            print('using cached model')
            return file_path
    with open(file_path, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(
                    chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write('\r[{}{}]'.format('█' * done,
                                                   '.' * (50 - done)))
                sys.stdout.flush()
    sys.stdout.write('\n')
    assert chksum == hashlib.md5(open(
        file_path, 'rb').read()).hexdigest()[:10], 'corrupted file!'
    return file_path


def get_kogpt2_model(ctx='cpu', cachedir='~/kogpt2/'):
    """Get KoGPT2 model after downloading"""

    model_info = {
        'url':
        'https://kobert.blob.core.windows.net/models/kogpt2/pytorch/pytorch_kogpt2_676e9bcfa7.params',
        'fname': 'pytorch_kogpt2_676e9bcfa7.params',
        'chksum': '676e9bcfa7'
    }

    kogpt2_config = {
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "n_ctx": 1024,
        "n_embd": 768,
        "n_head": 12,
        "n_layer": 12,
        "n_positions": 1024,
        "vocab_size": 50000,
        "activation_function": "gelu"
    }

    model_path = _download(model_info['url'],
                           model_info['fname'],
                           model_info['chksum'],
                           cachedir=cachedir)

    model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path=None,
                                            config=GPT2Config.from_dict(kogpt2_config),
                                            state_dict=torch.load(model_path))

    model.to(ctx)
    model.eval()

    return model


def get_kogpt2_tokenizer(cachedir='~/kogpt2/'):
    """Get KoGPT2 Tokenizer after downloading"""

    vocab_info = {
        'url':
        'https://kobert.blob.core.windows.net/models/kogpt2/tokenizer/kogpt2_news_wiki_ko_cased_818bfa919d.spiece',
        'fname': 'kogpt2_news_wiki_ko_cased_818bfa919d.spiece',
        'chksum': '818bfa919d'
    }

    vocab_path = _download(vocab_info['url'],
                           vocab_info['fname'],
                           vocab_info['chksum'],
                           cachedir=cachedir)

    tokenizer = SentencepieceTokenizer(vocab_path)
    detokenizer = SentencepieceDetokenizer(vocab_path)
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
                                                   mask_token=None,
                                                   sep_token=None,
                                                   cls_token=None,
                                                   unknown_token='<unk>',
                                                   padding_token='<pad>',
                                                   bos_token='<s>',
                                                   eos_token='</s>')

    return tokenizer, detokenizer, vocab


def get_dataset(tokenizer, vocab, dataset_path, dataset_cache):
    """Read PersonaChat json file and return tokenized dataset"""
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__

    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)

    else:
        logger.info("Reading {}".format(dataset_path))
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")

        def tokenize(obj):
            if isinstance(obj, str):
                return vocab[tokenizer(obj)]
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(dataset)
        torch.save(dataset, dataset_cache)

    return dataset


def make_logdir(model_name: str):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
        'runs', current_time + '_' + socket.gethostname() + '_' + model_name)
    return logdir
