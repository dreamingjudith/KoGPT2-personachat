# KoGPT2-personachat

Fine-tuned KoGPT2 chatbot demo with translated PersonaChat (ongoing) 

## Install

개발환경의 재현을 위해 [Anaconda](https://www.anaconda.com/products/individual) 환경 사용을 권장합니다.

```bash
$ git clone --recurse-submodules https://github.com/dreamingjudith/KoGPT2-personachat.git
$ cd KoGPT2-personachat
$ conda env create -f environment.yml
```

그러나 만약 `virtualenv` 같은 다른 가상환경을 사용할 경우 아래의 모듈을 설치했을 때 정상동작을 보장합니다. (괄호 안의 숫자는 개발 당시 사용한 버전입니다.)

- pytorch* (1.10.2)
- pytorch-lightning (1.5.10)
- tensorboard (2.8.0)
- tokenizers (0.10.3)
- transformers (4.3.3)

\* `cudatoolkit=={$CUDA_버전}`과 함께 설치하면 GPU 버전의 PyTorch를 설치합니다. 자세한 내용은 [링크](https://pytorch.org/get-started/locally/)를 참고하세요.

## Usage

### Train

학습 시 `--dataset_path`로 지정된 JSON 파일의 이름에 따라 미리 토크나이즈된 dataset_cache를 불러올 수도 있습니다. 따라서 정확한 파일 패스 지정이 필요합니다. 혹은 `--dataset_cache` 를 통해 캐시 파일의 위치를 직접 지정할 수도 있습니다.

```bash
$ conda activate cm

# Using dataset_path
$ python main.py --mode train --dataset_path dataset/sample.json --gpus 1

# Using dataset_cache
$ python main.py --mode train --dataset_cache dataset_cache_sample --gpus 1

# You can restore model from checkpoint
$ python main.py --mode train --dataset_path dataset/sample.json --gpus 1 --ckpt_path ${MODEL_CHECKPOINT_PATH}
```

더 많은 종류의 하이퍼파라미터 옵션을 확인하고 싶을 땐 아래와 같이 입력하세요.

```
$ python main.py --help
```

### :warning: Default hyperparameters used in PyTorch-Lightning Trainer

| flag name               | value |
| ----------------------- | ----- |
| max_epochs              | 3     |
| accumulate_grad_batches | 8     |
| gradient_clip_val       | 1.0   |

만약 위에 명시된 것과 다른 값을 사용하고 싶다면 명령 실행 시 `--max_epochs 10` 과 같이 사용하면 됩니다.

### Interactive chatting with pretrained checkpoint

```bash
$ conda activate cm
$ python main.py --mode chat --dataset_path dataset/sample.json --ckpt_path ${MODEL_CHECKPOINT_PATH}
```

## Reference
- [Transfer Learning for ConvAI2 by HuggingFace](https://github.com/huggingface/transfer-learning-conv-ai)
- [KoGPT2 by SK Telecom](https://github.com/SKT-AI/KoGPT2)
- [KoGPT2-chatbot by haven-jeon](https://github.com/haven-jeon/KoGPT2-chatbot)

## Contributors
- [@dreamingjudith](https://github.com/dreamingjudith)
- [@ModestyJ](https://github.com/ModestyJ)
- [@kheonCH](https://github.com/kheonCh)
- [@prodigyduck](https://github.com/prodigyduck)
- [You!!](https://github.com/dreamingjudith/KoGPT2-personachat/pulls)

## License

Modified MIT License
