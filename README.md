# KoGPT2-personachat

Fine-tuned KoGPT2 chatbot demo with translated PersonaChat (ongoing) 

## Install

개발환경의 재현을 위해 [Anaconda](https://www.anaconda.com/products/individual) 환경 사용을 권장합니다.

```
$ git clone --recurse-submodules https://github.com/dreamingjudith/KoGPT2-personachat.git
$ cd KoGPT2-personachat
$ conda env create -f environment.yml
```

그러나 만약 `virtualenv` 같은 다른 가상환경을 사용할 경우 아래의 모듈을 설치했을 때 정상동작을 보장합니다. (괄호 안의 숫자는 개발 당시 사용한 버전입니다.)

- gluonnlp (0.10.0)
- mxnet* (1.6.0)
- pytorch** (1.6.0)
- pytorch-ignite (0.4.1)
- pytorch-lightning (0.8.5)
- sentencepiece (0.1.92)
- tensorboardX (1.8)
- tensorflow (2.2.0)
- transformers (3.0.2)

\* 실제 설치한 모듈명은 mxnet-cu*101*입니다. CUDA 버전에 맞게 *101* 부분을 수정하여 설치하세요.<br />
\** `cudatoolkit=={$CUDA_버전}`과 함께 설치하면 GPU 버전의 PyTorch를 설치합니다. 자세한 내용은 [링크](https://pytorch.org/get-started/locally/)를 참고하세요.

## Usage

### Train

학습 시 dataset_path로 지정된 JSON 파일의 이름에 따라 미리 토크나이즈된 dataset_cache를 불러올 수도 있습니다. 따라서 정확한 파일 패스 지정이 필요합니다.

```
$ conda activate cm
$ CUDA_VISIBLE_DEVICES=0 python cm_kogpt2.py --train --max_epochs=3 --dataset_path dataset/sample.json
## OR you can restore model checkpoint
$ CUDA_VISIBLE_DEVICES=0 python cm_kogpt2.py --train --restore --max_epochs=3 --dataset_path dataset/sample.json --model_params logs/cm_kogpt2/version_0/checkpoints/model_last.ckpt
```

더 많은 종류의 하이퍼파라미터 세팅을 확인하고 싶을 땐 아래와 같이 입력하세요.

```
$ python cm_kogpt2.py --help
```

### Interactive chatting with pretrained checkpoint

```
$ conda activate cm
$ CUDA_VISIBLE_DEVICES=0 python cm_kogpt2.py --chat --dataset_path dataset/sample.json --model_params ${MODEL_CHECKPOINT_PATH}
```

## Reference
- [Transfer Learning for ConvAI2 by HuggingFace](https://github.com/huggingface/transfer-learning-conv-ai)
- [KoGPT2 by SK Telecom](https://github.com/SKT-AI/KoGPT2)
- [KoGPT2-chatbot by haven-jeon](https://github.com/haven-jeon/KoGPT2-chatbot)

## Contributors
- [@ModestyJ](https://github.com/ModestyJ)
- [@kheonCH](https://github.com/kheonCh)
- [@prodigyduck](https://github.com/prodigyduck)
- [You!!](https://github.com/dreamingjudith/KoGPT2-personachat/pulls)

## License

Modified MIT License

