import argparse
import json
import os
from time import sleep
from tqdm import tqdm


class BaseTranslator(object):
    def __init__(self):
        pass

    def translate(self):
        pass


class GoogleTranslator(BaseTranslator):
    """
    Translator class using googletrans module
    Translates single sentence
    """
    def __init__(self, src='en', dst='ko'):
        from googletrans import Translator
        self.src = src
        self.dst = dst
        self.translator = Translator()

    def translate(self, text):
        translated = self.translator.translate(text, src=self.src, dest=self.dst)
        return translated.text


class KakaoTranslator(BaseTranslator):
    """
    Translator class using kakaotrans module
    Translates single sentence
    But this module often fails because of API limit
    """
    def __init__(self, src='en', dst='kr'):
        from kakaotrans import Translator
        self.src = src
        self.dst = dst
        self.translator = Translator()

    def translate(self, text):
        return self.translator.translate(text, src=self.src, tgt=self.dst)


def read_personachat(filepath):
    with open(filepath, 'r') as f:
        dataset = json.load(f)

    return dataset


def translate_batch(translator, input_batch):
    ret = list()

    for line in input_batch:
        # googletrans가 3.X부터 batch translation이 제대로 되지 않아
        # line별로 번역한 뒤 append하는 방식으로 변경함
        translated_text = translator.translate(line)
        ret.append(translated_text)

    if len(ret) != len(input_batch):
        raise ValueError("Number of translated sentences are not match with input batch")
    else:
        return ret


def translate_single_example(translator_type, example):
    """하나의 training/evaluation example 번역하기
    """

    if translator_type == 'google':
        translator = GoogleTranslator()
    elif translator_type == 'kakao':
        translator = KakaoTranslator()
    else:
        raise ValueError(f"Unsupported translator type: {translator_type}")

    translated_dict = {
        "personality": None,
        "utterances": list()
    }

    # personality
    translated_dict['personality'] = translate_batch(translator, example['personality'])

    # utterances
    for utterance in example['utterances']:
        utterance_dict = dict()

        utterance_dict['candidates'] = translate_batch(translator, utterance['candidates'])
        utterance_dict['history'] = translate_batch(translator, utterance['history'])

        translated_dict['utterances'].append(utterance_dict)

    if len(example['utterances']) != len(translated_dict['utterances']):
        raise ValueError("Number of translated utterances are not match with input example")
    else:
        return translated_dict


def check_saved_file_number(save_dir):
    """파일 저장이 중단된 경우를 위해
    저장된 폴더 내에서 가장 큰 숫자 찾아내기
    """
    import re

    # save_num에 무조건 +1 할 거기 때문에 파일이 없으면 -1을 리턴하게 하기
    max_num = -1

    for filename in save_dir:
        try:
            number = int(re.findall("(\d+)", filename)[0])
        except IndexError:
            # 폴더 내에 파일이 없는 경우 -99를 지정
            number = -99
        if number > max_num:
            max_num = number

    return max_num


def translate_personachat(args):
    dataset = read_personachat(args.input_path)
    count = 0

    for mode in ['train', 'valid']:
        # 지정된 폴더 내에서 각각 train, valid 만들기 (분리 저장용)
        save_dir = os.path.join(args.output_dir, mode)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filelist = os.listdir(save_dir)
        if len(filelist) == len(dataset[mode]):
            print(f"Translation of {mode} is already done.")
            continue

        # 폴더 내에서 가장 큰 숫자 가져오기
        save_num = check_saved_file_number(filelist)

        # +1 해서 다음 순서 읽을 준비하기
        save_num += 1

        translated_dict = None

        for example in tqdm(dataset[mode][save_num:], desc=f'Total {mode} examples'):
            for try_num in range(1, args.max_try+1):
                try:
                    translated_dict = translate_single_example(args.type, example)
                    break
                except AttributeError:
                    if try_num == args.max_try:
                        raise
                    print(f"Connection error occured. Sleeping {args.wait_time} seconds...")
                    sleep(args.wait_time)
                    print(f"Retrying ({try_num})")

            if translated_dict is None:
                raise ValueError("translated_dict is None.")

            count += 1
            filename = os.path.join(save_dir, f"{save_num}.json")
            with open(filename, "w") as f:
                json.dump(translated_dict, f, ensure_ascii=False)

            save_num += 1
            sleep(5)  # googletrans에서 API 차단을 막기 위해 5초 sleep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="personachat_self_original.json filepath")
    parser.add_argument("--output-dir", type=str, required=True, help="Save path of translated dataset")
    parser.add_argument("--max-try", type=int, default=10, help="How many times to retry if translation failed")
    parser.add_argument("--wait-time", type=int, default=30, help="Seconds to wait if translation is failed")
    parser.add_argument("--type", type=str,
                        choices=['google', 'kakao'],
                        default='google',
                        help='Translator service provider')
    args = parser.parse_args()

    translate_personachat(args)


if __name__ == "__main__":
    main()
