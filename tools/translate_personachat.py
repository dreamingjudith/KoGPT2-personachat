import argparse
import json
import os
from time import sleep

from googletrans import Translator
from tqdm import tqdm


def read_personachat(filepath):
    with open(filepath, 'r') as f:
        dataset = json.load(f)
    
    return dataset


def translate_sentence(translator, input_line):
    translated = translator.translate(input_line, src='en', dest='ko')
    return translated.text


def translate_batch(translator, input_batch):
    ret = list()

    for line in input_batch:
        # googletrans가 3.X부터 batch translation이 제대로 되지 않아
        # line별로 번역한 뒤 append하는 방식으로 변경함
        translated_text = translate_sentence(translator, line)
        ret.append(translated_text)

    if len(ret) != len(input_batch):
        raise ValueError("Number of translated sentences are not match with input batch")
    else:
        return ret


def translate_single_example(translator, example):
    """하나의 training/evaluation example 번역하기
    """

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
    translator = Translator()
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

        for example in tqdm(dataset[mode][save_num:], desc=f'Total {mode} examples'):
            for try_num in range(args.max_try):
                try:
                    translated_dict = translate_single_example(translator, example)
                except AttributeError:
                    print("Connection error occured. Sleeping 30 seconds...")
                    sleep(30)
                    print("Retrying")
                break

            if try_num == args.max_try:
                raise RuntimeError("Translation failed. Try later.")

            if translated_dict is None:
                raise ValueError("translated_dict is None.")

            count += 1
            filename = os.path.join(save_dir, f"{save_num}.json")
            with open(filename, "w") as f:
                json.dump(translated_dict, f, ensure_ascii=False)

            sleep(10)  # googletrans에서 API 차단을 막기 위해 10초 sleep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="personachat_self_original.json filepath")
    parser.add_argument("--output-dir", type=str, required=True, help="Save path of translated dataset")
    parser.add_argument("--max-try", type=int, default=10, help="How many times retry if translation failed")
    args = parser.parse_args()

    translate_personachat(args)


if __name__ == "__main__":
    main()
