# 카카오로도 되는지 확인해본 스크립트. 사용가능 하나 자주 끊어져서 활용하진 않음

# -*- encoding: utf-8 -*-
import argparse
import json
import pdb
import time

import requests

def kor2eng(query):
    url = "https://translate.kakao.com/translator/translate.json"

    headers = {
        "Referer": "https://translate.kakao.com/",
        "User-Agent": "Mozilla/5.0"
    }

    data = {
        "queryLanguage": "en",
        "resultLanguage": "kr",
        "q": query
    }

    resp = requests.post(url, headers=headers, data=data)
    data = resp.json()
    output = data['result']['output'][0][0]
    return output

def translate_sentence_batch(input_batch):
    time.sleep(2)
    #translator = Translator()
    #translations = translator.translate(input_batch, src='en', dest='ko')
    url = "https://translate.kakao.com/translator/translate.json"

    headers = {
        "Referer": "https://translate.kakao.com/",
        "User-Agent": "Mozilla/5.0"
    }


    ret = list()
    for input in input_batch:
        data = {
           "queryLanguage": "en",
            "resultLanguage": "kr",
            "q": input
        }

        resp = requests.post(url, headers=headers, data=data)
        data = resp.json()  
        output = data['result']['output'][0][0]

        ret.append(output)

    return ret


def read_persona_chat(filepath):
    with open(filepath, 'r') as f:
        dataset = json.load(f)
    train = dataset['train']
    valid = dataset['valid']

    return train, valid

def write_persona_chat(filepath,dataset):
    with open(filepath, 'w') as f:
        json.dump(dataset, f, ensure_ascii=False)

def translate_persona_chat(dataset_mod):
    entryset_length = len(dataset_mod)
    dataset_translated = list()

    for entry in dataset_mod[:]:  # 여기 있는 인덱스로 앞에서 몇 개 번역해 넣을 건지 조절
        entry_dict = dict()
        personality = entry['personality']
        utterances = entry['utterances']
        utterances_length = len(utterances)
        utterances_translated = list()

        for utterance in utterances:
            utterance_dict = dict()
            candidates = utterance['candidates']
            history = utterance['history']

            utterance_dict['candidates'] = translate_sentence_batch(candidates)
            utterance_dict['history'] = translate_sentence_batch(history)

            utterances_translated.append(utterance_dict)

        assert len(utterances_translated) == utterances_length

        entry_dict['personality'] = translate_sentence_batch(personality)
        entry_dict['utterances'] = utterances_translated
        dataset_translated.append(entry_dict)

        global ii
        ii += 1
        #print (ii)

        filename = str(ii)
        write_persona_chat(filename +'_valid_kakao.json',entry_dict)

    #assert len(dataset_translated) == entryset_length

    return dataset_translated



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="personachat_self_original.json filepath")
    try:
        args = parser.parse_args()
        persona_train, persona_valid = read_persona_chat(args.file)
    except:
        args = ""
        #TEST_PATH = 'test.json'
        TEST_PATH = 'personachat_self_original.json'
        persona_train, persona_valid = read_persona_chat(TEST_PATH)
    
    valid_translated = translate_persona_chat(persona_valid)
    #write_persona_chat(TEST_PATH,train_translated)

    #pdb.set_trace()


if __name__ == "__main__":
    ii = 0
    main()
