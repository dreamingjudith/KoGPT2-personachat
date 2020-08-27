
# -*- encoding: utf-8 -*-
import argparse
import json
import pdb
import time

from googletrans import Translator



def translate_sentence_batch(input_batch):
    time.sleep(2)
    translator = Translator()
    translations = translator.translate(input_batch, src='en', dest='ko')

    ret = list()
    for translation in translations:
        ret.append(translation.text)

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
        write_persona_chat(filename +'.json',entry_dict)

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
        #TEST_PATH = 's1.json'
        TEST_PATH = 'personachat_self_original.json'
        persona_train, persona_valid = read_persona_chat(TEST_PATH)
    
    train_translated = translate_persona_chat(persona_train)
    #write_persona_chat(TEST_PATH,train_translated)

    #pdb.set_trace()


if __name__ == "__main__":
    ii = 0
    main()