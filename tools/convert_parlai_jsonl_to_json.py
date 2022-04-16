"""
Converts jsonl files converted from ParlAI (by facebookresearch)
into huggingface/transfer-learning-conv-ai compatible json format.

Workflow:
[ParlAI] convai2/train_self_original.txt --> convai2_train_self_original.jsonl
[ParlAI] convai2/valid_self_original.txt --> convai2_valid_self_original.jsonl
[This script] convai2_{train/valid}_self_original.jsonl --> convai2_self_original.json

However, this script makes almost same json file which HuggingFace provides.
https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json
So you actually don't have to use this script to generate JSON file.
"""

import argparse
import json
import os


# Original code can be found in below URL
# https://gist.github.com/agalea91/a4f11a9259dae05d88d6c46b837a520f#file-jsonl_io-py
def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))


def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data


def parse_episode(input_episode):
    """
    Parse single episode into list of persona and utterances
    """
    persona = list()
    history = list()
    utterances = list()

    for item in input_episode['dialog']:
        # 데이터에 대놓고 '\n'이라 되어 있어 하드코딩함.
        sentences = item[0]['text'].split(sep='\n')
        for sentence in sentences:
            if sentence.startswith('your persona'):
                persona.append(sentence.replace("your persona: ", ""))
            else:
                history.append(sentence)

        temp_dict = dict()
        temp_dict['candidates'] = item[0]['label_candidates']
        temp_dict['history'] = history.copy()
        utterances.append(temp_dict)
        del(temp_dict)

        history.append(item[0]['eval_labels'][0])

    return persona, utterances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        default='/mnt/d/workspace/ParlAI',
                        help='Input directory path (should include both train, valid jsonl')
    parser.add_argument('--output_path', type=str,
                        default='converted.json',
                        help='Output file path')
    args = parser.parse_args()

    # Parse jsonl file and make output_dict
    output_dict = dict()
    for keyname in ['train', 'valid']:
        data_list = list()

        input_path = os.path.join(args.input_dir,
                                  f'convai2_{keyname}_self_original.jsonl')
        episode_list = load_jsonl(input_path)
        for episode in episode_list:
            persona, utterances = parse_episode(episode)

            temp_dict = dict()
            temp_dict['personality'] = persona
            temp_dict['utterances'] = utterances
            data_list.append(temp_dict)
            del(temp_dict)

        output_dict[keyname] = data_list

    # Save file
    output_path = args.output
    print(f"Saving {output_path}")
    with open(output_path, 'w') as fp:
        json.dump(output_dict, fp)
    print("DONE")


if __name__ == "__main__":
    main()
