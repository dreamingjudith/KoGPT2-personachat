SPECIAL_TOKENS = ["<s>", "</s>", "<usr>", "<sys>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<usr>', '<sys>']}
MODEL_INPUTS = ["input_ids", "labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "labels", "token_type_ids"]
