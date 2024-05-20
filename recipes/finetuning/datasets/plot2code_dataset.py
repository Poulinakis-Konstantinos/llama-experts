import copy
import datasets
import itertools
import uuid


def tokenize_dialog(dialog, tokenizer):
    dialog_tokens = tokenizer.apply_chat_template(dialog, tokenize=True) 
    # Find end of turn (eot) tokens (assumed to be 128009)
    eot_indices = [i for i,n in enumerate(dialog_tokens) if n == 128009]
    labels = copy.copy(dialog_tokens)
    last_idx = 0
    for n, idx in enumerate(eot_indices):
        if n % 2 == 1:        # Odd indices correspond to assistant messages
            last_idx = idx
        else:                 # Even indices correspond to user messages
            labels[last_idx:idx+1] = [-100] * (idx-last_idx+1)
    dialog_tokens = [dialog_tokens]
    labels_tokens = [labels]
    
    combined_tokens = {
        "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
        "labels": list(itertools.chain(*(t for t in labels_tokens))),
    }
    # Shift labels to the left by one position
    combined_tokens["labels"] = combined_tokens["labels"][1:] + [-100]
    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))

def get_plot2code_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("TencentARC/Plot2Code", split='test')
    dataset = dataset.map(lambda sample:
                    {
                    'input_ids': sample['url'],
                    'instruction' : sample['instruction'],
                    'code' : sample['code'] },
                    batched=True,
                    remove_columns=list(dataset.features))
    
    def to_dialog(sample):
        dialog = [
            {
            'role': 'user',
            'content' : sample['instruction']},
            {
            'role': 'assistant',
            'content' : sample['code']
            }
            ]
        return {'dialog' : dialog}

    dataset = dataset.map(lambda x: to_dialog(x), remove_columns=list(dataset.features))
    print(dataset[0])
    dataset = dataset.map(lambda x: tokenize_dialog(x["dialog"], tokenizer),
                           remove_columns=list(dataset.features))

    return dataset

if  __name__=="__main__":
    from transformers import AutoTokenizer
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    token = ""
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
    data = get_plot2code(_, tok, 'test')
    print(data)
    print(data[0])