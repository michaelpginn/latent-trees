import datasets
from transformers import BertTokenizer
from itertools import chain
import multiprocessing

# bookcorpus = datasets.load_dataset("bookcorpus", split="train")
# wiki = datasets.load_dataset("wikipedia", "20220301.en", split="train")
# wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])
# dataset = datasets.concatenate_datasets([bookcorpus, wiki])

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
num_proc = multiprocessing.cpu_count()

# def tokenize_function(example):
#     return tokenizer(example['text'], max_length=tokenizer.model_max_length, truncation=True)
# tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=num_proc, load_from_cache_file=False)

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= tokenizer.model_max_length:
        total_length = (total_length // tokenizer.model_max_length) * tokenizer.model_max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + tokenizer.model_max_length] for i in range(0, total_length, tokenizer.model_max_length)]
        for k, t in concatenated_examples.items()
    }
    return result


tokenized_datasets = datasets.load_dataset("michaelginn/bert_dataset_tokenized")
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
print(tokenized_datasets)
tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=num_proc)
# shuffle dataset
tokenized_datasets = tokenized_datasets.shuffle(seed=34)
print(f"the dataset contains in total {len(tokenized_datasets) * tokenizer.model_max_length} tokens")

tokenized_datasets.push_to_hub(f"michaelginn/bert_data_tokenized_grouped")