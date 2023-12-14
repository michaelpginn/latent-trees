import datasets
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from src.eval import eval_preds
from src.TreeTransformer import TreeBertForSequenceClassification
import random
import torch

# torch.manual_seed(10)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

dataset = datasets.load_dataset("michaelginn/latent-trees-agreement-ID")


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_length = 100
def tokenize_function(example):
    return tokenizer(example['text'], max_length=max_length, truncation=True)
dataset = dataset.map(tokenize_function, batched=True, load_from_cache_file=False)

toy_dataset = dataset['train'].select(range(2, 4))

id2label = {0: "VIOLATION", 1: "GRAMMATICAL"}
label2id = {"VIOLATION": 0, "GRAMMATICAL": 1}

pretrained = False
if pretrained:
    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2, id2label=id2label, label2id=label2id)
else:
    # Create random initialized BERT model
    config = BertConfig(num_labels=2, id2label=id2label, label2id=label2id)

model = TreeBertForSequenceClassification(config=config, disable_treeing=True).to('mps')
# model = BertForSequenceClassification(config=config).to('mps')

args = TrainingArguments(
    output_dir=f"../training-checkpoints",
    learning_rate=0.01,
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    # gradient_accumulation_steps=1,
    save_strategy="epoch",
    weight_decay=0,
    save_total_limit=3,
    num_train_epochs=1000,
    load_best_model_at_end=False,
    logging_strategy='epoch',
)

def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = np.argmax(eval_pred.predictions, axis=-1)
    print(eval_pred.predictions)
    return eval_preds(preds, labels)


trainer = Trainer(
    model,
    args,
    train_dataset= toy_dataset, # dataset['train'],
    eval_dataset= toy_dataset, # dataset['eval'], # dataset['test'].select(range(20)),
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

print("ayo")
trainer.train()


# preds = trainer.predict(dataset['test'].select(range(20)))
# preds