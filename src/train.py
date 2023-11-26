import click
import torch
import wandb
import datasets
from WhitespaceTokenizer import WhitespaceTokenizer
from transformers import BertConfig, BertForSequenceClassification, TrainingArguments, Trainer, TrainerCallback, \
    BertTokenizer, EarlyStoppingCallback
import random
import numpy as np
from torch.utils.data import DataLoader

from eval import eval_preds

device = "cuda:0" if torch.cuda.is_available() else "mps"

def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = np.argmax(eval_pred.predictions, axis=-1)
    print(eval_pred.predictions)
    return eval_preds(preds, labels)


class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Print the logs or push them to your preferred logging framework
        print(logs)


@click.command()
@click.option('--dataset', required=True, type=click.Choice(['ID', 'GEN'], case_sensitive=False))
@click.option('--pretrained', is_flag=True)
@click.option('--train_epochs', type=int)
def train(dataset='ID', pretrained: bool = False,  batch_size=32, train_epochs=100, seed=1):
    random.seed(seed)
    run_name = f'transformer-pt{pretrained}-{dataset}'
    wandb.init(project='latent-trees-agreement', entity="michael-ginn", name=run_name, config={
        "random-seed": seed,
        "epochs": train_epochs,
        "dataset": dataset,
        "pretrained": pretrained
    })

    if dataset == 'ID':
        dataset = datasets.load_dataset("michaelginn/latent-trees-agreement-ID")
    else:
        dataset = datasets.load_dataset("michaelginn/latent-trees-agreement-GEN")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    max_length = 100
    def tokenize_function(example):
        return tokenizer(example['text'], max_length=max_length, truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, load_from_cache_file=False)

    id2label = {0: "VIOLATION", 1: "GRAMMATICAL"}
    label2id = {"VIOLATION": 0, "GRAMMATICAL": 1}

    if pretrained:
        config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2, id2label=id2label, label2id=label2id)
    else:
        # Create random initialized BERT model
        config = BertConfig(num_labels=2, id2label=id2label, label2id=label2id)

    model = BertForSequenceClassification(config=config).to(device)

    args = TrainingArguments(
        output_dir=f"../training-checkpoints",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=3,
        save_strategy="epoch",
        save_total_limit=3,
        num_train_epochs=train_epochs,
        load_best_model_at_end=False,
        logging_strategy='epoch',
        report_to='wandb'
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=compute_metrics,
        callbacks=[LogCallback, EarlyStoppingCallback(early_stopping_patience=3)],
        tokenizer=tokenizer
    )

    trainer.train()

    trainer.save_model(f"../models/{run_name}")
    model.push_to_hub(f"michaelginn/latent-trees-{run_name}", commit_message="Add trained model")

if __name__ == "__main__":
    train()
