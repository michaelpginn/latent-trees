import click
import torch
import wandb
import datasets
from WhitespaceTokenizer import WhitespaceTokenizer
from transformers import BertConfig, BertForSequenceClassification, TrainingArguments, Trainer, TrainerCallback, \
    BertTokenizer, EarlyStoppingCallback, TrainerState, TrainerControl
import random
import numpy as np
from torch.utils.data import DataLoader
import wandb
import TreeTransformer

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


class DelayedEarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self, *args, start_epoch=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Only start applying early stopping logic after start_epoch
        if state.epoch >= self.start_epoch:
            super().on_evaluate(args, state, control, **kwargs)
        else:
            # Reset the patience if we're before the start_epoch
            self.patience = 0


@click.command()
@click.option('--dataset', required=True, type=click.Choice(['ID', 'GEN', 'GENX'], case_sensitive=False))
@click.option('--pretrained_model', type=str)
@click.option('--train_epochs', type=int)
@click.option('--use_tree_bert', is_flag=True)
@click.option('--dataset_size', type=int)
@click.option('--seed', type=int)
def train(dataset='ID', pretrained_model: str = None,  batch_size=64, train_epochs=100,
          use_tree_bert: bool = False, dataset_size=None, seed=1):
    random.seed(seed)
    model_name = "BERT" if not use_tree_bert else "TreeBERT"
    run_name = f'{model_name}-pt{"False" if pretrained_model is None else "True"}-{dataset}-{seed}'
    wandb.init(project='latent-trees-agreement', entity="michael-ginn", name=run_name, config={
        "random-seed": seed,
        "epochs": train_epochs,
        "dataset": dataset,
        "pretrained": pretrained_model,
        "model": model_name,
        "train_size": "full" if dataset_size is None else dataset_size
    })

    if dataset == 'ID':
        dataset = datasets.load_dataset("michaelginn/latent-trees-agreement-ID")
    elif dataset == 'GEN':
        dataset = datasets.load_dataset("michaelginn/latent-trees-agreement-GEN")
    else:
        dataset = datasets.load_dataset("michaelginn/latent-trees-agreement-GENX", download_mode='force_redownload')

    if dataset_size is not None:
        dataset['train'] = dataset['train'].shuffle(seed=seed).select(range(dataset_size))

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    max_length = 100 if dataset != 'GENX' else 260

    def tokenize_function(example):
        return tokenizer(example['text'], max_length=max_length, truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, load_from_cache_file=False)

    id2label = {0: "VIOLATION", 1: "GRAMMATICAL"}
    label2id = {"VIOLATION": 0, "GRAMMATICAL": 1}

    if pretrained_model is not None:
        config = BertConfig.from_pretrained(pretrained_model, num_labels=2, id2label=id2label, label2id=label2id, force_download=True)
    else:
        # Create random initialized BERT model
        config = BertConfig(num_labels=2, id2label=id2label, label2id=label2id)

    # if not use_tree_bert:
    #     model = BertForSequenceClassification(config=config).to(device)
    # else:
    #     model = TreeTransformer.TreeBertForSequenceClassification(config=config).to(device)
    model = TreeTransformer.TreeBertForSequenceClassification(config=config, disable_treeing=not use_tree_bert).to(device)

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
        load_best_model_at_end=True,
        logging_strategy='epoch',
        report_to='wandb'
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['eval'],
        compute_metrics=compute_metrics,
        callbacks=[
            LogCallback,
            DelayedEarlyStoppingCallback(early_stopping_patience=3)
        ],
        tokenizer=tokenizer
    )

    trainer.train()

    trainer.save_model(f"../models/{run_name}")

    test_eval = trainer.evaluate(dataset['test'])
    test_eval = {k.replace('eval', 'test'): test_eval[k] for k in test_eval}
    wandb.log(test_eval)

    # model.push_to_hub(f"michaelginn/latent-trees-{run_name}", commit_message="Add trained model")

if __name__ == "__main__":
    train()
