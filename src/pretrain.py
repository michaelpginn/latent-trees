import click
import torch
import wandb
import datasets
from WhitespaceTokenizer import WhitespaceTokenizer
from transformers import BertConfig, BertForSequenceClassification, TrainingArguments, Trainer, TrainerCallback, \
    BertTokenizer, EarlyStoppingCallback, TrainerState, TrainerControl, DataCollatorForLanguageModeling
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
@click.option('--train_epochs', type=int)
def train(train_epochs=100):
    batch_size = 64
    random.seed(0)
    wandb.init(project='latent-trees-pretraining', entity="michael-ginn", config={
        "random-seed": 0,
        "epochs": train_epochs,
        "model": "TreeBERT"
    })

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = datasets.load_dataset("michaelginn/bert_dataset_tokenized_grouped")

    # Create random initialized BERT model
    config = BertConfig()

    model = TreeTransformer.TreeBertForMaskedLM(config=config).to(device)

    args = TrainingArguments(
        output_dir=f"../training-checkpoints",
        evaluation_strategy="epoch",
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=64,
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
        # eval_dataset=dataset['eval'],
        # compute_metrics=compute_metrics,
        callbacks=[
            LogCallback,
            # DelayedEarlyStoppingCallback(early_stopping_patience=3)
        ],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    )

    trainer.train()

    trainer.save_model(f"../models/pretrained-treebert")

    # model.push_to_hub(f"michaelginn/latent-trees-{run_name}", commit_message="Add trained model")

if __name__ == "__main__":
    train()
