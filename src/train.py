import click
import torch
import wandb
import datasets
from WhitespaceTokenizer import WhitespaceTokenizer
from transformers import BertConfig, BertForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
import random
import numpy as np
from torch.utils.data import DataLoader

from eval import eval_preds

device = "cuda:0" if torch.cuda.is_available() else "mps"

def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = np.argmax(eval_pred.predictions, axis=-1)
    return eval_preds(preds, labels)


class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Print the logs or push them to your preferred logging framework
        print(logs)


@click.command()
def train(batch_size=32, train_epochs=100, seed=0):
    random.seed(seed)
    wandb.init(project='latent-trees-agreement', entity="michael-ginn", name='transformer-no-ft', config={
        "random-seed": seed,
        "epochs": train_epochs,
    })

    dataset = datasets.load_dataset("michaelginn/latent-trees-agreement-ID")
    tokenizer = WhitespaceTokenizer(max_length=50)
    tokenizer.learn_vocab([row['text'] for row in dataset['train']])
    dataset = dataset.map(tokenizer.tokenize_batch, batched=True, load_from_cache_file=False)

    # Create random initialized BERT model
    config = BertConfig(vocab_size=tokenizer.vocab_size, num_labels=2, max_position_embeddings=tokenizer.model_max_length)
    model = BertForSequenceClassification(config=config).to(device)

    args = TrainingArguments(
        output_dir=f"../training-checkpoints",
        evaluation_strategy="epoch",
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
        callbacks=[LogCallback],
    )

    trainer.train()

    trainer.save_model(f"../models/transformer_id")
    tokenizer.save_vocabulary("../models/transformer_id")

if __name__ == "__main__":
    train()
