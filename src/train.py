import click
import torch
import wandb
import datasets
from tokenizer import WhitespaceTokenizer
from transformers import BertConfig, BertForSequenceClassification, TrainingArguments, Trainer
import random
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "mps"

def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = np.argmax(eval_pred.predictions, axis=-1)

    # Initialize variables to count true positives, false positives, etc.
    true_pos = np.zeros(np.max(labels) + 1)
    false_pos = np.zeros(np.max(labels) + 1)
    true_neg = np.zeros(np.max(labels) + 1)
    false_neg = np.zeros(np.max(labels) + 1)

    # Count labels and predictions
    for l, p in zip(labels, preds):
        if l == p:
            true_pos[l] += 1
        else:
            false_pos[p] += 1
            false_neg[l] += 1

    # Calculate metrics
    precision = np.sum(true_pos / (true_pos + false_pos + 1e-13)) / len(true_pos)
    recall = np.sum(true_pos / (true_pos + false_neg + 1e-13)) / len(true_pos)
    f1 = 2 * precision * recall / (precision + recall + 1e-13)
    accuracy = np.sum(true_pos) / len(labels)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
@click.command()
def train(batch_size=4, train_epochs=100, seed=0):
    random.seed(seed)
    wandb.init(project='latent-trees-agreement', entity="michael-ginn", name='transformer-no-ft', config={
        "random-seed": seed,
        "epochs": train_epochs,
    })

    dataset = datasets.load_dataset("michaelginn/latent-trees-agreement")
    tokenizer = WhitespaceTokenizer()
    tokenizer.learn_vocab([row['text'] for row in dataset['train']])
    dataset = dataset.map(tokenizer.tokenize_batch, batched=True)

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
    )

    trainer.train()


if __name__ == "__main__":
    train()
