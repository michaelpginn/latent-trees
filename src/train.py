import click
import torch
import wandb
import datasets
from tokenizer import WhitespaceTokenizer
from transformers import BertConfig, BertForSequenceClassification, TrainingArguments, Trainer
import evaluate
import random
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "mps"

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

@click.command()
def train(batch_size=16, train_epochs=100, seed=0):
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
