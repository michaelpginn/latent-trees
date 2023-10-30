import datasets
from src.WhitespaceTokenizer import WhitespaceTokenizer
from transformers import BertConfig, BertForSequenceClassification


dataset = datasets.load_dataset("michaelginn/latent-trees-agreement")
tokenizer = WhitespaceTokenizer()
tokenizer.learn_vocab([row['text'] for row in dataset['train']])
dataset = dataset.map(tokenizer.tokenize_batch, batched=True)


config = BertConfig(vocab_size=tokenizer.vocab_size, num_labels=2, max_position_embeddings=tokenizer.model_max_length)
model = BertForSequenceClassification(config=config).to('mps')

from transformers import TrainingArguments, Trainer

batch_size = 1
train_epochs = 10

toy_data = [dataset['train'][0], dataset['train'][26]]

args = TrainingArguments(
    output_dir=f"../training-checkpoints",
    evaluation_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=1,
    save_strategy="epoch",
    save_total_limit=3,
    num_train_epochs=train_epochs,
    load_best_model_at_end=False,
    logging_strategy='steps',
    logging_steps=1,
)

trainer = Trainer(
    model,
    args,
    train_dataset=toy_data,
    eval_dataset=toy_data,
)

trainer.train()