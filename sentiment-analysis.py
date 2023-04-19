from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


dataset = load_dataset('imdb')
train_data = dataset['train']['text']  # Get the text data from the training split
train_labels = dataset['train']['label']  # Get the labels from the training split

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the text data
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True)

dataset = dataset.map(tokenize, batched=True, batch_size=len(dataset))

# Load the pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',  # Output directory
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=64,  # Batch size for evaluation
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # Strength of weight decay
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=10,  # Number of steps for logging
)

# Define the trainer
trainer = Trainer(
    model=model,  # The pre-trained BERT model
    args=training_args,  # Training arguments
    train_dataset=dataset['train'],  # Training dataset
    eval_dataset=dataset['test'],  # Evaluation dataset
)

# Fine-tune the model
trainer.train()
print('Training complete')


