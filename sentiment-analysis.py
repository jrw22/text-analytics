from datasets import load_dataset

dataset = load_dataset('imdb')
train_data = dataset['train']['text']  # Get the text data from the training split
train_labels = dataset['train']['label']  # Get the labels from the training split

print(train_data[0])  # Print the first review in the training data
print(train_labels[0])  # Print the label of the first review in the training data
