from datasets import load_dataset

# Load the wikitext dataset
dataset = load_dataset('wikitext', 'wikitext-103-v1')

# Check the available splits (train, validation, test)
print("Available splits:", dataset.keys())

# Explore a few examples from the train split
print("\nExample from the train split:")
print(dataset['train'][0])

# Check the size of the dataset (number of samples)
print("\nNumber of samples in the train split:", len(dataset['train']))

# Explore a few examples from the validation split
print("\nExample from the validation split:")
print(dataset['validation'][0])

# Check the first few entries in the dataset
print("\nFirst few entries in the train split:")
for i in range(5):
    print(f"Entry {i}:")
    print(dataset['train'][i])
    print()

# If you want to inspect the text content specifically
sample_text = dataset['train'][0]['text']
print("\nText content of the first sample in the train split:")
print(sample_text)

# Check the distribution of lengths of texts (optional)
text_lengths = [len(sample['text']) for sample in dataset['train']]
print("\nAverage length of text in train split:", sum(text_lengths) / len(text_lengths))

