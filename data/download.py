from datasets import load_dataset

# Downloads and caches the dataset locally (~/.cache/huggingface).
# No need to save_to_disk: train.py loads it straight from the cache.
dataset = load_dataset("serbekun/CCAiM-CloudsDataset")

print(dataset)
print(f"classes: {dataset['train'].features['label'].names}")
