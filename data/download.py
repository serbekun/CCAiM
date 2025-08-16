from datasets import load_dataset
import os

current_dir = os.getcwd()

dataset = load_dataset("serbekun/CCAiM-CloudsDataset")

# Сохраняем датасет на диск
dataset.save_to_disk(current_dir)  # Основной метод сохранения

print(dataset)
print(f"dataset saved in: {current_dir}")
print("folder ls:", os.listdir(current_dir))