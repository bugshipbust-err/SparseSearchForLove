from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch


class WikiBatchTokenizer:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def sort_text(self, ret_len=False):
        sorted_lst = []
        for sample in self.dataset:
            if len(sample['text'].split(" ")) > 45:
                sorted_lst.append(sample["text"])
        sorted_lst.sort(key=len)
        self.sorted_lst = sorted_lst
        
        if ret_len:
            return len(sorted_lst)

    def gen_batch(self, iter):
        batch = self.sorted_lst[self.batch_size*iter : self.batch_size*(iter+1)]
        tokenized_batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True)

        min_length = 1000000
        for seq in tokenized_batch["input_ids"]:
            real_len = 0
            for token in seq:
                if token == 50256:
                    break
                real_len += 1
            if real_len < min_length:
                min_length = real_len

        tokenized_batch['input_ids'] = tokenized_batch['input_ids'][:, :min_length]
        tokenized_batch['attention_mask'] = tokenized_batch['attention_mask'][:, :min_length]

        return tokenized_batch



def test():
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

    main_set = dataset["train"]
    valid_set = dataset["test"]

    bt = WikiBatchTokenizer(valid_set, batch_size=3)
    print(bt.sort_text())

    for i in range(1000):
        try:
            print(bt.gen_batch(i)["input_ids"].size())
        except Exception:
            break

    print(i)
