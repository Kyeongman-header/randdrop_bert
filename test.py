from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

sent="hi.[SEP] My name is park. [SEP] Is he my friend? No! Acutually; he is my father..."

ids=tokenizer(sent,return_tensors="tf",padding="max_length",max_length=100, truncation=True).input_ids
print(ids)

reverse=tokenizer.decode([101, 3000, 1012, 102, 444])

print(reverse)