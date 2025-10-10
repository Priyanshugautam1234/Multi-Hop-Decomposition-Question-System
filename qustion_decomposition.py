import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import evaluate
import numpy as np
import torch




file_path = "/content/data_02.tsv"
df = pd.read_csv(file_path, sep="\t")


if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])


train_q, val_q, train_d, val_d = train_test_split(
    df["question"], df["decompose"], test_size=0.1, random_state=42
)

train_df = pd.DataFrame({"question": train_q, "decompose": train_d})
val_df = pd.DataFrame({"question": val_q, "decompose": val_d})

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(val_df)
})




model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f" Device set to {device}")




def preprocess_function(examples):
    inputs = ["decompose: " + q for q in examples["question"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    labels = tokenizer(examples["decompose"], max_length=256, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)




training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=500,
    save_steps=1000,
    eval_steps=1000,
    save_total_limit=2,
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)




rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return {k: round(v * 100, 2) for k, v in result.items()}




trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)




trainer.train()




model.save_pretrained("./t5-decomposition")
tokenizer.save_pretrained("./t5-decomposition")




def decompose_question(q):
    input_ids = tokenizer("decompose: " + q, return_tensors="pt").input_ids.to(device)
    output = model.generate(input_ids, max_length=256, num_beams=4, early_stopping=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)




print("\n Multi-hop Question Decomposition (type 'exit' to stop)")
while True:
    q = input("\nEnter a question: ")
    if q.lower() == "exit":
        print(" Exiting...")
        break
    print("Decomposed:", decompose_question(q))
