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




import os

model_output_dir = "./t5-decomposition"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Device set to {device}")

if os.path.exists(model_output_dir):
    print(f"Loading saved model from {model_output_dir}...")
    tokenizer = T5Tokenizer.from_pretrained(model_output_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_output_dir)
    model.to(device)

else:
    print("Saved model not found. Starting training...")
    
    file_path = "data.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # The user's data appears to be tab-separated
    try:
        df = pd.read_csv(file_path, sep="\t")
    except:
        # Fallback to comma if tab fails
        df = pd.read_csv(file_path)


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
    model.to(device)

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
        report_to="none",
        max_steps=100 # LIMIT FOR DEMO/TESTING SPEED - Remove this for full training
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

    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)




def decompose_question(q):
    input_ids = tokenizer("decompose: " + q, return_tensors="pt").input_ids.to(device)
    output = model.generate(input_ids, max_length=256, num_beams=4, early_stopping=True, no_repeat_ngram_size=3)
    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    from src.pipeline import MultiHopPipeline

    print("\n🧠 Multi-Hop Question Decomposition & Adaptive Reasoning System")
    print("=" * 65)
    print("Commands:")
    print("  - Type any complex question to decompose and reason")
    print("  - Type 't5: <question>' to see raw T5 baseline output")
    print("  - Type 'exit' to quit\n")

    pipeline = MultiHopPipeline()

    while True:
        try:
            q = input("\nEnter a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not q:
            continue
        if q.lower() == "exit":
            print("Exiting...")
            break

        if q.lower().startswith("t5:"):
            raw_q = q[3:].strip()
            print("Raw T5 Output:", decompose_question(raw_q))
        else:
            print("\nExecuting Adaptive Retrieve-Reason-Retrieve Loop...")
            result = pipeline.query(q)
            print(f"\n[Status: {'Abstained' if result['abstained'] else 'Resolved'} | Latency: {result['total_latency_ms']} ms | Hops: {result['total_hops']}]")
            print("-" * 65)
            print("Reasoning Trace:")
            for trace in result["reasoning_trace"]:
                dep = f" (Depends on: {trace['depends_on']})" if trace["depends_on"] else ""
                bound = f" -> Bound Entity: {trace['resolved_entity']}" if trace["resolved_entity"] else ""
                print(f"  Hop {trace['hop_number']}{dep}: {trace['sub_query']}{bound}")
                for chunk in trace["retrieved_chunks"][:2]:
                    print(f"    - [{chunk['chunk_id']}] {chunk['title']} (Score: {chunk['final_score']})")
            print("-" * 65)
            print(result["final_answer"])

