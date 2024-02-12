from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def prepare_dataset(tokenizer, ds_path):
    ds = load_dataset("csv", data_files=ds_path)
    
    def tokenize(sample):
        return tokenizer(sample["text"], padding=True, truncation=True)    

    ds = ds.map(tokenize, batched=True, batch_size=None)    
    ds = ds.shuffle(seed=42)
    ds = ds['train'].train_test_split(test_size=0.1)
    return ds


def main():
    model_name = "tohoku-nlp/bert-base-japanese-v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=20,
        weight_decay=0.01,
        save_total_limit=1,
        dataloader_pin_memory=False, 
        # evaluation_strategy="epoch",
        evaluation_strategy="steps",
        logging_steps=10,
        # logging_dir='./logs'
    )

    ds = prepare_dataset(tokenizer=tokenizer, ds_path="./sample.csv")
    
    trainer = trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == '__main__':
    main()