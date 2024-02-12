import argparse

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', default="tohoku-nlp/bert-base-japanese-v3")
    parser.add_argument('--model_path', default="tohoku-nlp/bert-base-japanese-v3")
    parser.add_argument('--prompt')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=6)

    text = tokenizer(args.prompt,
                    padding=True,
                    truncation=True,
                    return_tensors='pt')
    outputs = model(text.input_ids)
    print(outputs)

if __name__ == "__main__":
    main()