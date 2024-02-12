import argparse
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def id_to_label(id):
    if id==0:
        return "文学と芸術"
    if id==1:
        return "科学と技術"
    if id==2:        
        return "社会科学"
    if id==3:
        return "健康と医学"
    if id==4:
        return "ビジネスと経済"
    if id==5:
        return "レジャーとライフスタイル"

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
    y_preds = np.argmax(outputs.logits.to('cpu').detach().numpy().copy(), axis=1)
    print('y_preds', y_preds)
    print('label', id_to_label(y_preds))

if __name__ == "__main__":
    main()