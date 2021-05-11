from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
import os

def inference(model, tokenized_sent, device):
  dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
  model.eval()
  output_pred = []
  
  for i, data in enumerate(dataloader):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          # token_type_ids=data['token_type_ids'].to(device)  # RoBert에서는 주석처리
          )
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
  
  return np.array(output_pred).flatten()

def inference_kfold(models, tokenized_sent, device, weights):
  dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
  pred_logits = None
  output_pred = []
  for i in range(len(models)):
    models[i].eval()

  with torch.no_grad():
    for i, data in enumerate(dataloader):
      for i in range(len(models)):
        outputs = models[i](
        input_ids=data['input_ids'].to(device),
        attention_mask=data['attention_mask'].to(device),
        # token_type_ids=data['token_type_ids'].to(device)  # RoBert에서는 주석처리
        )
        weight = weights[i]
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        if i == 0:
          pred_logits = (weight * logits[:])
        else:
          pred_logits += (weight * logits[:])
      result = np.argmax(pred_logits, axis=-1)
      output_pred.append(result)
  
  return np.array(output_pred).flatten()
  

def load_test_dataset(dataset_dir, tokenizer):
  test_dataset = load_data(dataset_dir)
  test_label = test_dataset['label'].values
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return tokenized_test, test_label

def main(args):
  """
    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  # TOK_NAME = "monologg/koelectra-base-v3-discriminator" 
  
  tokenizer = AutoTokenizer.from_pretrained(args.model_name)

  # load my model
  # MODEL_NAME = args.model_dir # model dir.
  models = []
  model_paths = ["./checkpoints/exp0/checkpoint-2280", "./checkpoints/exp1/checkpoint-2280", "./checkpoints/exp2/checkpoint-2280", "./checkpoints/exp3/checkpoint-1368", "./checkpoints/exp4/checkpoint-2052", "./checkpoints/12_xlm-roberta-large-tune-check2000/checkpoint-2000"]

  # validaion accuracy에 따른 weight
  weights = [1.0, 1.1, 1.2, 1.3, 1.4, 1.2]

  for model_path in model_paths: 
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    models.append(model)

  # load test datset
  test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
  test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  test_dataset = RE_Dataset(test_dataset ,test_label)

  # predict answer
  # pred_answer = inference(model, test_dataset, device)
  pred_answer = inference_kfold(models, test_dataset, device, weights)
  # make csv file with predicted answer
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.

  output = pd.DataFrame(pred_answer, columns=['pred'])
  if not os.path.exists("./prediction"):
    os.mkdir("./prediction")
  output.to_csv(args.out_path, index=False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  # parser.add_argument('--model_dir', type=str, default="./checkpoints/exp0/checkpoint-2280")
  parser.add_argument('--out_path', type=str, default="./prediction/submission.csv")
  parser.add_argument('--model_name', type=str, default="xlm-roberta-large")
  args = parser.parse_args()
  print(args)
  main(args)
  
