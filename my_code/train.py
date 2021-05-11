import pickle
import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
from glob import glob
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig

from load_data import *
# import wandb


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }

def train():

	seed_everything(args.seed)

	# for wandb
	# hyperparameter_defaults = dict(
	# 	# dropout = 0.1,
	# 	batch_size = args.batch_size,
	# 	lr = args.lr,
	# 	epoch = args.epoch,
	# 	# model_name = args.model_name,
	# 	# tokenizer_name = 'BertTokenizer',
	# 	# smoothing = 0.2
    # )

	# wandb.init(config=hyperparameter_defaults, project="p-stage-2")
	# config = wandb.config

    # load model and tokenizer
	# MODEL_NAME = "xlm-roberta-large"
	tokenizer = AutoTokenizer.from_pretrained(args.model_name)

	# load dataset
	train_dataset = load_data("/opt/ml/input/data/train/train2.tsv")
	train_label = train_dataset['label'].values

	# tokenizing dataset
	tokenized_train = tokenized_dataset(train_dataset, tokenizer)
	RE_train_dataset = RE_Dataset(tokenized_train, train_label)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	# setting model hyperparameter
	# config 자체에는 학습 weight 정보 없기 때문에, from_pretrained 사용해 weight 가져올 수 있다

	# bert_config = BertConfig.from_pretrained(MODEL_NAME)
	# bert_config.num_labels = 42
	# model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=bert_config)

	# Auto
	model_config = AutoConfig.from_pretrained(args.model_name)
	model_config.num_labels = 42
	model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=model_config)

	model.parameters
	model.to(device)

	# 사용한 option 외에도 다양한 option들이 있습니다.
	# https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.

	training_args = TrainingArguments(
		output_dir=args.output_dir,          # output directory
		save_total_limit=3,              # number of total save model.
		save_steps=args.save_steps,                 # model saving step.
		num_train_epochs=args.epoch,              # total number of training epochs
		learning_rate=args.lr,               # learning_rate
		per_device_train_batch_size=args.batch_size,  # batch size per device during training
		#per_device_eval_batch_size=16,   # batch size for evaluation
		warmup_steps=args.warmup_steps,                # number of warmup steps for learning rate scheduler
		weight_decay=args.weight_decay,               # strength of weight decay
		logging_dir=args.logging_dir,            # directory for storing logs
		logging_steps=args.logging_steps,              # log saving step.
		# dataloader_num_workers=4,
		label_smoothing_factor=0.5,
		# evaluation_strategy='steps', # evaluation strategy to adopt during training
		# 							# `no`: No evaluation during training.
		# 							# `steps`: Evaluate every `eval_steps`.
		# 							# `epoch`: Evaluate every end of epoch.
		# eval_steps = 500,            # evaluation step.
	)
	trainer = Trainer(
		model=model,                         # the instantiated 🤗 Transformers model to be trained
		args=training_args,                  # training arguments, defined above
		train_dataset=RE_train_dataset,         # training dataset
		#eval_dataset=RE_dev_dataset,             # evaluation dataset
		#compute_metrics=compute_metrics         # define metrics function
	)

	# train model
	trainer.train()

def main():
	train()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--seed", type=int, default=2021, help="random seed (Default: 2021)")
	parser.add_argument('--model_name', type=str, default="xlm-roberta-large")
	parser.add_argument("--epoch", type=int, default=10, help="number of epochs to train (default : 10)")
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--lr', type=float, default=1e-5)
	parser.add_argument('--weight_decay', type=float, default=0.01)
	parser.add_argument('--warmup_steps', type=int, default=300)               # number of warmup steps for learning rate scheduler
	parser.add_argument('--output_dir', type=str, default='./checkpoints/expr')
	parser.add_argument('--save_steps', type=int, default=500)
	parser.add_argument('--save_total_limit', type=int, default=3)
	parser.add_argument('--logging_steps', type=int, default=100)
	parser.add_argument('--logging_dir', type=str, default='./logs')            # directory for storing logs

	args = parser.parse_args()
	print(args)

	main()