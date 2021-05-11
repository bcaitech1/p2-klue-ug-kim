import pickle
import os
import random
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler


from load_data import *

print(torch.cuda.get_device_name(0))
print("Using CUDA: ", torch.cuda.is_available())


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
def compute_acc(outputs, labels):
    preds = outputs.argmax(-1)
    acc = accuracy_score(labels, preds)
    return acc

class LabelSmoothingLoss(nn.Module):
	def __init__(self, classes=42, smoothing=0.0, dim=-1):
		super(LabelSmoothingLoss, self).__init__()
		self.confidence = 1.0 - smoothing
		self.smoothing = smoothing
		self.cls = classes
		self.dim = dim
		
	def forward(self, pred, target):
		pred = F.log_softmax(pred, dim=self.dim)
		# pred = pred.log_softmax(dim=self.dim)
		with torch.no_grad():
			true_dist = torch.zeros_like(pred)
			true_dist.fill_(self.smoothing / (self.cls - 1))
			true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
		return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def train():

	seed_everything(args.seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# setting model hyperparameter
	# config 자체에는 학습 weight 정보 없기 때문에, from_pretrained 사용해 weight 가져올 수 있다

	# bert_config = BertConfig.from_pretrained(MODEL_NAME)
	# bert_config.num_labels = 42
	# model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=bert_config)

	# Auto
	model_config = XLMRobertaConfig.from_pretrained(args.model_name)
	model_config.num_labels = 42
	model = XLMRobertaForSequenceClassification(model_config)
	model = torch.nn.DataParallel(model)
	model.load_state_dict(torch.load("./checkpoints/expr/best.pth"))


    # load model and tokenizer
	# MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
	# roberta: https://huggingface.co/transformers/model_doc/xlmroberta.html
	tokenizer = AutoTokenizer.from_pretrained(args.model_name)

	# load dataset
	dataset = load_data("/opt/ml/input/data/train/train.tsv")
	# label = dataset['label'].values

	train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=args.seed)
	tokenized_train = tokenized_dataset(train_dataset, tokenizer)
	tokenized_val = tokenized_dataset(val_dataset, tokenizer)

	tokenized_train_label = train_dataset['label'].values
	tokenized_val_label = val_dataset['label'].values

	# train_datasets = TokenDataset(train_dataset, tokenizer)
	# val_datasets = TokenDataset(val_dataset, tokenizer)
	RE_train_dataset = RE_Dataset(tokenized_train, tokenized_train_label)
	RE_val_dataset = RE_Dataset(tokenized_val, tokenized_val_label)

	# print(model.parameters)
	model.to(device)
	

	train_loader = DataLoader(
		RE_train_dataset, 
		batch_size=args.batch_size,
		# num_workers=8,
		pin_memory=torch.cuda.is_available(),
		shuffle=True,
	)
	val_loader = DataLoader(
		RE_val_dataset,
		batch_size=args.batch_size, 
		# num_workers=8,
		shuffle=False,
		pin_memory=torch.cuda.is_available(),
	)

	optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	loss_fn = LabelSmoothingLoss(smoothing=0.5)
	# loss_fn = nn.CrossEntropyLoss()

	# t_total = len(train_loader) * args.epoch
	t_total = args.epoch
	warmup_step = int(t_total * args.warmup_steps)
	scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

	log_dir = ""
	log_list = glob("./logs/*")
	if len(log_list) == 0:
		log_dir = "./logs/exp1"
	else:
		log_list = [int(log[-1]) for log in log_list]
		log_dir = "./logs/exp" + str(max(log_list) + 1)
	
	logger = SummaryWriter(log_dir=log_dir)

	scaler = GradScaler()

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	import time

	for epoch in tqdm(range(args.epoch)):
		train_acc = 0.0
		train_loss = 0.0
		val_acc = 0.0
		val_loss = 0.0
		best_acc = 0.0
		model.train()
		for batch_id, batch in enumerate(tqdm(train_loader)):
			input_ids = batch["input_ids"].to(device)
			attention_mask = batch["attention_mask"].to(device)
			labels = batch["labels"].to(device)
			
			optimizer.zero_grad()
			with autocast():
				outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
				loss = loss_fn(outputs.logits, labels)

			# loss.backward()
			# optimizer.step()

			scaler.scale(loss).backward()
			
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

			scaler.step(optimizer)
			scaler.update()
			
			train_acc += compute_acc(outputs.logits.cpu(), labels.cpu())
			train_loss += loss

			if (batch_id + 1) % args.logging_steps == 0:
				train_loss = train_loss.data.cpu().numpy()
				print(f"[Train] epoch {epoch + 1} | batch_id {batch_id + 1} | loss {(train_loss) / args.logging_steps:.4f} | train_acc {train_acc / args.logging_steps:.4f}")
				logger.add_scalar("Train/loss", train_loss / args.logging_steps, epoch * len(train_loader) + batch_id)
				logger.add_scalar("Train/acc", train_acc / args.logging_steps, epoch * len(train_loader) + batch_id)
				train_acc = 0.0
				train_loss = 0.0

		# scheduler.step()
		
		print("\nStart Validation Step!")
		with torch.no_grad():
			model.eval()
			for batch_id, batch in enumerate(tqdm(val_loader)):
				input_ids = batch["input_ids"].to(device)
				attention_mask = batch["attention_mask"].to(device)
				labels = batch["labels"].to(device)
				outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
				loss = loss_fn(outputs.logits, labels)
				val_acc += compute_acc(outputs.logits.cpu(), labels.cpu())
				val_loss += loss

			print(f"[Val] epoch {epoch + 1} | val_acc {val_acc / (batch_id + 1):.4f}")
			logger.add_scalar("Val/loss", val_loss / (batch_id + 1), epoch)
			logger.add_scalar("Val/acc", val_acc / (batch_id + 1), epoch)


			if val_acc >= best_acc:
				best_acc = val_acc
				# torch.save(model.state_dict(), os.path.join(args.output_dir, "saved_" + str(epoch) + ".pt"))	
				torch.save(model.state_dict(), os.path.join(args.output_dir, "best_2.pt"))
				print("Saved best acc model...")
		
		scheduler.step()

	torch.save(model.state_dict(), os.path.join(args.output_dir, "last.pt"))


def main():
	train()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--seed", type=int, default=2021, help="random seed (Default: 2021)")
	parser.add_argument('--model_name', type=str, default="xlm-roberta-large")
	parser.add_argument("--epoch", type=int, default=10, help="number of epochs to train (default : 10)")
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--lr', type=float, default=5e-6)
	parser.add_argument('--weight_decay', type=float, default=0.01)
	parser.add_argument('--warmup_steps', type=int, default=500)               # number of warmup steps for learning rate scheduler
	parser.add_argument('--output_dir', type=str, default='./checkpoints/expr')
	parser.add_argument('--save_steps', type=int, default=500)
	parser.add_argument('--save_total_limit', type=int, default=3)
	parser.add_argument('--logging_steps', type=int, default=10)
	parser.add_argument('--logging_dir', type=str, default='./logs')            # directory for storing logs
	parser.add_argument('--max_grad_norm', type=int, default=1)

	args = parser.parse_args()
	print(args)

	train()