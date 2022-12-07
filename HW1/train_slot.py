import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange
from model import SlotTagging
from dataset import SeqClsDataset, SeqTaggingClsDataset
from utils import Vocab
import sys
import logging
import os
import numpy as np
import csv

TRAIN = "train"
DEV = "eval"
TEST = "test"
SPLITS = [TRAIN, DEV, TEST]


def main(args):
    logger = get_loggings(args.ckpt_dir)
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "tag2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_loader = DataLoader(dataset=datasets['train'], batch_size=args.batch_size, 
                                shuffle=True, collate_fn=datasets['train'].collate_fn )
                                
    eval_loader = DataLoader(dataset=datasets['eval'], batch_size=args.batch_size, 
                                shuffle=False, collate_fn=datasets['eval'].collate_fn )

    test_loader = DataLoader(dataset=datasets['test'], batch_size=1, 
                                shuffle=False, collate_fn=datasets['test'].collatePred_fn )                       

    logger.info(f"train, valid dataset len: { len(datasets['train']) }, { len(datasets['eval']) }")
    num_class = len(datasets['train'].label_mapping)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SlotTagging(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, num_class)
    model.to(args.device)
    logger.info(model)

    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, min_lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_eval_loss = np.inf


    for epoch in epoch_pbar:
        model.train()
        loss_train, accToken_train, accJoint_train, iter_train, n_token = 0, 0, 0, 0, 0
        for batch, target in train_loader:
            batch, target = batch.to(args.device), target.to(args.device)
            output = model( batch ) # output [batch, seq_num, features]

            
            
            

            # calculate loss and update parameters
            loss = criterion(output, target)          
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate loss, accuracy
            iter_train += 1
            loss_train += loss.item()
            pred = output.max(1, keepdim=True)[1]
            # print(pred.size())---> torch.Size([16, 1, 50])
            # print(target.size())---> torch.Size([16, 50])

            # Joint acc
            batch_len = pred.size()[0]

            for i in range(batch_len):
                pred_ = pred[i]
                target_ = target[i]
                pred_ = pred_.view_as(target_)
                idx = (target_ != -100)
                target_ = target_[idx]
                pred_ = pred_[idx]
                accJoint_train += pred_.eq(target_).all().item()

            # Token acc
            target_ = target.view_as(pred)
            idx = (target_ != -100)
            target_ = target_[idx]
            pred_ = pred[idx]
            n_token += pred_.size()[0]
            accToken_train += pred_.eq(target_.view_as(pred_)).sum().item()

            
        loss_train /= len(datasets['train'])
        accJoint_train  /= len(datasets['train'])
        accToken_train  /= n_token

        # Evaluation loop - calculate accuracy and save model weights
        model.eval()        
        with torch.no_grad():
            loss_eval, accToken_eval, accJoint_eval, iter_eval, n_token = 0, 0, 0, 0, 0
            for batch, target in eval_loader:
                batch, target = batch.to(args.device), target.to(args.device)
                output = model( batch ) 

                # calculate loss and update parameters
                loss = criterion(output, target)

                # accumulate loss, accuracy
                iter_eval += 1
                loss_eval += loss.item()
                pred = output.max(1, keepdim=True)[1]


                # Joint acc
                batch_len = pred.size()[0]

                for i in range(batch_len):
                    pred_ = pred[i]
                    target_ = target[i]
                    pred_ = pred_.view_as(target_)
                    idx = (target_ != -100)
                    target_ = target_[idx]
                    pred_ = pred_[idx]
                    accJoint_eval += pred_.eq(target_).all().item()

                # Token acc
                target_ = target.view_as(pred)
                idx = (target_ != -100)
                target_ = target_[idx]
                pred_ = pred[idx]
                n_token += pred_.size()[0]
                accToken_eval += pred_.eq(target_.view_as(pred_)).sum().item()

            loss_eval /= len(datasets['eval'])
            accJoint_eval  /= len(datasets['eval'])
            accToken_eval  /= n_token

        logger.info(f"epoch: {epoch:4d}, accToken_train: {accToken_train:.5f}, accToken_eval: {accToken_eval:.5f}, accJoint_train: {accJoint_train:.5f}, accJoint_eval: {accJoint_eval:.5f} , train_loss: {loss_train:.5f}, eval_loss: {loss_eval:.5f}")
        sys.stdout.flush()
        scheduler.step(loss_eval)

        # save model
        if loss_eval < best_eval_loss:
            best_eval_loss = loss_eval
            logger.info(f"Trained model saved, eval loss: {best_eval_loss:.4f}")
            best_model_path = args.ckpt_dir / 'model.pt'
            torch.save(model.state_dict(), best_model_path)
		
        
            


    # Inference on test set
	# first load-in best model
    model = SlotTagging(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, num_class)
    model.to(args.device)
    best_model_path = args.ckpt_dir / 'model.pt'
    model.load_state_dict(torch.load(best_model_path) )
    model.eval()
    with torch.no_grad():
        with open(args.pred_file, 'w', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'tags'])
            id = 0
            for batch in test_loader:
                batch = batch.to(args.device)
                output = model( batch ) 
                pred = output.max(1, keepdim=True)[1]
                # print(pred.size()) torch.Size([1, 1, 50])
                real_len = batch[ batch != 0 ].size()[0]


                data = []
                for i in range(real_len):
                    data.append(  datasets['test'].idx2label( pred[0][0][i].item() )  )
                
                tex=''
                for i in range(len(data)):
                    tex += str(data[i])
                    if (i != len(data)-1):
                        tex += " "
                # 寫入一列資料
                writer.writerow([ str(f"test-{id}"), tex])
                id += 1


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=400)

    args = parser.parse_args()
    return args

def get_loggings(ckpt_dir):
	logger = logging.getLogger(name='TASK-slot')
	logger.setLevel(level=logging.INFO)
	# set formatter
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	# console handler
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)
	# file handler
	file_handler = logging.FileHandler(os.path.join(ckpt_dir, "record.log"))
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	return logger

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)