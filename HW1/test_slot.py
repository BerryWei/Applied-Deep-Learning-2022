import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from torch.utils.data import DataLoader
import torch
from model import SlotTagging
from dataset import SeqTaggingClsDataset
from utils import Vocab
import csv


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "tag2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    test_loader = DataLoader(dataset=dataset, batch_size=1, 
                                shuffle=False, collate_fn=dataset.collatePred_fn )  



    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    num_class = len(dataset.label_mapping)
    model = SlotTagging(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, num_class)
    model.to(args.device)

    model.eval()
    # load weights into model
    best_model_path = args.ckpt_path / 'model.pt'
    device = torch.device('cpu')
    model.load_state_dict( torch.load(best_model_path, map_location=device))
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
                    data.append(  dataset.idx2label( pred[0][0][i].item() )  )
                
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
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
