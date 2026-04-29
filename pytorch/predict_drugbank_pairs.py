import argparse
import json
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from train import process_dataset
from utils.initialization_utils import initialize_model
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl
from model.Classifier_model import Classifier_model


def load_params(exp_dir, gpu_override=None):
    params_path = os.path.join(exp_dir, "params.json")
    with open(params_path, "r", encoding="utf-8") as f:
        params_dict = json.load(f)

    params = SimpleNamespace(**params_dict)
    params.main_dir = os.path.dirname(os.path.abspath(__file__))
    params.exp_dir = exp_dir
    params.load_model = True
    params.eval_only = True

    if gpu_override is not None:
        params.gpu = gpu_override

    if not getattr(params, "disable_cuda", False) and torch.cuda.is_available():
        params.device = torch.device(f"cuda:{params.gpu}")
    else:
        params.device = torch.device("cpu")

    params.file_paths = {
        "train": os.path.join(params.main_dir, f"../data/{params.dataset}/{params.train_file}.txt"),
        "valid": os.path.join(params.main_dir, f"../data/{params.dataset}/{params.valid_file}.txt"),
        "test": os.path.join(params.main_dir, f"../data/{params.dataset}/{params.test_file}.txt"),
    }
    return params


def load_relation_text_map(repo_root):
    rel_map = {}
    rel_path = os.path.join(repo_root, "raw_data", "Drugbank", "id2rel.txt")
    with open(rel_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",", 3)
            if len(parts) != 4:
                continue
            raw_id, desc, _, ddi_type = parts
            rel_map[int(raw_id)] = {
                "description": desc,
                "ddi_type": ddi_type,
            }
    return rel_map


def load_test_triplets(test_path):
    triplets = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h, t, r = map(int, line.split())
            triplets.append((h, t, r))
    return triplets


def format_relation(rel_id, rel_text_map):
    info = rel_text_map.get(rel_id)
    if not info:
        return f"relation_id={rel_id}"
    return f"relation_id={rel_id}, {info['ddi_type']}, {info['description']}"


def predict_one(classifier, dataset, index, device):
    sample = dataset[index]
    batch = collate_dgl([sample])
    graph, labels, _ = move_batch_to_device_dgl(batch, device, multi_type=1)

    with torch.no_grad():
        logits = classifier(graph)
        probs = F.softmax(logits, dim=1)
        pred_id = int(torch.argmax(probs, dim=1).item())
        pred_score = float(probs[0, pred_id].item())
        true_id = int(labels.item())
        topk_scores, topk_ids = torch.topk(probs, k=min(3, probs.shape[1]), dim=1)

    topk = []
    for cls_id, score in zip(topk_ids[0].tolist(), topk_scores[0].tolist()):
        topk.append({"class_id": int(cls_id), "score": float(score)})

    return {
        "true_id": true_id,
        "pred_id": pred_id,
        "pred_score": pred_score,
        "topk": topk,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict specific DrugBank pairs with a saved baseline model.")
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "Drugbank"),
        help="Experiment directory containing params.json and best_graph_classifier.pth",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        nargs="+",
        default=[47, 51, 309, 610],
        help="Drug ids in head tail order, e.g. --pairs 47 51 309 610",
    )
    parser.add_argument("--gpu", type=int, default=None, help="Override GPU id from params.json")
    args = parser.parse_args()

    if len(args.pairs) % 2 != 0:
        raise ValueError("--pairs must contain an even number of integers")

    params = load_params(args.exp_dir, args.gpu)
    _, _, test_data = process_dataset(params)
    classifier = initialize_model(params, Classifier_model)
    classifier.eval()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rel_text_map = load_relation_text_map(repo_root)
    test_triplets = load_test_triplets(params.file_paths["test"])

    target_pairs = [(args.pairs[i], args.pairs[i + 1]) for i in range(0, len(args.pairs), 2)]

    print(f"Using model: {os.path.join(args.exp_dir, 'best_graph_classifier.pth')}")
    print(f"Device: {params.device}")
    print("")

    for head_id, tail_id in target_pairs:
        match_index = None
        for idx, (h, t, _) in enumerate(test_triplets):
            if h == head_id and t == tail_id:
                match_index = idx
                break

        if match_index is None:
            print(f"Pair ({head_id}, {tail_id}) not found in test set.")
            print("")
            continue

        result = predict_one(classifier, test_data, match_index, params.device)
        print(f"Pair: {head_id} {tail_id}")
        print(f"Test index: {match_index}")
        print(f"True label: {format_relation(result['true_id'], rel_text_map)}")
        print(f"Pred label: {format_relation(result['pred_id'], rel_text_map)}")
        print(f"Pred score: {result['pred_score']:.6f}")
        print("Top-3 predictions:")
        for item in result["topk"]:
            print(f"  - {format_relation(item['class_id'], rel_text_map)}, score={item['score']:.6f}")
        print("")


if __name__ == "__main__":
    main()
