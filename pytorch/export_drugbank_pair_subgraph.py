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


def format_relation(rel_id, rel_text_map):
    info = rel_text_map.get(rel_id)
    if not info:
        return {"relation_id": rel_id, "ddi_type": "", "description": ""}
    return {
        "relation_id": rel_id,
        "ddi_type": info["ddi_type"],
        "description": info["description"],
    }


def forward_with_graph(classifier, graph):
    classifier.global_graph.ndata["h"] = classifier.embedding_model(classifier.global_graph)
    graph.ndata["h"] = classifier.global_graph.nodes[graph.ndata["idx"]].data["h"]
    graph.ndata["repr"] = classifier.global_graph.nodes[graph.ndata["idx"]].data["repr"]

    head_ids = (graph.ndata["id"] == 1).nonzero().squeeze(1)
    tail_ids = (graph.ndata["id"] == 2).nonzero().squeeze(1)

    complete_graph = classifier.gsl_model(graph)

    gsl_hidden = complete_graph.ndata["repr"]
    head_hidden = gsl_hidden[head_ids].view(-1, classifier.score_dim)
    tail_hidden = gsl_hidden[tail_ids].view(-1, classifier.score_dim)
    g_out = torch.mean(gsl_hidden, dim=0, keepdim=True).view(-1, classifier.score_dim)

    pred = torch.cat([g_out, head_hidden, tail_hidden], dim=1)
    logits = classifier.W_final(pred)
    return logits, complete_graph


def build_entity_name_map(dataset):
    return {int(k): v for k, v in dataset.id2entity.items()}


def export_pair(classifier, dataset, test_triplets, target_pair, rel_text_map, output_dir, device, topk_edges):
    head_id, tail_id = target_pair
    match_index = None
    for idx, (h, t, _) in enumerate(test_triplets):
        if h == head_id and t == tail_id:
            match_index = idx
            break

    if match_index is None:
        print(f"Pair ({head_id}, {tail_id}) not found in test set.")
        return

    sample = dataset[match_index]
    batch = collate_dgl([sample])
    graph, labels, _ = move_batch_to_device_dgl(batch, device, multi_type=1)

    with torch.no_grad():
        logits, complete_graph = forward_with_graph(classifier, graph)
        probs = F.softmax(logits, dim=1)

    pred_id = int(torch.argmax(probs, dim=1).item())
    pred_score = float(probs[0, pred_id].item())
    true_id = int(labels.item())

    src, dst = complete_graph.edges()
    weights = complete_graph.edata["weight"].squeeze(1)
    rel_types = complete_graph.edata["type"]
    original_mask = complete_graph.edata.get("is_original_edge", torch.zeros_like(weights).unsqueeze(1)).squeeze(1)
    node_idx = complete_graph.ndata["idx"]
    entity_map = build_entity_name_map(dataset)

    edge_records = []
    for i in range(complete_graph.number_of_edges()):
        src_local = int(src[i].item())
        dst_local = int(dst[i].item())
        src_global = int(node_idx[src_local].item())
        dst_global = int(node_idx[dst_local].item())
        rel_id = int(rel_types[i].item())
        weight = float(weights[i].item())
        edge_records.append(
            {
                "src_local": src_local,
                "dst_local": dst_local,
                "src_global": src_global,
                "dst_global": dst_global,
                "src_entity": entity_map.get(src_global, str(src_global)),
                "dst_entity": entity_map.get(dst_global, str(dst_global)),
                "relation_id": rel_id,
                "relation_name": dataset.id2relation.get(rel_id, str(rel_id)),
                "weight": weight,
                "is_original_edge": bool(original_mask[i].item()),
            }
        )

    edge_records.sort(key=lambda x: x["weight"], reverse=True)
    edge_records = edge_records[:topk_edges]

    result = {
        "pair": {
            "head_id": head_id,
            "tail_id": tail_id,
            "head_entity": entity_map.get(head_id, str(head_id)),
            "tail_entity": entity_map.get(tail_id, str(tail_id)),
            "test_index": match_index,
        },
        "true_label": format_relation(true_id, rel_text_map),
        "pred_label": format_relation(pred_id, rel_text_map),
        "pred_score": pred_score,
        "top_edges": edge_records,
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"pair_{head_id}_{tail_id}_subgraph.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Exported: {output_path}")
    print(f"Pair: {head_id} {tail_id}")
    print(f"True label: {result['true_label']}")
    print(f"Pred label: {result['pred_label']}")
    print(f"Pred score: {pred_score:.6f}")
    print("Top weighted edges:")
    for edge in edge_records[:10]:
        print(
            f"  - {edge['src_entity']} -> {edge['dst_entity']}, "
            f"rel={edge['relation_name']}, weight={edge['weight']:.6f}, original={edge['is_original_edge']}"
        )
    print("")


def main():
    parser = argparse.ArgumentParser(description="Export weighted subgraph edges for specific DrugBank pairs.")
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "Drugbank"),
    )
    parser.add_argument(
        "--pairs",
        type=int,
        nargs="+",
        default=[47, 51, 309, 610],
        help="Drug ids in head tail order",
    )
    parser.add_argument("--gpu", type=int, default=None, help="Override GPU id from params.json")
    parser.add_argument("--topk_edges", type=int, default=30, help="Number of highest-weight edges to export")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "Drugbank", "pair_exports"),
    )
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
    for pair in target_pairs:
        export_pair(
            classifier,
            test_data,
            test_triplets,
            pair,
            rel_text_map,
            args.output_dir,
            params.device,
            args.topk_edges,
        )


if __name__ == "__main__":
    main()
