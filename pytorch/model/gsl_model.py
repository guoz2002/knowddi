import torch
import torch.nn as nn
from collections import OrderedDict
import dgl
from dgl.nn.functional import edge_softmax


class MLP(nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_layers, batch_norm=True, dropout=0.):
        super(MLP, self).__init__()
        layer_list = OrderedDict()
        in_dim = inp_dim
        for l in range(num_layers):
            if l < num_layers - 1:
                layer_list['fc{}'.format(l)] = nn.Linear(in_dim, hidden_dim)
                if batch_norm:
                    layer_list['norm{}'.format(l)] = nn.BatchNorm1d(num_features=hidden_dim)
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()
                if dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=dropout)
                in_dim = hidden_dim
            else:
                layer_list['fc_score'] = nn.Linear(in_dim, 1)
        self.network = nn.Sequential(layer_list)

    def forward(self, emb):
        out = self.network(emb)
        return out


class EdgeGateNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.gate_head = nn.Linear(hidden_dim, 1)
        self.denoise_head = nn.Linear(hidden_dim, 1)
        self.completion_head = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        hidden = self.encoder(features)
        return {
            'hidden': hidden,
            'gate': self.gate_head(hidden),
            'denoise': self.denoise_head(hidden),
            'completion': self.completion_head(hidden),
        }


class NodeUpdateModule(nn.Module):
    def __init__(self, emb_dim):
        super(NodeUpdateModule, self).__init__()
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.activation = nn.ReLU()

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(node.data['h'])
        return {'h': h}


class graph_structure_learner(torch.nn.Module):
    def __init__(self, params, rel_emb):
        super().__init__()
        self.params = params
        self.lamda = params.lamda
        self.edge_softmax = params.edge_softmax
        self.sparsify = params.sparsify
        self.threshold = params.threshold
        self.func_num = params.func_num
        self.emb_dim = params.emb_dim
        self.rel_emb = rel_emb
        self.gsl_rel_emb_dim = params.gsl_rel_emb_dim
        self.gsl_mode = getattr(params, 'gsl_mode', 'baseline')
        self.use_denoise = bool(getattr(params, 'use_denoise', 0))
        self.use_completion = bool(getattr(params, 'use_completion', 0))
        self.gate_hidden_dim = getattr(params, 'gate_hidden_dim', params.MLP_hidden_dim)
        self.gate_dropout = getattr(params, 'gate_dropout', params.MLP_dropout)
        self.completion_threshold = getattr(params, 'completion_threshold', self.threshold)
        self.completion_topk = getattr(params, 'completion_topk', 0)
        self.denoise_alpha = getattr(params, 'denoise_alpha', 1.0)
        self.completion_alpha = getattr(params, 'completion_alpha', 1.0)
        self.role_emb_dim = max(4, self.gate_hidden_dim // 4)
        self.role_emb = nn.Embedding(3, self.role_emb_dim)
        self.last_stats = {}

        if self.func_num == 1:
            if self.params.gsl_has_edge_emb:
                self.MLP = MLP(
                    inp_dim=self.emb_dim + self.gsl_rel_emb_dim,
                    hidden_dim=params.MLP_hidden_dim,
                    num_layers=params.MLP_num_layers,
                    batch_norm=True,
                    dropout=params.MLP_dropout,
                )
            else:
                self.MLP = MLP(
                    inp_dim=self.emb_dim,
                    hidden_dim=params.MLP_hidden_dim,
                    num_layers=params.MLP_num_layers,
                    batch_norm=True,
                    dropout=params.MLP_dropout,
                )
        elif self.func_num == 2:
            self.l1_norm = nn.PairwiseDistance(p=1, keepdim=True)
        elif self.func_num == 3:
            self.l2_norm = nn.PairwiseDistance(p=2, keepdim=True)
        elif self.func_num == 4:
            self.cos = nn.CosineSimilarity(dim=1)

        adaptive_input_dim = self.emb_dim + 3
        if self.params.gsl_has_edge_emb:
            adaptive_input_dim += self.gsl_rel_emb_dim
        adaptive_input_dim += self.role_emb_dim * 2
        self.edge_gate_network = EdgeGateNetwork(
            input_dim=adaptive_input_dim,
            hidden_dim=self.gate_hidden_dim,
            dropout=self.gate_dropout,
        )

    def compute_similarity(self, src_hidden, dst_hidden, rel_embedding, func_num):
        if func_num == 0:
            weights = torch.ones((src_hidden.shape[0], 1)).to(device=src_hidden.device)
        elif func_num == 1:
            if self.params.gsl_has_edge_emb:
                weights = self.MLP(torch.cat([torch.exp(-torch.abs(src_hidden - dst_hidden)), rel_embedding], dim=1))
            else:
                weights = self.MLP(torch.exp(-torch.abs(src_hidden - dst_hidden)))
        elif func_num == 2:
            weights = self.l1_norm(src_hidden, dst_hidden)
        elif func_num == 3:
            weights = self.l2_norm(src_hidden, dst_hidden)
        elif func_num == 4:
            weights = self.cos(src_hidden, dst_hidden).unsqueeze(1)

        return weights

    def _build_adaptive_features(self, complete_graph, ori_graph, row, col, rel_embedding):
        node_roles = complete_graph.ndata['id'].clamp(min=0, max=2).long()
        role_features = self.role_emb(node_roles)
        degree = (ori_graph.in_degrees() + ori_graph.out_degrees()).float().to(complete_graph.device)
        degree = torch.log1p(degree).unsqueeze(1)

        src_hidden = complete_graph.ndata['h'][row]
        dst_hidden = complete_graph.ndata['h'][col]
        src_role = role_features[row]
        dst_role = role_features[col]
        src_degree = degree[row]
        dst_degree = degree[col]
        is_original_edge = complete_graph.edata['is_original_edge'].float()

        feature_list = [
            torch.exp(-torch.abs(src_hidden - dst_hidden)),
            is_original_edge,
            src_degree,
            dst_degree,
            src_role,
            dst_role,
        ]
        if self.params.gsl_has_edge_emb:
            feature_list.insert(1, rel_embedding)
        return torch.cat(feature_list, dim=1)

    def _apply_completion_topk(self, complete_graph, completion_score):
        if self.completion_topk <= 0:
            return torch.ones_like(completion_score, dtype=torch.bool)

        row, _ = complete_graph.all_edges()
        batch_num_nodes = complete_graph.batch_num_nodes()
        block_begin_idx = torch.cat([batch_num_nodes.new_zeros(1), batch_num_nodes.cumsum(dim=0)[:-1]], dim=0)
        block_end_idx = batch_num_nodes.cumsum(dim=0)
        keep_mask = torch.zeros_like(completion_score, dtype=torch.bool)
        original_mask = complete_graph.edata['is_original_edge'].squeeze(1).bool()

        for idx_b, idx_e in zip(block_begin_idx.tolist(), block_end_idx.tolist()):
            graph_edge_mask = (row >= idx_b) & (row < idx_e)
            candidate_mask = graph_edge_mask & (~original_mask)
            candidate_idx = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
            if candidate_idx.numel() == 0:
                continue
            if candidate_idx.numel() <= self.completion_topk:
                keep_mask[candidate_idx] = True
                continue
            candidate_scores = completion_score[candidate_idx]
            topk_idx = torch.topk(candidate_scores, self.completion_topk, sorted=False).indices
            keep_mask[candidate_idx[topk_idx]] = True

        return keep_mask

    def _record_stats(self, complete_graph):
        weight = complete_graph.edata['weight']
        original_mask = complete_graph.edata['is_original_edge'].bool()
        completion_mask = ~original_mask
        active_mask = weight > 0

        stats = {
            'edge_keep_ratio': active_mask.float().mean().item(),
            'edge_weight_mean': weight.mean().item(),
            'origin_edge_keep_ratio': active_mask[original_mask].float().mean().item() if original_mask.any() else 0.0,
            'completion_activation_ratio': active_mask[completion_mask].float().mean().item() if completion_mask.any() else 0.0,
        }
        if 'gate_score' in complete_graph.edata:
            stats['gate_score_mean'] = complete_graph.edata['gate_score'].mean().item()
        if 'denoise_score' in complete_graph.edata:
            stats['denoise_score_mean'] = complete_graph.edata['denoise_score'][original_mask].mean().item() if original_mask.any() else 0.0
        if 'completion_score' in complete_graph.edata:
            stats['completion_score_mean'] = complete_graph.edata['completion_score'][completion_mask].mean().item() if completion_mask.any() else 0.0
        self.last_stats = stats

    def forward(self, complete_graph, ori_graph):
        n_feat = complete_graph.ndata['h']
        row, col = complete_graph.all_edges()
        rel_embedding = self.rel_emb(complete_graph.edata['type'])
        gsl_mode = getattr(self, 'gsl_mode', 'baseline')
        use_denoise = getattr(self, 'use_denoise', False)
        use_completion = getattr(self, 'use_completion', False)
        denoise_alpha = getattr(self, 'denoise_alpha', 1.0)
        completion_alpha = getattr(self, 'completion_alpha', 1.0)
        completion_threshold = getattr(self, 'completion_threshold', self.threshold)

        if gsl_mode == 'baseline':
            weights = self.compute_similarity(n_feat[row], n_feat[col], rel_embedding, func_num=self.params.func_num)
            complete_graph.edata['weight'] = weights

            ori_row, ori_col = ori_graph.all_edges()
            ori_e_weight = torch.ones((ori_graph.number_of_edges(), 1), dtype=torch.float, device=ori_graph.device)
            complete_graph.edges[ori_row, ori_col].data['weight'] = (
                (1 - self.lamda) * complete_graph.edges[ori_row, ori_col].data['weight'] + self.lamda * ori_e_weight
            )
        else:
            adaptive_features = self._build_adaptive_features(complete_graph, ori_graph, row, col, rel_embedding)
            edge_outputs = self.edge_gate_network(adaptive_features)
            gate_score = torch.sigmoid(edge_outputs['gate'])
            complete_graph.edata['gate_score'] = gate_score
            final_weight = gate_score.clone()
            original_mask = complete_graph.edata['is_original_edge'].float()

            if use_denoise:
                denoise_score = torch.sigmoid(edge_outputs['denoise'])
                complete_graph.edata['denoise_score'] = denoise_score
                final_weight = torch.where(
                    original_mask.bool(),
                    final_weight * (denoise_alpha * denoise_score),
                    final_weight,
                )

            if use_completion:
                completion_score = torch.sigmoid(edge_outputs['completion'])
                keep_mask = self._apply_completion_topk(complete_graph, completion_score.squeeze(1)).unsqueeze(1)
                keep_mask = keep_mask & (completion_score >= completion_threshold)
                completion_score = completion_score * keep_mask.float()
                complete_graph.edata['completion_score'] = completion_score
                final_weight = torch.where(
                    original_mask.bool(),
                    final_weight,
                    final_weight * (completion_alpha * completion_score),
                )

            complete_graph.edata['weight'] = final_weight

        if self.edge_softmax:
            complete_graph.edata['weight'] = edge_softmax(complete_graph, complete_graph.edata['weight'])
        if self.sparsify:
            complete_graph.edata['weight'] = torch.where(
                complete_graph.edata['weight'] > self.threshold,
                complete_graph.edata['weight'],
                torch.zeros(complete_graph.edata['weight'].shape).to(complete_graph.device),
            )

        self._record_stats(complete_graph)
        return complete_graph


class gsl_layer(torch.nn.Module):
    def __init__(self, params, rel_emb):
        super().__init__()
        self.params = params
        self.emb_dim = params.emb_dim
        self.rel_emb = rel_emb
        self.graph_structure_learner = graph_structure_learner(self.params, self.rel_emb)
        self.apply_mod = NodeUpdateModule(self.emb_dim)
        self.last_stats = {}

    def forward(self, complete_graph, ori_graph):
        def msg_func(edges):
            w = edges.data['weight']
            x = edges.src['h']
            msg = x * w
            return {'msg': msg}

        def reduce_func(nodes):
            return {'h': torch.sum(nodes.mailbox['msg'], dim=1)}

        complete_graph = self.graph_structure_learner(complete_graph, ori_graph)
        self.last_stats = self.graph_structure_learner.last_stats
        complete_graph.update_all(msg_func, reduce_func)
        complete_graph.apply_nodes(func=self.apply_mod)

        return complete_graph


class gsl_model(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.ni_layer = params.num_infer_layers
        self.gsl_layers = nn.ModuleList()
        self.num_rels = params.num_rels
        self.aug_num_rels = params.aug_num_rels
        self.gsl_rel_emb_dim = params.gsl_rel_emb_dim
        self.rel_emb = nn.Embedding(self.aug_num_rels, self.gsl_rel_emb_dim, sparse=False)
        self.last_graph_stats = {}
        for i in range(self.ni_layer):
            self.gsl_layers.append(gsl_layer(params, self.rel_emb))

    def build_full_connect_graph(self, ori_graph):
        batch_num_nodes = ori_graph.batch_num_nodes()
        block_begin_idx = torch.cat([batch_num_nodes.new_zeros(1), batch_num_nodes.cumsum(dim=0)[:-1]], dim=0)
        block_end_idx = batch_num_nodes.cumsum(dim=0)
        dense_adj = torch.zeros(
            (ori_graph.num_nodes(), ori_graph.num_nodes()),
            dtype=torch.float,
            device=ori_graph.device,
        )
        for idx_b, idx_e in zip(block_begin_idx, block_end_idx):
            dense_adj[idx_b:idx_e, idx_b:idx_e] = 1.
        row, col = torch.nonzero(dense_adj).t().contiguous()

        complete_graph = dgl.graph((row, col)).to(ori_graph.device)
        batch_num_edges = torch.pow(batch_num_nodes, 2)
        complete_graph.set_batch_num_nodes(batch_num_nodes)
        complete_graph.set_batch_num_edges(batch_num_edges)
        complete_graph.ndata['h'] = ori_graph.ndata['h']
        complete_graph.ndata['repr'] = ori_graph.ndata['repr']
        complete_graph.ndata['id'] = ori_graph.ndata['id']
        complete_graph.ndata['idx'] = ori_graph.ndata['idx']
        complete_graph.edata['type'] = torch.full(
            (complete_graph.number_of_edges(),),
            self.num_rels + 1,
            dtype=ori_graph.edata['type'].dtype,
            device=ori_graph.device,
        )
        complete_graph.edata['is_original_edge'] = torch.zeros(
            (complete_graph.number_of_edges(), 1),
            dtype=torch.float,
            device=ori_graph.device,
        )
        ori_row, ori_col = ori_graph.all_edges()
        complete_graph.edges[ori_row, ori_col].data['type'] = ori_graph.edges[ori_row, ori_col].data['type']
        complete_graph.edges[ori_row, ori_col].data['is_original_edge'] = torch.ones(
            (ori_graph.number_of_edges(), 1),
            dtype=torch.float,
            device=ori_graph.device,
        )

        return complete_graph

    def forward(self, g):
        ori_graph = g
        complete_graph = self.build_full_connect_graph(ori_graph)
        layer_stats = []

        for i in range(self.ni_layer):
            complete_graph = self.gsl_layers[i](complete_graph, ori_graph)
            layer_stats.append(self.gsl_layers[i].last_stats)
            complete_graph.ndata['repr'] = torch.cat([complete_graph.ndata['repr'], complete_graph.ndata['h'].unsqueeze(1)], dim=1)

        if layer_stats:
            self.last_graph_stats = {
                key: sum(layer_stat.get(key, 0.0) for layer_stat in layer_stats) / len(layer_stats)
                for key in layer_stats[0]
            }
        else:
            self.last_graph_stats = {}

        return complete_graph
