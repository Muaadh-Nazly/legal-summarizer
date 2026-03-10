"""
GNN Model Inference Utilities

Loads a trained GNN encoder and converts clause graph JSON into
PyTorch Geometric format for use in clause ranking. Model type is detected from
checkpoint (model_type key or state_dict keys).
"""

import json
from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import HGTConv, Linear, RGCNConv


def _is_rgcn_checkpoint(state_dict: dict) -> bool:
    """Return True if state_dict looks like R-GCN."""
    keys = set(state_dict.keys())
    return any("convs.0.weight" in k for k in keys) and any(
        "convs.0.comp" in k for k in keys
    )


class RGCNModel(torch.nn.Module):
    """R-GCN encoder matching Graph Construction training."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 3,
        num_relations: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_ssl: bool = False,
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(RGCNConv(in_dim, hidden_dim, num_relations, num_bases=50))
        for _ in range(num_layers - 1):
            self.convs.append(
                RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=50)
            )
        self.lin = torch.nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout
        self.use_ssl = use_ssl
        if use_ssl:
            self.reconstruction_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, in_dim),
            )

    def forward(self, data, return_embeddings: bool = False):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        node_embeddings = x
        out = self.lin(x)
        if return_embeddings:
            return node_embeddings
        return out


def load_model(
    model_path: str, in_dim: int = 768, device: str = "cuda"
) -> torch.nn.Module:
    """
    Load trained GNN model from checkpoint.

    Args:
        model_path: Path to saved model
        in_dim: Input feature dimension
        device: Device to load model

    Returns:
        Loaded GNN model in eval mode
    """

    # Define model architecture.
    class HGTModel(torch.nn.Module):
        def __init__(
            self,
            in_dim: int,
            hidden_dim: int = 256,
            out_dim: int = 3,
            num_edge_types: int = 4,
            num_heads: int = 4,
            num_layers: int = 2,
            dropout: float = 0.2,
            use_ssl: bool = False,
        ):
            super().__init__()
            self.use_ssl = use_ssl
            self.num_edge_types = num_edge_types

            self.node_type_names = ["node"]
            edge_type_names = [
                "follows",
                "green_support",
                "red_opposes",
                "inter_citation",
            ]
            self.edge_types = [
                (self.node_type_names[0], edge_type_names[i], self.node_type_names[0])
                for i in range(num_edge_types)
            ]
            self.metadata = (self.node_type_names, self.edge_types)

            # Single node-type embedding
            self.node_type_emb = torch.nn.Embedding(1, in_dim)

            # HGT layers
            self.convs = torch.nn.ModuleList()
            self.convs.append(HGTConv(in_dim, hidden_dim, self.metadata, num_heads))
            for _ in range(num_layers - 1):
                self.convs.append(
                    HGTConv(hidden_dim, hidden_dim, self.metadata, num_heads)
                )

            # Final classification layer
            self.lin = Linear(hidden_dim, out_dim)
            self.dropout = dropout

            # SSL reconstruction head
            if use_ssl:
                self.reconstruction_head = torch.nn.Sequential(
                    Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    Linear(hidden_dim, in_dim),
                )

        def forward(self, data, return_embeddings: bool = False):
            node_type = torch.zeros(
                data.x.size(0), dtype=torch.long, device=data.x.device
            )
            edge_type = data.edge_type
            edge_index = data.edge_index

            original_embeddings = getattr(data, "original_embeddings", None)
            if original_embeddings is None and self.use_ssl:
                original_embeddings = data.x.clone()

            # Small-weight node type embedding
            x = data.x + self.node_type_emb(node_type) * 0.1

            # Convert to HeteroData format
            x_dict = {self.node_type_names[0]: x}
            edge_index_dict = {}

            # Group edges by type
            for et in range(self.num_edge_types):
                mask = edge_type == et
                if mask.sum() > 0:
                    edge_key = self.edge_types[et]
                    edge_index_dict[edge_key] = edge_index[:, mask]

            # HGT message passing
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
                if x_dict[self.node_type_names[0]] is not None:
                    x = x_dict[self.node_type_names[0]]
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                    x_dict[self.node_type_names[0]] = x
                else:
                    x = data.x

            node_embeddings = x
            out = self.lin(x)

            if return_embeddings:
                return node_embeddings
            return out

        def compute_reconstruction_loss(
            self, embeddings, original_embeddings, external_mask
        ):
            if not self.use_ssl or external_mask.sum() == 0:
                return torch.tensor(0.0, device=embeddings.device)
            reconstructed = self.reconstruction_head(embeddings[external_mask])
            targets = original_embeddings[external_mask]
            return F.mse_loss(reconstructed, targets, reduction="mean")

    # Load model state
    try:
        checkpoint = torch.load(model_path, map_location=device)

        # Infer model parameters from checkpoint
        if isinstance(checkpoint, dict):
            saved_cfg = checkpoint.get("config", {}) or {}
            if "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Detect model type
            model_type = (checkpoint.get("model_type") or "").lower().strip()
            if not model_type and isinstance(checkpoint, dict):
                model_type = "rgcn" if _is_rgcn_checkpoint(state_dict) else "hgt"

            use_ssl = any("reconstruction_head" in k for k in state_dict.keys())
            hidden_dim = int(saved_cfg.get("hidden_dim", 256))
            if "lin.weight" in state_dict:
                hidden_dim = int(state_dict["lin.weight"].shape[1])
            out_dim = 3
            if "lin.weight" in state_dict:
                out_dim = int(state_dict["lin.weight"].shape[0])

            if model_type == "rgcn":
                num_relations = int(saved_cfg.get("num_relations", 4))
                num_layers = int(saved_cfg.get("num_layers", 2))
                print("📥 Loading R-GCN encoder...")
                model = RGCNModel(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    out_dim=out_dim,
                    num_relations=num_relations,
                    num_layers=num_layers,
                    dropout=float(saved_cfg.get("dropout", 0.2)),
                    use_ssl=use_ssl,
                )
            else:
                print("📥 Loading HGT encoder...")
                num_edge_types = int(saved_cfg.get("num_edge_types", 4))
                model = HGTModel(
                    in_dim=in_dim,
                    num_edge_types=num_edge_types,
                    hidden_dim=hidden_dim,
                    out_dim=out_dim,
                    num_heads=int(saved_cfg.get("num_heads", 4)),
                    num_layers=int(saved_cfg.get("num_layers", 2)),
                    use_ssl=use_ssl,
                )

            # Load state dict
            try:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"⚠️  Missing keys (up to 10): {list(missing)[:10]}")
                if unexpected:
                    print(f"⚠️  Unexpected keys (up to 10): {list(unexpected)[:10]}")
            except Exception as e:
                print(f"⚠️  Warning: Could not load all weights: {e}")
                print("   Using partial weights or default initialization")
        else:
            # Assume it's a full model object
            model = checkpoint
            if hasattr(model, "eval"):
                model.eval()
    except Exception as e:
        print(f"⚠️  Error loading model: {e}")
        print("   Creating model with default architecture")
        model = HGTModel(in_dim=in_dim, use_ssl=False)

    model = model.to(device)
    model.eval()
    return model


def graph_json_to_pyg_inference(
    graph_json_path: str, tokenizer, bert_model, device: str = "cuda"
) -> Optional[Data]:
    """
    Convert graph JSON to PyTorch Geometric Data object for inference.

    Args:
        graph_json_path: Path to graph JSON file
        tokenizer: InCaseLawBert tokenizer
        bert_model: InCaseLawBert model
        device: Device to run on

    Returns:
        PyG Data object or None if error
    """
    try:
        with open(graph_json_path, "r", encoding="utf-8") as f:
            graph = json.load(f)

        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        if not nodes:
            return None

        # Filter to only internal nodes
        internal_nodes = [n for n in nodes if not n.get("external", False)]

        if not internal_nodes:
            return None

        # Node ID to index mapping
        id2idx = {n["id"]: i for i, n in enumerate(internal_nodes)}

        # Get embeddings for internal nodes
        texts = [n["text"] for n in internal_nodes]

        with torch.no_grad():
            tokens = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            out = bert_model(**tokens)
            x = out.last_hidden_state[:, 0, :].cpu()

        # Labels
        label_map = {"Premise": 0, "Opposition": 1, "Claim": 2}
        y = torch.tensor(
            [label_map.get(n.get("label", "None"), 0) for n in internal_nodes],
            dtype=torch.long,
        )
        node_type = torch.zeros(len(internal_nodes), dtype=torch.long)

        # Train mask
        train_mask = torch.ones(len(internal_nodes), dtype=torch.bool)

        # Edges
        rel_map = {
            "follows": 0,
            "green_support": 1,
            "red_opposes": 2,
            "inter_citation": 3,
        }

        src, tgt, etype = [], [], []
        for edge in edges:
            relation = edge.get("relation", "")
            if relation not in rel_map:
                continue

            source = edge.get("source")
            target = edge.get("target")

            # Only include edges between internal nodes
            if source in id2idx and target in id2idx:
                src.append(id2idx[source])
                tgt.append(id2idx[target])
                etype.append(rel_map[relation])

        if len(src) == 0:
            # Create minimal graph with no edges
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)
        else:
            edge_index = torch.tensor([src, tgt], dtype=torch.long)
            edge_type = torch.tensor(etype, dtype=torch.long)

        # Create PyG Data object
        data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            edge_type=edge_type,
            train_mask=train_mask,
            node_type=node_type,
            doc_id=graph.get("doc_id", ""),
        )

        return data

    except Exception as e:
        print(f"⚠️  Error converting graph to PyG: {e}")
        import traceback

        traceback.print_exc()
        return None


def extract_embeddings(model, data: Data, device: str = "cuda") -> torch.Tensor:
    """
    Extract node embeddings from HGT model.
    Args:
        model: Trained HGT model
        data: PyG Data object
        device: Device to run on

    Returns:
        Tensor of shape [num_nodes, embedding_dim]
    """
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        if hasattr(model, "forward"):
            try:
                out = model(data, return_embeddings=True)
                embeddings = out[1] if isinstance(out, tuple) else out
            except TypeError:
                embeddings = data.x
        else:
            embeddings = data.x

    return embeddings.cpu()
