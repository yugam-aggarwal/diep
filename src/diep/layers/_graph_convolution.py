"""Graph convolution layers required for DIEP."""

from __future__ import annotations

import dgl
import dgl.function as fn
import torch
from torch import Tensor, nn
from torch.nn import Dropout, Module

from diep.layers._core import GatedMLP, MLP


class M3GNetGraphConv(Module):
    """M3GNet graph convolution layer implemented with DGL primitives."""

    def __init__(
        self,
        include_state: bool,
        edge_update_func: Module,
        edge_weight_func: Module,
        node_update_func: Module,
        node_weight_func: Module,
        state_update_func: Module | None,
    ) -> None:
        """
        Args:
            include_state: Whether global state features are present.
            edge_update_func: Edge update network.
            edge_weight_func: Linear projection producing edge weights from radial basis features.
            node_update_func: Node update network.
            node_weight_func: Linear projection producing node weights from radial basis features.
            state_update_func: State update network (optional when `include_state` is False).
        """
        super().__init__()
        self.include_state = include_state
        self.edge_update_func = edge_update_func
        self.edge_weight_func = edge_weight_func
        self.node_update_func = node_update_func
        self.node_weight_func = node_weight_func
        self.state_update_func = state_update_func

    @staticmethod
    def from_dims(
        degree: int,
        include_state: bool,
        edge_dims: list[int],
        node_dims: list[int],
        state_dims: list[int] | None,
        activation: Module,
    ) -> M3GNetGraphConv:
        """Construct a graph convolution layer from network dimensions."""
        edge_update_func = GatedMLP(in_feats=edge_dims[0], dims=edge_dims[1:])
        edge_weight_func = nn.Linear(in_features=degree, out_features=edge_dims[-1], bias=False)

        node_update_func = GatedMLP(in_feats=node_dims[0], dims=node_dims[1:])
        node_weight_func = nn.Linear(in_features=degree, out_features=node_dims[-1], bias=False)
        state_update_func = MLP(state_dims, activation, activate_last=True) if include_state else None  # type: ignore[arg-type]
        return M3GNetGraphConv(
            include_state,
            edge_update_func,
            edge_weight_func,
            node_update_func,
            node_weight_func,
            state_update_func,
        )

    def _edge_udf(self, edges: dgl.udf.EdgeBatch):
        """Edge update function used with DGL `apply_edges`."""
        vi = edges.src["v"]
        vj = edges.dst["v"]
        eij = edges.data.pop("e")
        rbf = edges.data["rbf"]
        if self.include_state:
            u = edges.src["u"]
            inputs = torch.hstack([vi, vj, eij, u])
        else:
            inputs = torch.hstack([vi, vj, eij])
        mij = {"mij": self.edge_update_func(inputs) * self.edge_weight_func(rbf)}
        return mij

    def edge_update_(self, graph: dgl.DGLGraph) -> Tensor:
        """Apply the edge update and return the new edge features."""
        graph.apply_edges(self._edge_udf)
        return graph.edata.pop("mij")

    def node_update_(self, graph: dgl.DGLGraph, state_feat: Tensor | None) -> Tensor:
        """Apply the node update and return the aggregated node features."""
        eij = graph.edata["e"]
        src_id, dst_id = graph.edges()
        vi = graph.ndata["v"][src_id]
        vj = graph.ndata["v"][dst_id]
        rbf = graph.edata["rbf"]
        if self.include_state and state_feat is not None:
            u = dgl.broadcast_edges(graph, state_feat)
            inputs = torch.hstack([vi, vj, eij, u])
        else:
            inputs = torch.hstack([vi, vj, eij])
        graph.edata["mess"] = self.node_update_func(inputs) * self.node_weight_func(rbf)
        graph.update_all(fn.copy_e("mess", "mess"), fn.sum("mess", "ve"))
        return graph.ndata.pop("ve")

    def state_update_(self, graph: dgl.DGLGraph, state_feat: Tensor) -> Tensor:
        """Update the global state features."""
        uv = dgl.readout_nodes(graph, feat="v", op="mean")
        inputs = torch.hstack([state_feat, uv])
        return self.state_update_func(inputs)  # type: ignore[arg-type]

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: Tensor,
        node_feat: Tensor,
        state_feat: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Perform edge, node, and optional state updates."""
        with graph.local_scope():
            graph.edata["e"] = edge_feat
            graph.ndata["v"] = node_feat
            if self.include_state and state_feat is not None:
                graph.ndata["u"] = dgl.broadcast_nodes(graph, state_feat)

            edge_update = self.edge_update_(graph)
            graph.edata["e"] = edge_feat + edge_update
            node_update = self.node_update_(graph, state_feat)
            graph.ndata["v"] = node_feat + node_update
            if self.include_state and state_feat is not None:
                state_feat = self.state_update_(graph, state_feat)

        return edge_feat + edge_update, node_feat + node_update, state_feat


class M3GNetBlock(Module):
    """Stacked graph convolution block following the M3GNet design."""

    def __init__(
        self,
        degree: int,
        activation: Module,
        conv_hiddens: list[int],
        dim_node_feats: int,
        dim_edge_feats: int,
        dim_state_feats: int = 0,
        include_state: bool = False,
        dropout: float | None = None,
    ) -> None:
        super().__init__()

        self.include_state = include_state
        self.activation = activation

        if include_state:
            edge_in = 2 * dim_node_feats + dim_edge_feats + dim_state_feats
            node_in = 2 * dim_node_feats + dim_edge_feats + dim_state_feats
            state_in = dim_node_feats + dim_state_feats
            state_dims = [state_in, *conv_hiddens, dim_state_feats]
        else:
            edge_in = 2 * dim_node_feats + dim_edge_feats
            node_in = 2 * dim_node_feats + dim_edge_feats
            state_dims = None

        self.conv = M3GNetGraphConv.from_dims(
            degree=degree,
            include_state=include_state,
            edge_dims=[edge_in, *conv_hiddens, dim_edge_feats],
            node_dims=[node_in, *conv_hiddens, dim_node_feats],
            state_dims=state_dims,  # type: ignore[arg-type]
            activation=self.activation,
        )
        self.dropout = Dropout(dropout) if dropout else None

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: Tensor,
        node_feat: Tensor,
        state_feat: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Run a forward pass through the block."""
        edge_feat, node_feat, state_feat = self.conv(graph, edge_feat, node_feat, state_feat)

        if self.dropout:
            edge_feat = self.dropout(edge_feat)
            node_feat = self.dropout(node_feat)
            if state_feat is not None:
                state_feat = self.dropout(state_feat)

        return edge_feat, node_feat, state_feat
