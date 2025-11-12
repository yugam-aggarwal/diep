"""Implementation of the Direct Integration of the External Potential (DIEP) model.

DIEP is a physics-informed graph neural network that embeds electron-ion potential integrals as edge features,
providing improved physical grounding for interatomic potential predictions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import dgl
import torch
from pymatgen.core import Element
from torch import nn

import diep
from diep.config import DEFAULT_ELEMENTS
from diep.graph.compute import (
    compute_pair_vector_and_distance,
    create_line_graph,
    ensure_line_graph_compatibility,
)
from diep.layers import (
    ActivationFunction,
    DIEPIntegrator,
    EmbeddingBlock,
    GatedMLP,
    M3GNetBlock,
    MLP,
    ReduceReadOut,
    Set2SetReadOut,
    ThreeBodyInteractions,
    WeightedAtomReadOut,
    WeightedReadOut,
)
from diep.utils.cutoff import polynomial_cutoff

from ._core import MatGLModel

if TYPE_CHECKING:
    from diep.graph.converters import GraphConverter


class DIEP(MatGLModel):
    """The main DIEP model."""

    __version__ = 1

    def __init__(
        self,
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        dim_node_embedding: int = 64,
        dim_edge_embedding: int = 64,
        dim_state_embedding: int = 0,
        ntypes_state: int | None = None,
        dim_state_feats: int | None = None,
        nblocks: int = 3,
        is_intensive: bool = True,
        readout_type: Literal["set2set", "weighted_atom", "reduce_atom"] = "weighted_atom",
        task_type: Literal["classification", "regression"] = "regression",
        cutoff: float = 5.0,
        threebody_cutoff: float = 4.0,
        units: int = 64,
        ntargets: int = 1,
        niters_set2set: int = 3,
        nlayers_set2set: int = 3,
        field: Literal["node_feat", "edge_feat"] = "node_feat",
        include_state: bool = False,
        activation_type: Literal["swish", "tanh", "sigmoid", "softplus2", "softexp"] = "swish",
        dropout: float | None = None,
        grid_half_length: float = 5.0,
        base_spacing: float = 1.0,
        gaussian_sigma: float = 1.0,
        integral_mode: Literal["sum", "grid"] = "grid",
        softening_epsilon: float = 0.5,
        use_effective_charge: bool = True,
        **kwargs,
    ):
        """
        Args:
            element_types (tuple): List of elements appearing in the dataset. Default to DEFAULT_ELEMENTS.
            dim_node_embedding (int): Number of embedded atomic features
            dim_edge_embedding (int): Number of edge features
            dim_state_embedding (int): Number of hidden neurons in state embedding
            dim_state_feats (int): Number of state features after linear layer
            ntypes_state (int): Number of state labels
            nblocks (int): Number of convolution blocks
            is_intensive (bool): Whether the prediction is intensive
            readout_type (str): Readout function type, `set2set`, `weighted_atom` (default) or `reduce_atom`.
            task_type (str): `classification` or `regression` (default).
            cutoff (float): Cutoff radius of the graph
            threebody_cutoff (float): Cutoff radius for 3 body interaction
            units (int): Number of neurons in each MLP layer
            ntargets (int): Number of target properties
            field (str): Using either "node_feat" or "edge_feat" for Set2Set and Reduced readout
            niters_set2set (int): Number of set2set iterations
            nlayers_set2set (int): Number of set2set layers
            include_state (bool): Whether to include states features
            activation_type (str): Activation type. choose from 'swish', 'tanh', 'sigmoid', 'softplus2', 'softexp'
            dropout (float): Dropout probability to apply in graph layers during training
            grid_half_length (float): Half-length of the 2D integration grid for DIEP
            base_spacing (float): Base grid spacing for DIEP integration
            gaussian_sigma (float): Width parameter for Gaussian electron density
            integral_mode (str): Integration mode, either "sum" or "grid"
            softening_epsilon (float): Softening parameter to prevent 1/r singularities (default: 0.5)
            use_effective_charge (bool): If True, use sqrt(Z) instead of Z for better scaling (default: True)
            **kwargs: For future flexibility. Not used at the moment.
        """
        super().__init__()

        self.save_args(locals(), kwargs)

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None

        self.element_types = element_types or DEFAULT_ELEMENTS
        self.register_buffer(
            "atomic_number_table",
            torch.tensor([Element(el).Z for el in self.element_types], dtype=diep.float_th),
            persistent=False,
        )

        self.diep_integrator = DIEPIntegrator(
            grid_half_length=grid_half_length,
            base_spacing=base_spacing,
            sigma=gaussian_sigma,
            mode=integral_mode,
            softening_epsilon=softening_epsilon,
            use_effective_charge=use_effective_charge,
        )
        degree = self.diep_integrator.edge_dim
        degree_rbf = degree

        self.embedding = EmbeddingBlock(
            degree_rbf=degree_rbf,
            dim_node_embedding=dim_node_embedding,
            dim_edge_embedding=dim_edge_embedding,
            ntypes_node=len(element_types),
            ntypes_state=ntypes_state,
            dim_state_feats=dim_state_feats,
            include_state=include_state,
            dim_state_embedding=dim_state_embedding,
            activation=activation,
        )

        self.three_body_interactions = nn.ModuleList(
            [
                ThreeBodyInteractions(
                    update_network_atom=MLP(
                        dims=[dim_node_embedding, degree],
                        activation=nn.Sigmoid(),
                        activate_last=True,
                    ),
                    update_network_bond=GatedMLP(in_feats=degree, dims=[dim_edge_embedding], use_bias=False),
                )
                for _ in range(nblocks)
            ]
        )

        dim_state_feats = dim_state_embedding

        self.graph_layers = nn.ModuleList(
            {
                M3GNetBlock(
                    degree=degree_rbf,
                    activation=activation,
                    conv_hiddens=[units, units],
                    dim_node_feats=dim_node_embedding,
                    dim_edge_feats=dim_edge_embedding,
                    dim_state_feats=dim_state_feats,
                    include_state=include_state,
                    dropout=dropout,
                )
                for _ in range(nblocks)
            }
        )
        if is_intensive:
            input_feats = dim_node_embedding if field == "node_feat" else dim_edge_embedding
            if readout_type == "set2set":
                self.readout = Set2SetReadOut(
                    in_feats=input_feats,
                    n_iters=niters_set2set,
                    n_layers=nlayers_set2set,
                    field=field,
                )
                readout_feats = 2 * input_feats + dim_state_feats if include_state else 2 * input_feats  # type: ignore
            elif readout_type == "weighted_atom":
                self.readout = WeightedAtomReadOut(in_feats=input_feats, dims=[units, units], activation=activation)  # type: ignore[assignment]
                readout_feats = units + dim_state_feats if include_state else units  # type: ignore
            else:
                self.readout = ReduceReadOut("mean", field=field)  # type: ignore
                readout_feats = input_feats + dim_state_feats if include_state else input_feats  # type: ignore

            dims_final_layer = [readout_feats, units, units, ntargets]
            self.final_layer = MLP(dims_final_layer, activation, activate_last=False)
            if task_type == "classification":
                self.sigmoid = nn.Sigmoid()

        else:
            if task_type == "classification":
                raise ValueError("Classification task cannot be extensive.")
            self.final_layer = WeightedReadOut(
                in_feats=dim_node_embedding,
                dims=[units, units],
                num_targets=ntargets,  # type: ignore
            )

        self.n_blocks = nblocks
        self.units = units
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        self.include_state = include_state
        self.task_type = task_type
        self.is_intensive = is_intensive

    def forward(
        self,
        g: dgl.DGLGraph,
        state_attr: torch.Tensor | None = None,
        l_g: dgl.DGLGraph | None = None,
        return_all_layer_output: bool = False,
    ):
        """Performs message passing and updates node representations.

        Args:
            g : DGLGraph for a batch of graphs.
            state_attr: State attrs for a batch of graphs.
            l_g : DGLGraph for a batch of line graphs.
            return_all_layer_output: Whether to return outputs of all DIEP layers. By default, only the final layer
                output is returned.

        Returns:
            output: Output property for a batch of graphs.
        """
        node_types = g.ndata["node_type"]
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["bond_vec"] = bond_vec
        g.edata["bond_dist"] = bond_dist

        if l_g is None:
            l_g = create_line_graph(g, self.threebody_cutoff)
        else:
            l_g = ensure_line_graph_compatibility(g, l_g, self.threebody_cutoff)

        atomic_table = self.atomic_number_table
        if atomic_table.device != node_types.device:
            atomic_table = atomic_table.to(node_types.device)
        atomic_numbers = atomic_table[node_types].to(diep.float_th)
        bond_features, triplet_features = self.diep_integrator(g, l_g, atomic_numbers)

        g.edata["rbf"] = bond_features
        three_body_basis = triplet_features
        three_body_cutoff = polynomial_cutoff(g.edata["bond_dist"], self.threebody_cutoff)

        node_feat, edge_feat, state_feat = self.embedding(node_types, g.edata["rbf"], state_attr)
        fea_dict = {"diep_embedding": g.edata["rbf"]}
        for i in range(self.n_blocks):
            edge_feat = self.three_body_interactions[i](
                g,
                l_g,
                three_body_basis,
                three_body_cutoff,
                node_feat,
                edge_feat,
            )
            edge_feat, node_feat, state_feat = self.graph_layers[i](g, edge_feat, node_feat, state_feat)
            fea_dict[f"gc_{i + 1}"] = {
                "node_feat": node_feat,
                "edge_feat": edge_feat,
                "state_feat": state_feat,
            }
        g.ndata["node_feat"] = node_feat
        g.edata["edge_feat"] = edge_feat
        if self.is_intensive:
            field_vec = self.readout(g)
            readout_vec = torch.hstack([field_vec, state_feat]) if self.include_state else field_vec  # type: ignore
            fea_dict["readout"] = readout_vec
            output = self.final_layer(readout_vec)
            if self.task_type == "classification":
                output = self.sigmoid(output)
        else:
            g.ndata["atomic_properties"] = self.final_layer(g)
            fea_dict["readout"] = g.ndata["atomic_properties"]
            output = dgl.readout_nodes(g, "atomic_properties", op="sum")
        fea_dict["final"] = output
        if return_all_layer_output:
            return fea_dict
        return torch.squeeze(output)

    def predict_structure(
        self,
        structure,
        state_feats: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
        output_layers: list | None = None,
        return_features: bool = False,
    ):
        """Convenience method to featurize or predict properties of a structure with DIEP model.

        Args:
            structure: An input crystal/molecule.
            state_feats (torch.tensor): Graph attributes.
            graph_converter: Object that implements a get_graph_from_structure.
            output_layers: List of names for the layer of GNN as output. Choose from "diep_embedding",
                "gc_1", "gc_2", "gc_3", "readout", and "final". By default, all DIEP layer
                outputs are returned. Ignored if `return_features` is False.
            return_features (bool): If True, return specified layer outputs. If False, only return final output.

        Returns:
            output (dict or torch.tensor): DIEP intermediate and final layer outputs for a structure, or final
                predicted property if `return_features` is False.
        """
        allowed_output_layers = ["diep_embedding", "readout", "final"] + [f"gc_{i + 1}" for i in range(self.n_blocks)]

        if not return_features:
            output_layers = ["final"]
        elif output_layers is None:
            output_layers = allowed_output_layers
        elif not isinstance(output_layers, list) or set(output_layers).difference(allowed_output_layers):
            raise ValueError(f"Invalid output_layers, it must be a sublist of {allowed_output_layers}.")

        if graph_converter is None:
            from diep.ext.pymatgen import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)  # type: ignore

        g, lat, state_feats_default = graph_converter.get_graph(structure)
        g.edata["pbc_offshift"] = torch.matmul(g.edata["pbc_offset"], lat[0])
        g.ndata["pos"] = g.ndata["frac_coords"] @ lat[0]

        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)

        model_output = self(g=g, state_attr=state_feats, return_all_layer_output=True)

        if not return_features:
            return model_output["final"].detach()

        return {k: v for k, v in model_output.items() if k in output_layers}
