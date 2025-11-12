"""This package implements the layers for different Graph Neural Networks."""

from __future__ import annotations

from diep.layers._activations import ActivationFunction
from diep.layers._atom_ref import AtomRef
from diep.layers._basis import FourierExpansion, RadialBesselFunction, SphericalBesselWithHarmonics
from diep.layers._bond import BondExpansion
from diep.layers._core import MLP, EdgeSet2Set, GatedEquivariantBlock, GatedMLP, MLP_norm, build_gated_equivariant_mlp
from diep.layers._diep import DIEPIntegrator
from diep.layers._embedding import EmbeddingBlock
from diep.layers._graph_convolution import M3GNetBlock, M3GNetGraphConv
from diep.layers._norm import GraphNorm
from diep.layers._readout import (
    AttentiveFPReadout,
    GlobalPool,
    ReduceReadOut,
    Set2SetReadOut,
    WeightedAtomReadOut,
    WeightedReadOut,
    WeightedReadOutPair,
)
from diep.layers._three_body import ThreeBodyInteractions
from diep.layers._zbl import NuclearRepulsion
