"""DIEP layer implementing electron-ion potential integrations.

This module provides the DIEPIntegrator layer that computes physics-informed edge features
based on electron-ion potential integrals over 2D grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

import diep


def _compute_delta_area(grid_axis: torch.Tensor) -> torch.Tensor:
    """Return the area element for a uniform Cartesian grid."""
    if grid_axis.numel() < 2:
        return torch.tensor(1.0, dtype=diep.float_th, device=grid_axis.device)
    step = grid_axis[1] - grid_axis[0]
    return step * step


@dataclass
class DIEPGrid:
    """Utility container storing the 2D grid definition."""

    half_length: float
    spacing: float
    device: torch.device

    def __post_init__(self):
        num_points = int(round(2 * self.half_length / self.spacing)) + 1
        axis = torch.linspace(
            -self.half_length,
            self.half_length,
            steps=num_points,
            dtype=diep.float_th,
            device=self.device,
        )
        self.axis = axis
        self.points = torch.stack(torch.meshgrid(axis, axis, indexing="xy"), dim=-1).reshape(-1, 2)
        self.delta_area = _compute_delta_area(axis)


def _gaussian_density(diff_sq: torch.Tensor, sigma: float) -> torch.Tensor:
    """Compute Gaussian electron density contribution with squared-distance decay.
    
    Returns normalized density for 2D integration.
    """
    denom = torch.tensor(sigma, dtype=diff_sq.dtype, device=diff_sq.device)
    # Normalize by (pi * sigma) for proper 2D Gaussian normalization
    normalization = torch.pi * denom
    return torch.exp(-torch.clamp(diff_sq, min=0.0) / denom) / normalization


def _normalize(vec: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return the unit vector, guarding against zero norm."""
    norm = torch.norm(vec)
    if norm <= eps:
        raise ValueError("Cannot normalise a zero-length vector.")
    return vec / norm


def _choose_perpendicular_unit(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return a unit vector perpendicular to the provided direction."""
    device = vec.device
    dtype = vec.dtype
    candidates = [
        torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device),
        torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device),
        torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device),
    ]
    for ref in candidates:
        trial = ref - torch.dot(ref, vec) * vec
        length = torch.norm(trial)
        if length > eps:
            return trial / length
    raise ValueError("Failed to construct perpendicular vector.")


def _project_points(points: torch.Tensor, origin: torch.Tensor, e_x: torch.Tensor, e_y: torch.Tensor) -> torch.Tensor:
    """Project 3D coordinates into the local 2D frame spanned by (e_x, e_y) with origin."""
    diff = points - origin
    x = torch.matmul(diff, e_x)
    y = torch.matmul(diff, e_y)
    return torch.stack([x, y], dim=-1)


@dataclass
class LocalFrame2D:
    """Representation of a 2D frame embedded in 3D space."""

    origin: torch.Tensor
    e_x: torch.Tensor
    e_y: torch.Tensor
    translation: torch.Tensor

    def project(self, points: torch.Tensor) -> torch.Tensor:
        """Project 3D points into this frame and apply the stored translation shift."""
        coords = _project_points(points, self.origin, self.e_x, self.e_y)
        return coords - self.translation


def _build_bond_frame(pos_src: torch.Tensor, pos_dst: torch.Tensor) -> LocalFrame2D:
    """Construct the local 2D frame for a bonded pair centred at the midpoint."""
    vec = pos_dst - pos_src
    e_x = _normalize(vec)
    e_y = _choose_perpendicular_unit(e_x)
    origin = 0.5 * (pos_src + pos_dst)
    translation = torch.zeros(2, dtype=pos_src.dtype, device=pos_src.device)
    return LocalFrame2D(origin=origin, e_x=e_x, e_y=e_y, translation=translation)


def _build_bond_frames_vectorized(pos_src: torch.Tensor, pos_dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct local 2D frames for all bonded pairs in a vectorized manner.
    
    Args:
        pos_src: (N_edges, 3) source atom positions
        pos_dst: (N_edges, 3) destination atom positions
    
    Returns:
        origins: (N_edges, 3) frame origins (midpoints)
        e_x_batch: (N_edges, 3) x-axis unit vectors
        e_y_batch: (N_edges, 3) y-axis unit vectors
    """
    # Compute bond vectors and origins
    vecs = pos_dst - pos_src  # (N_edges, 3)
    origins = 0.5 * (pos_src + pos_dst)  # (N_edges, 3)
    
    # Normalize bond vectors to get e_x
    norms = torch.norm(vecs, dim=1, keepdim=True)  # (N_edges, 1)
    norms = torch.clamp(norms, min=1e-12)  # Prevent division by zero
    e_x_batch = vecs / norms  # (N_edges, 3)
    
    # Choose perpendicular vectors for e_y (vectorized)
    # Try [1,0,0] first, then [0,1,0], then [0,0,1]
    device = pos_src.device
    dtype = pos_src.dtype
    n_edges = pos_src.shape[0]
    
    candidate_1 = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device).expand(n_edges, 3)
    candidate_2 = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).expand(n_edges, 3)
    candidate_3 = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device).expand(n_edges, 3)
    
    # Gram-Schmidt: subtract projection onto e_x
    def make_perpendicular(candidate, e_x):
        proj = (candidate * e_x).sum(dim=1, keepdim=True)  # (N_edges, 1)
        perp = candidate - proj * e_x  # (N_edges, 3)
        return perp
    
    trial_1 = make_perpendicular(candidate_1, e_x_batch)
    lengths_1 = torch.norm(trial_1, dim=1, keepdim=True)
    
    trial_2 = make_perpendicular(candidate_2, e_x_batch)
    lengths_2 = torch.norm(trial_2, dim=1, keepdim=True)
    
    trial_3 = make_perpendicular(candidate_3, e_x_batch)
    lengths_3 = torch.norm(trial_3, dim=1, keepdim=True)
    
    # Use the trial with the largest length (most perpendicular)
    max_lengths = torch.maximum(torch.maximum(lengths_1, lengths_2), lengths_3)
    
    # Select best trial for each edge
    use_1 = (lengths_1 >= max_lengths - 1e-10).squeeze()
    use_2 = (lengths_2 >= max_lengths - 1e-10).squeeze() & ~use_1
    use_3 = ~use_1 & ~use_2
    
    e_y_batch = torch.zeros_like(e_x_batch)
    e_y_batch[use_1] = trial_1[use_1] / lengths_1[use_1]
    e_y_batch[use_2] = trial_2[use_2] / lengths_2[use_2]
    e_y_batch[use_3] = trial_3[use_3] / lengths_3[use_3]
    
    return origins, e_x_batch, e_y_batch


def _project_points_batch(points: torch.Tensor, origins: torch.Tensor, e_x: torch.Tensor, e_y: torch.Tensor) -> torch.Tensor:
    """Project 3D coordinates into local 2D frames (vectorized).
    
    Args:
        points: (N_edges, N_atoms_per_edge, 3) 3D coordinates
        origins: (N_edges, 3) frame origins
        e_x: (N_edges, 3) x-axis vectors
        e_y: (N_edges, 3) y-axis vectors
    
    Returns:
        coords_2d: (N_edges, N_atoms_per_edge, 2) 2D coordinates
    """
    diff = points - origins.unsqueeze(1)  # (N_edges, N_atoms, 3)
    x = (diff * e_x.unsqueeze(1)).sum(dim=-1)  # (N_edges, N_atoms)
    y = (diff * e_y.unsqueeze(1)).sum(dim=-1)  # (N_edges, N_atoms)
    return torch.stack([x, y], dim=-1)  # (N_edges, N_atoms, 2)


def _triangle_edge_lengths(coords: torch.Tensor) -> list[tuple[float, tuple[int, int]]]:
    """Return list of edge lengths with their vertex indices."""
    pairs = [(0, 1), (1, 2), (0, 2)]
    result = []
    for i, j in pairs:
        length = torch.norm(coords[i] - coords[j]).item()
        result.append((length, (i, j)))
    return result


def _canonicalize_triplet(
    coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, LocalFrame2D]:
    """Return ordered coordinates, atomic numbers, and the frame following the DIEP specification."""
    edge_data = _triangle_edge_lengths(coords)
    edge_data.sort(key=lambda x: x[0], reverse=True)
    _, longest_pair = edge_data[0]

    u_idx, v_idx = longest_pair
    remaining = {0, 1, 2}.difference({u_idx, v_idx})
    w_idx = remaining.pop()

    len_u_w = torch.norm(coords[u_idx] - coords[w_idx])
    len_v_w = torch.norm(coords[v_idx] - coords[w_idx])
    if len_v_w + eps < len_u_w:
        u_idx, v_idx = v_idx, u_idx
        len_u_w, len_v_w = len_v_w, len_u_w

    order = torch.tensor([u_idx, v_idx, w_idx], dtype=torch.long, device=coords.device)
    ordered_numbers = atomic_numbers[order].clone()

    pos_u = coords[u_idx]
    pos_v = coords[v_idx]
    origin = 0.5 * (pos_u + pos_v)
    vec_uv = pos_v - pos_u
    if torch.norm(vec_uv) < eps:
        raise ValueError("Degenerate triplet with overlapping atoms.")
    e_x = _normalize(vec_uv)

    pos_w = coords[w_idx]
    vec_w = pos_w - origin
    projection = torch.dot(vec_w, e_x) * e_x
    perp = vec_w - projection
    if torch.norm(perp) < eps:
        e_y = _choose_perpendicular_unit(e_x)
    else:
        e_y = perp / torch.norm(perp)
    if torch.dot(vec_w, e_y) < 0:
        e_y = -e_y

    projected_raw = _project_points(coords, origin, e_x, e_y)
    translation = projected_raw[order].mean(dim=0)
    frame = LocalFrame2D(origin=origin, e_x=e_x, e_y=e_y, translation=translation)
    canonical = projected_raw[order] - translation

    return canonical.to(diep.float_th), ordered_numbers.to(diep.float_th), frame


def _canonicalize_triplets_batch(
    coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised canonicalisation for triplets following DIEP conventions.

    Args:
        coords: Tensor of shape (n_triplets, 3, 3) containing Cartesian coordinates of the three atoms
            forming each triplet. The second axis is ordered as (neighbor_i, center, neighbor_k).
        atomic_numbers: Tensor of shape (n_triplets, 3) with the atomic numbers in the same ordering as coords.
        eps: Numerical stability tolerance.

    Returns:
        canonical_coords: Tensor of shape (n_triplets, 3, 2) containing 2D coordinates in the canonical frame.
        ordered_numbers: Tensor of shape (n_triplets, 3) containing atomic numbers reordered to match
            canonical_coords.
    """
    if coords.numel() == 0:
        canonical = torch.zeros((0, 3, 2), dtype=diep.float_th, device=coords.device)
        numbers = torch.zeros((0, 3), dtype=diep.float_th, device=coords.device)
        return canonical, numbers

    device = coords.device
    dtype = coords.dtype
    n_triplets = coords.shape[0]

    # Identify the longest edge (u, v) for each triplet
    vec01 = coords[:, 0] - coords[:, 1]
    vec12 = coords[:, 1] - coords[:, 2]
    vec02 = coords[:, 0] - coords[:, 2]
    lengths = torch.stack(
        [torch.norm(vec01, dim=1), torch.norm(vec12, dim=1), torch.norm(vec02, dim=1)],
        dim=1,
    )
    pair_indices = torch.tensor([[0, 1], [1, 2], [0, 2]], device=device)
    longest_pair = pair_indices[lengths.argmax(dim=1)]
    u_idx = longest_pair[:, 0]
    v_idx = longest_pair[:, 1]

    # Remaining index corresponds to the third atom in the triangle
    w_idx = 3 - u_idx - v_idx
    batch_range = torch.arange(n_triplets, device=device)

    # Swap u and v if required to satisfy len_v_w >= len_u_w
    pos_u = coords[batch_range, u_idx]
    pos_v = coords[batch_range, v_idx]
    pos_w = coords[batch_range, w_idx]
    len_u_w = torch.norm(pos_u - pos_w, dim=1)
    len_v_w = torch.norm(pos_v - pos_w, dim=1)
    swap_mask = len_v_w + eps < len_u_w
    u_idx, v_idx = torch.where(swap_mask, v_idx, u_idx), torch.where(swap_mask, u_idx, v_idx)
    w_idx = 3 - u_idx - v_idx

    pos_u = coords[batch_range, u_idx]
    pos_v = coords[batch_range, v_idx]
    pos_w = coords[batch_range, w_idx]

    order = torch.stack([u_idx, v_idx, w_idx], dim=1)
    ordered_numbers = torch.gather(atomic_numbers, 1, order)

    origin = 0.5 * (pos_u + pos_v)
    vec_uv = pos_v - pos_u
    bond_len = torch.norm(vec_uv, dim=1, keepdim=True).clamp_min(eps)
    e_x = vec_uv / bond_len

    vec_w = pos_w - origin
    proj = (vec_w * e_x).sum(dim=1, keepdim=True) * e_x
    perp = vec_w - proj
    perp_norm = torch.norm(perp, dim=1, keepdim=True)
    use_fallback = perp_norm.squeeze(-1) < eps

    # Fallback axes for degenerate configurations
    ref1 = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device).expand_as(e_x)
    ref2 = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).expand_as(e_x)
    cross1 = torch.cross(e_x, ref1, dim=1)
    cross1_norm = torch.norm(cross1, dim=1, keepdim=True).clamp_min(eps)
    cross2 = torch.cross(e_x, ref2, dim=1)
    cross2_norm = torch.norm(cross2, dim=1, keepdim=True).clamp_min(eps)
    e_y_fallback = torch.where(
        cross1_norm > eps,
        cross1 / cross1_norm,
        cross2 / cross2_norm,
    )
    e_y = torch.where(
        use_fallback.unsqueeze(-1),
        e_y_fallback,
        perp / perp_norm.clamp_min(eps),
    )
    orientation = (vec_w * e_y).sum(dim=1, keepdim=True)
    flip_mask = orientation < 0
    e_y = torch.where(flip_mask, -e_y, e_y)

    diff = coords - origin.unsqueeze(1)
    x = (diff * e_x.unsqueeze(1)).sum(dim=-1)
    y = (diff * e_y.unsqueeze(1)).sum(dim=-1)
    projected = torch.stack([x, y], dim=-1)

    gather_idx = order.unsqueeze(-1).expand(-1, -1, 2)
    canonical = torch.gather(projected, 1, gather_idx)
    translation = canonical.mean(dim=1, keepdim=True)
    canonical = canonical - translation

    return canonical.to(diep.float_th), ordered_numbers.to(diep.float_th)


class DIEPIntegrator(nn.Module):
    """Compute DIEP bond and triplet embeddings via electron-ion potential integration."""

    def __init__(
        self,
        grid_half_length: float = 5.0,
        base_spacing: float = 1.0,
        sigma: float = 1.0,
        mode: Literal["sum", "grid"] = "sum",
        softening_epsilon: float = 0.5,
        use_effective_charge: bool = True,
    ):
        """
        Args:
            grid_half_length (float): Half-length of the 2D integration grid
            base_spacing (float): Grid spacing
            sigma (float): Width parameter for Gaussian electron density
            mode (str): Integration mode, either "sum" (scalar) or "grid" (full grid features)
            softening_epsilon (float): Softening parameter to prevent 1/r singularities
            use_effective_charge (bool): If True, use sqrt(Z) instead of Z for better scaling
        """
        super().__init__()
        if sigma <= 0:
            raise ValueError("Sigma must be positive.")
        if base_spacing <= 0:
            raise ValueError("Grid spacing must be positive.")
        if grid_half_length <= 0:
            raise ValueError("Grid half length must be positive.")
        if softening_epsilon < 0:
            raise ValueError("Softening epsilon must be non-negative.")

        self.grid_half_length = grid_half_length
        self.base_spacing = base_spacing
        self.sigma = sigma
        self.mode = mode
        self.softening_epsilon = softening_epsilon
        self.use_effective_charge = use_effective_charge
        self._grid: DIEPGrid | None = None
        self._grid_device: torch.device | None = None

    @property
    def edge_dim(self) -> int:
        if self.mode == "grid":
            num_axis = int(round(2 * self.grid_half_length / self.base_spacing)) + 1
            return num_axis * num_axis
        return 1

    def _ensure_grid(self, device: torch.device):
        if self._grid_device == device and self._grid is not None:
            return
        self._grid = DIEPGrid(self.grid_half_length, self.base_spacing, device)
        self._grid_device = device

    def forward(  # type: ignore[override]
        self,
        g,
        l_g,
        atomic_numbers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return DIEP bond and triplet features (vectorized implementation)."""
        self._ensure_grid(g.device)
        if self._grid is None:
            raise RuntimeError("Integration grid was not initialised.")
        
        src, dst = g.edges()
        n_edges = src.shape[0]
        
        if n_edges == 0:
            # Handle empty graph
            if self.mode == "sum":
                return (torch.zeros((0, 1), dtype=diep.float_th, device=g.device),
                       torch.zeros((0, 1), dtype=diep.float_th, device=g.device))
            else:
                grid_size = self.edge_dim
                return (torch.zeros((0, grid_size), dtype=diep.float_th, device=g.device),
                       torch.zeros((0, grid_size), dtype=diep.float_th, device=g.device))
        
        z_src = atomic_numbers[src].to(diep.float_th)
        z_dst = atomic_numbers[dst].to(diep.float_th)

        pos = g.ndata["pos"].to(diep.float_th)
        
        has_pbc_offset = "pbc_offshift" in g.edata
        if has_pbc_offset:
            pbc_offshift = g.edata["pbc_offshift"].to(diep.float_th)
        else:
            pbc_offshift = torch.zeros((n_edges, 3), dtype=diep.float_th, device=g.device)

        grid = self._grid
        grid_points = grid.points.to(diep.float_th)  # (P, 2)
        n_grid_points = grid_points.shape[0]
        delta_area = grid.delta_area.to(diep.float_th)

        # Vectorized bond feature computation
        pos_src_batch = pos[src]  # (N_edges, 3)
        pos_dst_batch = pos[dst] + pbc_offshift  # (N_edges, 3)
        
        # Build frames for all edges at once
        origins, e_x_batch, e_y_batch = _build_bond_frames_vectorized(pos_src_batch, pos_dst_batch)
        
        # Stack src and dst positions for projection
        atom_positions_3d = torch.stack([pos_src_batch, pos_dst_batch], dim=1)  # (N_edges, 2, 3)
        
        # Project to 2D frames
        fragment_coords = _project_points_batch(atom_positions_3d, origins, e_x_batch, e_y_batch)  # (N_edges, 2, 2)
        
        # Compute distances to all grid points (vectorized)
        # fragment_coords: (N_edges, 2_atoms, 2_coords)
        # grid_points: (P, 2)
        # diff: (N_edges, 2_atoms, P, 2)
        diff_fragment = fragment_coords.unsqueeze(2) - grid_points.unsqueeze(0).unsqueeze(0)
        dist_sq_fragment = torch.sum(diff_fragment**2, dim=-1)  # (N_edges, 2_atoms, P)
        
        # Compute Gaussian density (vectorized)
        rho_fragment = _gaussian_density(dist_sq_fragment, self.sigma)  # (N_edges, 2, P)
        rho_total = rho_fragment.sum(dim=1)  # (N_edges, P)
        
        # Compute Coulomb potentials (vectorized)
        eps_sq = self.softening_epsilon ** 2
        denom = torch.sqrt(dist_sq_fragment + eps_sq)  # (N_edges, 2, P)
        
        # Use effective charges
        if self.use_effective_charge:
            z_eff = torch.stack([torch.sqrt(z_src), torch.sqrt(z_dst)], dim=1)  # (N_edges, 2)
        else:
            z_eff = torch.stack([z_src, z_dst], dim=1)  # (N_edges, 2)
        
        potential = (z_eff.unsqueeze(2) / denom).sum(dim=1)  # (N_edges, P)
        
        # Compute integrand
        integrand = delta_area * rho_total * potential  # (N_edges, P)
        
        # Output based on mode
        if self.mode == "sum":
            bond_feat = integrand.sum(dim=1, keepdim=True)  # (N_edges, 1)
        else:
            bond_feat = integrand  # (N_edges, P)

        # Triplet processing (vectorized over all triplets)
        lg_src, lg_dst = l_g.edges()
        bond_vec = g.edata["bond_vec"].to(diep.float_th)
        max_triplets = lg_src.shape[0]
        
        if max_triplets == 0:
            # Handle no triplets case
            if self.mode == "sum":
                triplet_feat = torch.zeros((0, 1), dtype=diep.float_th, device=g.device)
            else:
                triplet_feat = torch.zeros((0, n_grid_points), dtype=diep.float_th, device=g.device)
            return bond_feat.to(diep.float_th), triplet_feat.to(diep.float_th)

        center_idx = src[lg_src]
        neighbor_i = dst[lg_src]
        neighbor_k = dst[lg_dst]

        pos_center = pos[center_idx]
        pos_first = pos_center + bond_vec[lg_src]
        pos_second = pos_center + bond_vec[lg_dst]

        raw_coords = torch.stack([pos_first, pos_center, pos_second], dim=1)
        triplet_indices = torch.stack([neighbor_i, center_idx, neighbor_k], dim=1)
        triplet_numbers = atomic_numbers[triplet_indices]

        canonical_coords, ordered_numbers = _canonicalize_triplets_batch(
            raw_coords, triplet_numbers
        )

        fragment_coords = canonical_coords  # (T, 3, 2)
        diff_fragment = fragment_coords.unsqueeze(2) - grid_points.unsqueeze(0)
        dist_sq_fragment = torch.sum(diff_fragment**2, dim=-1)

        rho_total = _gaussian_density(dist_sq_fragment, self.sigma).sum(dim=1)

        eps_sq = self.softening_epsilon**2
        denom_triplet = torch.sqrt(dist_sq_fragment + eps_sq)

        if self.use_effective_charge:
            ordered_numbers_eff = torch.sqrt(ordered_numbers.clamp_min(0.0))
        else:
            ordered_numbers_eff = ordered_numbers

        potential_triplet = (ordered_numbers_eff.unsqueeze(-1) / denom_triplet).sum(dim=1)

        integrand = delta_area * rho_total * potential_triplet

        if self.mode == "sum":
            triplet_feat = integrand.sum(dim=1, keepdim=True)
        else:
            triplet_feat = integrand

        return bond_feat.to(diep.float_th), triplet_feat.to(diep.float_th)
