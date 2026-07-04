import math
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

from etflow.commons.jacobian_4d_selection import select_jacobian_4d_bonds
from etflow.commons.jacobian_4d_velocity import (
    apply_jacobian_4d_correction,
    combine_jacobian_4d_velocity,
)

from .modules import CoorsNorm, EquivariantVectorOutput
from .utils import CosineCutoff, NeighborEmbedding, act_class_mapping, rbf_class_mapping


def center(pos, batch):
    pos_center = pos - scatter(pos, batch, dim=0, reduce="mean")[batch]
    return pos_center


class EquivariantMultiHeadAttention(MessagePassing):
    def __init__(
        self,
        hidden_channels: int,
        num_rbf: int,
        distance_influence: str,
        num_heads: int,
        activation: str,
        attn_activation: str,
        cutoff_lower: float,
        cutoff_upper: float,
        node_attr_dim: int = 0,
        qk_norm: bool = False,
        norm_coors: bool = False,
        norm_coors_scale_init: float = 1e-2,
        so3_equivariant: bool = False,
    ):
        super(EquivariantMultiHeadAttention, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.so3_equivariant = so3_equivariant
        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.node_attr_dim = node_attr_dim
        self.norm_coors = norm_coors  # boolean
        self.coors_norm = (
            CoorsNorm(scale_init=norm_coors_scale_init) if norm_coors else nn.Identity()
        )
        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.qk_norm = qk_norm

        input_channels = (
            hidden_channels + 1 + (hidden_channels if node_attr_dim > 0 else 0)
        )
        self.mixing_mlp = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        if qk_norm:
            # add layer norm to q and k projections
            # based on https://arxiv.org/pdf/2302.05442.pdf
            self.q_proj = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
            )
            self.k_proj = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
            )
        else:
            self.q_proj = nn.Linear(hidden_channels, hidden_channels)
            self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(
            hidden_channels, hidden_channels * (3 + int(so3_equivariant))
        )
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)

        # projection linear layers for edge attributes
        self.dk_proj = nn.Linear(num_rbf, hidden_channels)
        self.dv_proj = nn.Linear(num_rbf, hidden_channels * (3 + int(so3_equivariant)))

        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        if self.qk_norm:
            self.q_proj[0].bias.data.fill_(0)
            nn.init.xavier_uniform_(self.q_proj[0].weight)
            self.k_proj[0].bias.data.fill_(0)
            nn.init.xavier_uniform_(self.k_proj[0].weight)
        else:
            self.q_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.q_proj.weight)
            self.k_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.k_proj.weight)

        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vec_proj.weight)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij, t, node_attr):
        # Mix x with node_attr and time
        x = self.mixing_mlp(torch.cat([x, t, node_attr], dim=1))

        # Input features: (num_atoms, hidden_channels)
        x = self.layernorm(x)
        # key/query features: (num_atoms, num_heads, head_dim)
        # where head_dim * num_heads == hidden_channels
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        # value features: (num_atoms, num_heads, 3 * head_dim)
        v = self.v_proj(x).reshape(
            -1, self.num_heads, self.head_dim * (3 + int(self.so3_equivariant))
        )

        # vec features: (num_atoms, 3, hidden_channels) (all invariant)
        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec = vec.reshape(-1, 3, self.num_heads, self.head_dim)
        vec_dot = (vec1 * vec2).sum(dim=1)

        # transform edge attributes (relative distances and user provided edge attributes)
        # into dk and dv vectors with shape (num_edges, num_heads, head_dim)
        # and (num_edges, num_heads, 3 * head_dim) respectively
        dk = self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(f_ij)).reshape(
            -1, self.num_heads, self.head_dim * (3 + int(self.so3_equivariant))
        )

        # Message Passing Propagate
        x, vec = self.propagate(
            edge_index,  # (2, edges)
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dv=dv,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        # new shape: (num_atoms, hidden_channels)
        x = x.reshape(-1, self.hidden_channels)
        # new shape: (num_atoms, 3, hidden_channels)
        vec = vec.reshape(-1, 3, self.hidden_channels)
        # normalize the vec if norm_coors is True
        vec = self.coors_norm(vec)

        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dvec = vec3 * o1.unsqueeze(1) + vec
        dx = vec_dot * o2 + o3
        return dx, dvec

    def message(
        self,
        q_i: Tensor,  # (num_edges, num_heads, head_dim)
        k_j: Tensor,  # (num_edges, num_heads, head_dim)
        v_j: Tensor,  # (num_edges, num_heads, head_dim * 3)
        vec_j: Tensor,  # (num_edges, 3, num_heads, head_dim)
        dk: Tensor,  # (num_edges, num_heads, head_dim)
        dv: Tensor,  # (num_edges, num_heads, head_dim * 3)
        r_ij: Tensor,  # (num_edges,) edge distances
        d_ij: Tensor,  # (num_edges, 3) edge vectors (unit vectors)
    ):
        # dot product attention, a score for each edge
        attn = (q_i * k_j * dk).sum(dim=-1)  # (num_edges, num_heads)

        # apply attention activation function
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        # value pathway
        v_j = v_j * dv  # multiply with edge attr features

        if self.so3_equivariant:
            x, vec1, vec2, vec3 = torch.split(v_j, self.head_dim, dim=2)
        else:
            x, vec1, vec2 = torch.split(v_j, self.head_dim, dim=2)
            vec3 = None

        # update scalar features
        x = x * attn.unsqueeze(2)  # (num_edges, num_heads, head_dim)
        # update vector features (num_edges, 3, num_heads, head_dim)
        if self.so3_equivariant:
            vec = (
                vec_j * vec1.unsqueeze(1)
                + vec2.unsqueeze(1) * d_ij.unsqueeze(2).unsqueeze(3)
                + vec3.unsqueeze(1)
                * torch.cross(d_ij.unsqueeze(2).unsqueeze(3), vec_j, dim=1)
            )
        else:
            vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * d_ij.unsqueeze(
                2
            ).unsqueeze(3)
        return x, vec

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        # scatter edge-level features (for x and vec) to node-level
        # x shape: (num_atoms, num_heads, head_dim)
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        # vec shape: (num_atoms, 3, num_heads, head_dim)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class TorchMD_ET_dynamics(nn.Module):
    r"""The TorchMD equivariant Transformer architecture.

    Parameters
    ----------
    hidden_channels (int, optional): Hidden embedding size.
        (default: :obj:`128`)
    num_layers (int, optional): The number of attention layers.
        (default: :obj:`6`)
    num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
        (default: :obj:`50`)
    rbf_type (string, optional): The type of radial basis function to use.
        (default: :obj:`"expnorm"`)
    trainable_rbf (bool, optional): Whether to train RBF parameters with
        backpropagation. (default: :obj:`True`)
    activation (string, optional): The type of activation function to use.
        (default: :obj:`"silu"`)
    attn_activation (string, optional): The type of activation function to use
        inside the attention mechanism. (default: :obj:`"silu"`)
    neighbor_embedding (bool, optional): Whether to perform an initial neighbor
        embedding step. (default: :obj:`True`)
    num_heads (int, optional): Number of attention heads.
        (default: :obj:`8`)
    distance_influence (string, optional): Where distance information is used inside
        the attention mechanism. (default: :obj:`"both"`)
    cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions.
        (default: :obj:`0.0`)
    cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions.
        (default: :obj:`5.0`)
    max_z (int, optional): Maximum atomic number. Used for initializing embeddings.
        (default: :obj:`100`)
    qk_norm (bool, optional):
        Applies layer norm to q and k projections. Supposed to
        stabilize the training based on
        https://arxiv.org/pdf/2302.05442.pdf. (default: :obj:`False`)
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 6,
        num_rbf: int = 50,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = True,
        activation: str = "silu",
        attn_activation: str = "silu",
        neighbor_embedding: bool = True,
        num_heads: int = 8,
        distance_influence: str = "both",
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 10.0,
        max_z: int = 100,
        node_attr_dim: int = 0,
        edge_attr_dim: int = 0,
        qk_norm: bool = False,
        norm_coors: bool = False,
        norm_coors_scale_init: float = 1e-2,
        clip_during_norm: bool = False,
        so3_equivariant: bool = False,
    ):
        super(TorchMD_ET_dynamics, self).__init__()

        assert distance_influence in ["keys", "values", "both", "none"]
        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )
        assert attn_activation in act_class_mapping, (
            f'Unknown attention activation function "{attn_activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.neighbor_embedding = neighbor_embedding
        self.num_heads = num_heads
        self.distance_influence = distance_influence
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_z = max_z
        self.node_attr_dim = node_attr_dim
        self.edge_attr_dim = edge_attr_dim
        self.clip_during_norm = clip_during_norm

        act_class = act_class_mapping[activation]

        self.embedding = nn.Embedding(self.max_z, self.hidden_channels)

        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.neighbor_embedding = (
            NeighborEmbedding(
                hidden_channels,
                num_rbf + edge_attr_dim,
                cutoff_lower,
                cutoff_upper,
                self.max_z,
            )
            if neighbor_embedding
            else None
        )

        if self.node_attr_dim > 0:
            self.node_mlp = nn.Sequential(
                nn.Linear(node_attr_dim, hidden_channels),
                act_class(),
                nn.LayerNorm(hidden_channels),
                nn.Linear(hidden_channels, hidden_channels),
            )

        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = EquivariantMultiHeadAttention(
                hidden_channels,
                num_rbf + edge_attr_dim,
                distance_influence,
                num_heads,
                act_class,
                attn_activation,
                cutoff_lower,
                cutoff_upper,
                node_attr_dim=node_attr_dim,
                qk_norm=qk_norm,
                norm_coors=norm_coors,
                norm_coors_scale_init=norm_coors_scale_init,
                so3_equivariant=so3_equivariant,
            )  # .jittable() TODO: Removing for now
            self.attention_layers.append(layer)

        self.out_norm = nn.LayerNorm(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.out_norm.reset_parameters()

    def forward(
        self,
        z: Tensor,
        t: Tensor,
        pos: Tensor,
        batch: Tensor,
        edge_index: Optional[Tensor] = None,
        node_attr: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # embed atomic numbers using an embedding layer
        if z.dim() > 1:
            z = z.squeeze()  # (num_atoms,)
        x = self.embedding(z)  # (num_atoms, hidden_channels)

        # append time to node features
        if self.node_attr_dim > 0:
            node_attr = self.node_mlp(node_attr)
        else:
            node_attr = None

        # compute distances
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        edge_weight = (edge_vec**2).sum(dim=-1, keepdim=False)

        # update edge_attributes with user input if they are given
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(1)  # (num_edges, 1)
            # (num_edges, num_rbf + edge_attr_dim)
            edge_attr = torch.cat(
                [self.distance_expansion(edge_weight), edge_attr], dim=-1
            )
        else:
            edge_attr = self.distance_expansion(edge_weight)

        mask = edge_index[0] == edge_index[1]
        masked_edge_weight = edge_weight.masked_fill(mask, 1).unsqueeze(1)

        if self.clip_during_norm:
            # clip edge_weight to avoid exploding values if two nodes are close
            masked_edge_weight = masked_edge_weight.clamp(min=1.0e-2)

        edge_vec = edge_vec / masked_edge_weight

        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)

        # vec here is invariant values, we are not modifying the vectors.
        # (num_atoms, 3, hidden_channels)
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)
        for attn in self.attention_layers:
            dx, dvec = attn(
                x,
                vec,
                edge_index,
                edge_weight,
                edge_attr,
                edge_vec,
                node_attr=node_attr,
                t=t,
            )
            x = x + dx
            vec = vec + dvec
        x = self.out_norm(x)  # apply layer norm in the end.

        return x, vec, z, pos, batch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"rbf_type={self.rbf_type}, "
            f"trainable_rbf={self.trainable_rbf}, "
            f"activation={self.activation}, "
            f"attn_activation={self.attn_activation}, "
            f"neighbor_embedding={self.neighbor_embedding}, "
            f"num_heads={self.num_heads}, "
            f"distance_influence={self.distance_influence}, "
            f"cutoff_lower={self.cutoff_lower}, "
            f"cutoff_upper={self.cutoff_upper})"
        )


class TorchMDDynamics(nn.Module):
    """
    TorchMDDynamics Model for DDPM training.

    Parameters
    ----------
    hidden_channels (int, optional):
        Hidden embedding size. (default: :obj:`128`)
    num_layers (int, optional):
        The number of attention layers. (default: :obj:`8`)
    num_rbf (int, optional):
        The number of radial basis functions :math:`\mu`.
        (default: :obj:`64`)
    rbf_type (string, optional):
        The type of radial basis function to use.
        (default: :obj:`"expnorm"`)
    trainable_rbf (bool, optional):
        Whether to train RBF parameters with backpropagation.
        (default: :obj:`False`)
    activation (string, optional):
        The type of activation function to use. (default: :obj:`"silu"`)
    neighbor_embedding (bool, optional):
        Whether to perform an initial neighbor embedding step.
        (default: :obj:`True`)
    cutoff_lower (float, optional):
        Lower cutoff distance for interatomic interactions.
        (default: :obj:`0.0`)
    cutoff_upper (float, optional):
        Upper cutoff distance for interatomic interactions.
        (default: :obj:`5.0`)
    max_z (int, optional):
        Maximum atomic number. Used for initializing embeddings.
        (default: :obj:`100`)
    node_attr_dim (int, optional):
        Dimension of additional input node  features (non-atomic numbers).
    attn_activation (string, optional):
        The type of activation function to use inside the attention
        mechanism. (default: :obj:`"silu"`)
    num_heads (int, optional):
        Number of attention heads. (default: :obj:`8`)
    distance_influence (string, optional):
        Where distance information is used inside the attention
        mechanism. (default: :obj:`"both"`)
    qk_norm (bool, optional):
        Applies layer norm to q and k projections. Supposed to
        stabilize the training based on
        https://arxiv.org/pdf/2302.05442.pdf. (default: :obj:`False`)
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 8,
        num_rbf: int = 64,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = False,
        activation: str = "silu",
        neighbor_embedding: int = True,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 10.0,
        max_z: int = 100,
        node_attr_dim: int = 0,
        edge_attr_dim: int = 0,
        attn_activation: str = "silu",
        num_heads: int = 8,
        distance_influence: str = "both",
        reduce_op: str = "sum",
        qk_norm: bool = False,
        output_layer_norm: bool = True,
        clip_during_norm: bool = False,
        so3_equivariant: bool = False,
        use_angular_head: bool = False,
        angular_mu: float = 1.0,
        angular_head_hidden_channels: Optional[int] = None,
        angular_mu_schedule: str = "constant",
        angular_mu_max: float = 0.3,
        angular_mu_sigmoid_k: float = 10.0,
        angular_mu_sigmoid_t0: float = 0.5,
        use_jacobian_4d_correction: bool = False,
        jacobian_4d_min_affected_atoms: int = 2,
        jacobian_4d_max_bonds_per_mol: int = 16,
    ):
        super().__init__()
        self.representation_model = TorchMD_ET_dynamics(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_type=rbf_type,
            trainable_rbf=trainable_rbf,
            activation=activation,
            neighbor_embedding=neighbor_embedding,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_z=max_z,
            attn_activation=attn_activation,
            num_heads=num_heads,
            distance_influence=distance_influence,
            node_attr_dim=node_attr_dim,
            edge_attr_dim=edge_attr_dim,
            qk_norm=qk_norm,
            clip_during_norm=clip_during_norm,
            so3_equivariant=so3_equivariant,
        )
        self.output_model = EquivariantVectorOutput(
            hidden_channels=hidden_channels,
            activation=activation,
            reduce_op=reduce_op,
            layer_norm=output_layer_norm,
        )
        if use_angular_head and not so3_equivariant:
            raise ValueError(
                "Bond-angular velocity is only supported with so3_equivariant=True."
            )
        self.use_angular_head = use_angular_head
        self.angular_mu = float(angular_mu)
        self.angular_mu_schedule = str(angular_mu_schedule).lower()
        self.angular_mu_max = float(angular_mu_max)
        self.angular_mu_sigmoid_k = float(angular_mu_sigmoid_k)
        self.angular_mu_sigmoid_t0 = float(angular_mu_sigmoid_t0)
        valid_angular_mu_schedules = {"constant", "quadratic", "sigmoid"}
        if self.angular_mu_schedule not in valid_angular_mu_schedules:
            raise ValueError(
                "angular_mu_schedule must be one of "
                f"{sorted(valid_angular_mu_schedules)}, got {angular_mu_schedule}."
            )
        if not math.isfinite(self.angular_mu_max) or self.angular_mu_max < 0:
            raise ValueError(
                "angular_mu_max must be finite and non-negative, got "
                f"{angular_mu_max}."
            )
        if not math.isfinite(self.angular_mu_sigmoid_k):
            raise ValueError(
                f"angular_mu_sigmoid_k must be finite, got {angular_mu_sigmoid_k}."
            )
        if not math.isfinite(self.angular_mu_sigmoid_t0):
            raise ValueError(
                f"angular_mu_sigmoid_t0 must be finite, got {angular_mu_sigmoid_t0}."
            )
        self.last_angular_stats = {}
        if use_angular_head:
            angular_hidden = angular_head_hidden_channels or hidden_channels
            act_class = act_class_mapping[activation]
            self.bond_angular_head = nn.Sequential(
                nn.Linear(2 * hidden_channels, angular_hidden),
                act_class(),
                nn.Linear(angular_hidden, 1),
            )
        else:
            self.bond_angular_head = None
        self.use_jacobian_4d_correction = bool(use_jacobian_4d_correction)
        self.jacobian_4d_min_affected_atoms = int(
            jacobian_4d_min_affected_atoms
        )
        self.jacobian_4d_max_bonds_per_mol = int(
            jacobian_4d_max_bonds_per_mol
        )
        if self.use_jacobian_4d_correction:
            jacobian_hidden = hidden_channels
            act_class = act_class_mapping[activation]
            self.jacobian_4d_head = nn.Sequential(
                nn.Linear(3 * hidden_channels, jacobian_hidden),
                act_class(),
                nn.Linear(jacobian_hidden, 4),
            )
        else:
            self.jacobian_4d_head = None
        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.bond_angular_head is not None:
            nn.init.xavier_uniform_(self.bond_angular_head[0].weight)
            self.bond_angular_head[0].bias.data.zero_()
            # Start exactly from the pretrained residual velocity field.
            self.bond_angular_head[-1].weight.data.zero_()
            self.bond_angular_head[-1].bias.data.zero_()
        if self.jacobian_4d_head is not None:
            nn.init.xavier_uniform_(self.jacobian_4d_head[0].weight)
            self.jacobian_4d_head[0].bias.data.zero_()
            # The prototype is a residual correction and must start as a no-op.
            self.jacobian_4d_head[-1].weight.data.zero_()
            self.jacobian_4d_head[-1].bias.data.zero_()

    def _get_angular_mu_t(
        self,
        t: Tensor,
        batch: Optional[Tensor],
        v_ang: Tensor,
    ) -> Tensor:
        """Return an atom-level angular scale broadcastable to ``v_ang``."""

        num_atoms = v_ang.size(0)
        if self.angular_mu_schedule == "constant":
            # The constant path is not capped by angular_mu_max so old angular_mu
            # values remain exactly backward compatible.
            angular_mu = torch.as_tensor(
                self.angular_mu,
                device=v_ang.device,
                dtype=v_ang.dtype,
            )
            return angular_mu.expand(num_atoms, 1)

        if not torch.is_tensor(t):
            raise TypeError(
                "t must be a tensor for a time-dependent angular_mu schedule, "
                f"got {type(t).__name__}."
            )

        flat_t = t.to(device=v_ang.device, dtype=v_ang.dtype).reshape(-1)
        if flat_t.numel() == 1:
            atom_t = flat_t.expand(num_atoms)
        elif flat_t.numel() == num_atoms:
            atom_t = flat_t
        elif batch is not None:
            atom_batch = batch.to(device=v_ang.device, dtype=torch.long).reshape(-1)
            if atom_batch.numel() != num_atoms:
                raise ValueError(
                    "batch must contain one graph index per atom, got "
                    f"{atom_batch.numel()} indices for {num_atoms} atoms."
                )
            num_graphs = int(atom_batch.max().item()) + 1 if atom_batch.numel() else 0
            if flat_t.numel() != num_graphs:
                raise ValueError(
                    "Could not map t to atoms: expected a scalar, one value per "
                    f"atom ({num_atoms}), or one value per graph ({num_graphs}), "
                    f"got shape {tuple(t.shape)}."
                )
            atom_t = flat_t[atom_batch]
        else:
            raise ValueError(
                "Could not map t to atoms without batch: expected a scalar or "
                f"{num_atoms} values, got shape {tuple(t.shape)}."
            )

        atom_t = atom_t.reshape(num_atoms, 1)
        return self._apply_angular_mu_schedule(atom_t, v_ang)

    def _apply_angular_mu_schedule(self, atom_t: Tensor, v_ang: Tensor) -> Tensor:
        angular_mu_max = torch.as_tensor(
            self.angular_mu_max,
            device=v_ang.device,
            dtype=v_ang.dtype,
        )
        if self.angular_mu_schedule == "quadratic":
            angular_mu_t = angular_mu_max * atom_t.square()
        else:
            sigmoid_k = torch.as_tensor(
                self.angular_mu_sigmoid_k,
                device=v_ang.device,
                dtype=v_ang.dtype,
            )
            sigmoid_t0 = torch.as_tensor(
                self.angular_mu_sigmoid_t0,
                device=v_ang.device,
                dtype=v_ang.dtype,
            )
            angular_mu_t = angular_mu_max * torch.sigmoid(
                sigmoid_k * (atom_t - sigmoid_t0)
            )

        zero = torch.zeros((), device=v_ang.device, dtype=v_ang.dtype)
        return torch.minimum(
            torch.maximum(angular_mu_t, zero),
            angular_mu_max,
        )

    def forward(
        self,
        z: Tensor,
        t: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_attr: Optional[Tensor] = None,
        rotatable_bond_index: Optional[Tensor] = None,
        atom_bond_influence_index: Optional[Tensor] = None,
        jacobian_4d_correction_scale=0.0,
        jacobian_4d_warmup_scale=1.0,
        return_aux: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """Forward pass over torchmd-net model.

        Parameters
        ----------
        z: torch.Tensor
            Atomic numbers, shape (num_atoms,)
        t: torch.Tensor
            Time steps of diffusion, shape (num_atoms,)
        pos: torch.Tensor
            Atomic positions, shape (num_atoms, 3)
        edge_index: torch.Tensor
            Edge index, shape (2, num_edges)
        batch: torch.Tensor, optional
            Batch vector representing which atoms belong to which molecule,
            shape (num_atoms,). If not given, all atoms are assumed to belong
            to the same molecule.
        edge_attr: torch.Tensor, optional
            Edge attributes, shape (num_edges, edge_attr_dim)
        node_attr: torch.Tensor, optional
            Node attributes, shape (num_atoms, node_attr_dim)
        """
        # run the potentially wrapped representation model
        x, v, z, pos, batch = self.representation_model(
            z=z,
            t=t,
            pos=pos,
            batch=batch,
            node_attr=node_attr,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # Original atom velocity head: this remains the residual velocity branch.
        _, v = self.output_model.pre_reduce(x, v, z, pos, batch)
        v_res = center(v - pos, batch)

        aux: Dict[str, Tensor] = {}
        v_atom = v_res
        needs_bond_tensors = (
            self.use_angular_head or self.use_jacobian_4d_correction
        )
        if needs_bond_tensors:
            if rotatable_bond_index is None or atom_bond_influence_index is None:
                raise ValueError(
                    "The angular/Jacobian branch requires rotatable_bond_index "
                    "and atom_bond_influence_index."
                )
            if (
                rotatable_bond_index.dim() != 2
                or rotatable_bond_index.size(0) != 2
            ):
                raise ValueError(
                    "rotatable_bond_index must have shape [2, B], got "
                    f"{tuple(rotatable_bond_index.shape)}."
                )
            if (
                atom_bond_influence_index.dim() != 2
                or atom_bond_influence_index.size(0) != 2
            ):
                raise ValueError(
                    "atom_bond_influence_index must have shape [2, K], got "
                    f"{tuple(atom_bond_influence_index.shape)}."
                )

        if self.use_angular_head:
            num_atoms = pos.size(0)
            num_rotatable_bonds = rotatable_bond_index.size(1)
            v_ang = torch.zeros_like(v_res)
            if num_rotatable_bonds == 0:
                if atom_bond_influence_index.size(1) != 0:
                    raise ValueError(
                        "atom_bond_influence_index must be empty when there are "
                        "no rotatable bonds."
                    )
                dot_tau = pos.new_empty((0,))
            else:
                fixed_atom = rotatable_bond_index[0]
                rotating_atom = rotatable_bond_index[1]
                if (
                    rotatable_bond_index.min() < 0
                    or rotatable_bond_index.max() >= num_atoms
                ):
                    raise IndexError(
                        "rotatable_bond_index contains an invalid atom index."
                    )

                bond_hidden = torch.cat(
                    [x[fixed_atom], x[rotating_atom]], dim=-1
                )
                dot_tau = self.bond_angular_head(bond_hidden).squeeze(-1)

                bond_vector = pos[rotating_atom] - pos[fixed_atom]
                bond_axis = bond_vector / torch.linalg.norm(
                    bond_vector,
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1.0e-8)
                bond_center = 0.5 * (pos[fixed_atom] + pos[rotating_atom])
                omega = dot_tau.unsqueeze(-1) * bond_axis

                if atom_bond_influence_index.size(1) > 0:
                    atom_index = atom_bond_influence_index[0]
                    bond_index = atom_bond_influence_index[1]
                    if atom_index.min() < 0 or atom_index.max() >= num_atoms:
                        raise IndexError(
                            "atom_bond_influence_index contains an invalid atom "
                            "index."
                        )
                    if bond_index.min() < 0 or bond_index.max() >= num_rotatable_bonds:
                        raise IndexError(
                            "atom_bond_influence_index contains an invalid bond "
                            "index."
                        )

                    lever = pos[atom_index] - bond_center[bond_index]
                    contribution = torch.cross(
                        omega[bond_index], lever, dim=-1
                    )
                    v_ang = scatter(
                        contribution,
                        atom_index,
                        dim=0,
                        dim_size=num_atoms,
                        reduce="sum",
                    )

            angular_mu_t = self._get_angular_mu_t(
                t=t, batch=batch, v_ang=v_ang
            )
            scaled_v_ang = angular_mu_t * v_ang
            v_atom = center(v_res + scaled_v_ang, batch)
            v_res_rms = v_res.detach().pow(2).mean().sqrt()
            scaled_v_ang_rms = scaled_v_ang.detach().pow(2).mean().sqrt()
            self.last_angular_stats = {
                "mean_num_rotatable_bonds": pos.new_tensor(
                    float(num_rotatable_bonds)
                    / float(int(batch.max().item()) + 1 if batch.numel() else 1)
                ),
                "mean_abs_dot_tau": (
                    dot_tau.detach().abs().mean()
                    if dot_tau.numel() > 0
                    else pos.new_zeros(())
                ),
                "v_res_rms": v_res_rms,
                "v_ang_rms": v_ang.detach().pow(2).mean().sqrt(),
                "scaled_v_ang_rms": scaled_v_ang_rms,
                "scaled_angular_to_res_ratio": scaled_v_ang_rms
                / v_res_rms.clamp_min(1.0e-8),
                "mu_t_mean": (
                    angular_mu_t.detach().mean()
                    if angular_mu_t.numel() > 0
                    else pos.new_zeros(())
                ),
                "mu_t_max": (
                    angular_mu_t.detach().max()
                    if angular_mu_t.numel() > 0
                    else pos.new_zeros(())
                ),
                "mu_t_min": (
                    angular_mu_t.detach().min()
                    if angular_mu_t.numel() > 0
                    else pos.new_zeros(())
                ),
            }
            aux.update(
                {
                    "dot_tau_pred": dot_tau,
                    "rotatable_bond_index": rotatable_bond_index,
                    "atom_bond_influence_index": atom_bond_influence_index,
                    "pos": pos,
                }
            )
        else:
            self.last_angular_stats = {}

        if not self.use_jacobian_4d_correction:
            if return_aux:
                return v_atom, aux
            return v_atom

        selection = select_jacobian_4d_bonds(
            rotatable_bond_index=rotatable_bond_index,
            atom_bond_influence_index=atom_bond_influence_index,
            batch=batch,
            min_affected_atoms=self.jacobian_4d_min_affected_atoms,
            max_bonds_per_mol=self.jacobian_4d_max_bonds_per_mol,
        )
        anchor_index = selection["anchor_index"]
        moving_index = selection["moving_index"]
        bond_feature = torch.cat(
            [
                x[anchor_index],
                x[moving_index],
                x[moving_index] - x[anchor_index],
            ],
            dim=-1,
        )
        q_pred = self.jacobian_4d_head(bond_feature)
        v_corr, contribution_count, geometry_valid = (
            apply_jacobian_4d_correction(
                pos=pos,
                q_pred=q_pred,
                anchor_index=anchor_index,
                moving_index=moving_index,
                affected_atom_index=selection["affected_atom_index"],
                affected_bond_index=selection["affected_bond_index"],
            )
        )
        v_final, scaled_v_corr = combine_jacobian_4d_velocity(
            v_atom=v_atom,
            v_corr=v_corr,
            correction_scale=jacobian_4d_correction_scale,
            warmup_scale=jacobian_4d_warmup_scale,
        )
        aux.update(selection)
        aux.update(
            {
                "pos": pos,
                "v_atom": v_atom,
                "v_corr": v_corr,
                "scaled_v_corr": scaled_v_corr,
                "q_pred": q_pred,
                "contribution_count": contribution_count,
                "geometry_valid": geometry_valid,
            }
        )
        if return_aux:
            return v_final, aux
        return v_final
