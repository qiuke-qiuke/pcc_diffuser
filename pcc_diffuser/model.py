"""Conditional U-Net for unified PCC path diffusion."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 4:
            raise ValueError("time embedding dimension must be at least four")
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        exponent = -math.log(10_000.0) * torch.arange(
            half, device=timesteps.device, dtype=torch.float32
        ) / max(half - 1, 1)
        angles = timesteps.float().unsqueeze(-1) * exponent.exp().unsqueeze(0)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.dimension:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class DirectedSphereMessageLayer(MessagePassing):
    """Update every sphere node from all directed incoming neighbours."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__(aggr="add", flow="source_to_target")
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.normalisation = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        centres: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        messages = self.propagate(
            edge_index,
            x=features,
            centres=centres,
            size=(len(features), len(features)),
        )
        update = self.update_mlp(torch.cat((features, messages), dim=-1))
        return self.normalisation(features + update)

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        centres_i: torch.Tensor,
        centres_j: torch.Tensor,
    ) -> torch.Tensor:
        relative = centres_j - centres_i
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        return self.message_mlp(torch.cat((x_i, x_j, relative, distance), dim=-1))


class SphereGraphEncoder(nn.Module):
    """Encode variable-size sphere graphs with centre-and-radius node features."""

    def __init__(self, output_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        if hidden_dim < 1 or layers < 1:
            raise ValueError("graph hidden dimension and layer count must be positive")
        self.node_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            DirectedSphereMessageLayer(hidden_dim) for _ in range(layers)
        )
        self.graph_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.no_obstacle = nn.Parameter(torch.zeros(output_dim))

    def forward(self, graph: Batch) -> torch.Tensor:
        if not isinstance(graph, Batch):
            raise TypeError("obstacle_graph must be a torch_geometric Batch")
        if graph.x.ndim != 2 or graph.x.shape[1] != 4:
            raise ValueError("obstacle graph node features must have shape [M,4]")
        graph_count = int(graph.num_graphs)
        hidden_dim = self.node_mlp[0].out_features
        if len(graph.x):
            features = self.node_mlp(graph.x)
            for layer in self.layers:
                features = layer(features, graph.x[:, :3], graph.edge_index)
        else:
            features = graph.x.new_empty((0, hidden_dim))

        counts = torch.bincount(graph.batch, minlength=graph_count).to(graph.x.dtype)
        summed = graph.x.new_zeros((graph_count, hidden_dim))
        if len(features):
            summed.index_add_(0, graph.batch, features)
        mean = summed / counts.clamp_min(1).unsqueeze(-1)

        maximum = graph.x.new_full((graph_count, hidden_dim), -torch.inf)
        if len(features):
            indices = graph.batch[:, None].expand_as(features)
            maximum.scatter_reduce_(0, indices, features, reduce="amax", include_self=True)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        encoded = self.graph_mlp(
            torch.cat((mean, maximum, torch.log1p(counts).unsqueeze(-1)), dim=-1)
        )
        empty = counts == 0
        return torch.where(empty[:, None], self.no_obstacle[None], encoded)


class FiLMResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        context_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.context = nn.Sequential(nn.SiLU(), nn.Linear(context_dim, 2 * out_channels))
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        scale, shift = self.context(context).unsqueeze(-1).chunk(2, dim=1)
        hidden = self.norm2(hidden) * (1.0 + scale) + shift
        hidden = self.conv2(self.dropout(torch.nn.functional.silu(hidden)))
        return hidden + self.residual(x)


@dataclass(frozen=True)
class ModelConfig:
    horizon: int
    transition_dim: int
    model_dim: int
    dim_mults: tuple[int, ...]
    context_dim: int
    dropout: float
    graph_hidden_dim: int
    graph_layers: int


class ConditionalTemporalUNet(nn.Module):
    """Predict diffusion noise conditioned on start, goal tip, and sphere set.

    Only the initial configuration is hard-conditioned.  The terminal frame is
    deliberately latent so distinct initial noise can choose distinct IK modes.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.transition_dim != 6:
            raise ValueError("this implementation expects a 6-D PCC configuration")
        if not config.dim_mults or any(multiplier <= 0 for multiplier in config.dim_mults):
            raise ValueError("dim_mults must contain positive values")
        downsample_factor = 2 ** (len(config.dim_mults) - 1)
        if config.horizon % downsample_factor:
            raise ValueError(f"horizon must be divisible by {downsample_factor}")
        self.config = config

        channels = [config.model_dim * multiplier for multiplier in config.dim_mults]
        context_dim = config.context_dim
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(config.model_dim),
            nn.Linear(config.model_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )
        self.start_encoder = nn.Sequential(
            nn.Linear(6, context_dim), nn.SiLU(), nn.Linear(context_dim, context_dim)
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(3, context_dim), nn.SiLU(), nn.Linear(context_dim, context_dim)
        )
        self.obstacle_encoder = SphereGraphEncoder(
            context_dim, config.graph_hidden_dim, config.graph_layers
        )
        self.context_mixer = nn.Sequential(
            nn.SiLU(), nn.Linear(context_dim, context_dim), nn.SiLU()
        )

        # The seventh input channel is the hard-conditioning mask.
        self.input_projection = nn.Conv1d(config.transition_dim + 1, channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        for level, channel in enumerate(channels):
            next_channel = channels[level + 1] if level + 1 < len(channels) else channel
            self.down_blocks.append(
                nn.ModuleList(
                    (
                        FiLMResidualBlock(channel, channel, context_dim, config.dropout),
                        FiLMResidualBlock(channel, channel, context_dim, config.dropout),
                        nn.Conv1d(channel, next_channel, 4, stride=2, padding=1)
                        if level + 1 < len(channels)
                        else nn.Identity(),
                    )
                )
            )

        deepest = channels[-1]
        self.mid_blocks = nn.ModuleList(
            (
                FiLMResidualBlock(deepest, deepest, context_dim, config.dropout),
                FiLMResidualBlock(deepest, deepest, context_dim, config.dropout),
            )
        )

        self.up_blocks = nn.ModuleList()
        for level in reversed(range(len(channels) - 1)):
            channel = channels[level]
            deeper = channels[level + 1]
            self.up_blocks.append(
                nn.ModuleList(
                    (
                        nn.ConvTranspose1d(deeper, channel, 4, stride=2, padding=1),
                        FiLMResidualBlock(2 * channel, channel, context_dim, config.dropout),
                        FiLMResidualBlock(channel, channel, context_dim, config.dropout),
                    )
                )
            )

        self.output = nn.Sequential(
            nn.GroupNorm(_group_count(channels[0]), channels[0]),
            nn.SiLU(),
            nn.Conv1d(channels[0], config.transition_dim, 3, padding=1),
        )

    def config_dict(self) -> dict[str, object]:
        result = asdict(self.config)
        result["dim_mults"] = list(self.config.dim_mults)
        return result

    def forward(
        self,
        noisy_path: torch.Tensor,
        timesteps: torch.Tensor,
        start: torch.Tensor,
        target_tip: torch.Tensor,
        obstacle_graph: Batch,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, horizon, transition_dim = noisy_path.shape
        if (horizon, transition_dim) != (self.config.horizon, self.config.transition_dim):
            raise ValueError(
                f"expected path [B,{self.config.horizon},{self.config.transition_dim}]"
            )
        if (
            timesteps.shape != (batch,)
            or start.shape != (batch, 6)
            or target_tip.shape != (batch, 3)
        ):
            raise ValueError("condition batch shapes do not match path")
        if obstacle_graph.num_graphs != batch:
            raise ValueError("obstacle graph batch size does not match path batch size")
        if condition_mask is None:
            condition_mask = noisy_path.new_zeros((batch, horizon, 1))
            condition_mask[:, 0] = 1.0
        if condition_mask.shape != (batch, horizon, 1):
            raise ValueError("condition_mask must have shape [B,H,1]")

        context = self.context_mixer(
            self.time_encoder(timesteps)
            + self.start_encoder(start)
            + self.goal_encoder(target_tip)
            + self.obstacle_encoder(obstacle_graph)
        )
        x = torch.cat((noisy_path, condition_mask.to(noisy_path.dtype)), dim=-1)
        x = self.input_projection(x.transpose(1, 2))
        skips: list[torch.Tensor] = []
        for first, second, downsample in self.down_blocks:
            x = second(first(x, context), context)
            skips.append(x)
            x = downsample(x)
        for block in self.mid_blocks:
            x = block(x, context)
        # The deepest skip represents the same resolution as the bottleneck;
        # the remaining skips pair with each upsample stage.
        skips.pop()
        for upsample, first, second in self.up_blocks:
            x = upsample(x)
            skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]:
                raise RuntimeError("temporal U-Net shape mismatch; check horizon")
            x = second(first(torch.cat((x, skip), dim=1), context), context)
        return self.output(x).transpose(1, 2)
