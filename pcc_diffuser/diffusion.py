"""DDPM training objective and correctly respaced DDIM path sampler."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .kinematics import (
    backbone_point_jacobians,
    backbone_points,
    tip_position_and_jacobian,
)


def cosine_beta_schedule(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    """Nichol--Dhariwal cosine schedule in float64 before final casting."""
    if timesteps < 2:
        raise ValueError("training diffusion needs at least two timesteps")
    steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    cumulative = torch.cos(((steps / timesteps) + offset) / (1 + offset) * math.pi / 2).square()
    cumulative = cumulative / cumulative[0]
    betas = 1.0 - cumulative[1:] / cumulative[:-1]
    return betas.clamp(1e-8, 0.999).float()


def ddim_timesteps(train_timesteps: int, sample_steps: int) -> torch.Tensor:
    """Return unique descending DDIM indices spanning the trained schedule."""
    if sample_steps < 1 or sample_steps > train_timesteps:
        raise ValueError("sample_steps must be in [1, train_timesteps]")
    if sample_steps == 1:
        return torch.tensor([train_timesteps - 1], dtype=torch.long)
    ascending = torch.linspace(0, train_timesteps - 1, sample_steps).round().long()
    if torch.unique(ascending).numel() != sample_steps:
        raise RuntimeError("DDIM timestep rounding produced duplicate indices")
    return ascending.flip(0)


def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    result = values.gather(0, timesteps)
    return result.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


def terminal_fk_energy_and_gradient(
    terminal_configuration: torch.Tensor,
    target_tip: torch.Tensor,
    section_lengths: tuple[float, float, float],
    weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return terminal tip energy and its analytical configuration gradient."""
    if terminal_configuration.ndim != 2 or terminal_configuration.shape[-1] != 6:
        raise ValueError("terminal_configuration must have shape [B,6]")
    if target_tip.shape != (terminal_configuration.shape[0], 3):
        raise ValueError("target_tip must have shape [B,3]")
    if weight < 0:
        raise ValueError("weight must be non-negative")
    current_tip, jacobian = tip_position_and_jacobian(
        terminal_configuration, section_lengths
    )
    residual = current_tip - target_tip
    energy = 0.5 * float(weight) * residual.square().sum(dim=-1)
    gradient = float(weight) * (
        jacobian.transpose(-1, -2) @ residual.unsqueeze(-1)
    ).squeeze(-1)
    return energy, gradient


def _obstacle_repulsion_directions(
    configurations: torch.Tensor,
    obstacle_centres: torch.Tensor,
    obstacle_mask: torch.Tensor,
    section_lengths: tuple[float, float, float],
    safe_radius: float,
    points_per_section: int = 3,
) -> torch.Tensor:
    """Return deepest-point Jacobian-transpose repulsion directions."""
    if configurations.ndim != 3 or configurations.shape[-1] != 6:
        raise ValueError("configurations must have shape [B,H,6]")
    if obstacle_centres.ndim != 3 or obstacle_centres.shape[:1] != configurations.shape[:1]:
        raise ValueError("obstacle_centres must have shape [B,K,3]")
    if obstacle_centres.shape[-1] != 3:
        raise ValueError("obstacle_centres must have shape [B,K,3]")
    if obstacle_mask.shape != obstacle_centres.shape[:2] or obstacle_mask.dtype != torch.bool:
        raise ValueError("obstacle_mask must be boolean with shape [B,K]")
    if safe_radius <= 0:
        raise ValueError("safe_radius must be positive")
    if obstacle_centres.shape[1] == 0:
        return torch.zeros_like(configurations)

    has_obstacles = obstacle_mask.any(dim=1)
    selected_configurations = configurations[has_obstacles]
    selected_centres = obstacle_centres[has_obstacles]
    selected_mask = obstacle_mask[has_obstacles]
    batch_size, horizon = selected_configurations.shape[:2]
    sampled = backbone_points(
        selected_configurations.reshape(-1, 6),
        section_lengths,
        points_per_section,
    )[:, 1:].reshape(batch_size, horizon, -1, 3)
    offsets = sampled.unsqueeze(-2) - selected_centres[:, None, None]
    distances = torch.linalg.vector_norm(offsets, dim=-1)
    penetration = float(safe_radius) - distances
    penetration = penetration.masked_fill(
        ~selected_mask[:, None, None], -torch.inf
    )
    obstacle_count = obstacle_centres.shape[1]
    deepest, flat_indices = penetration.flatten(start_dim=2).max(dim=-1)
    point_indices = torch.div(flat_indices, obstacle_count, rounding_mode="floor")
    obstacle_indices = flat_indices.remainder(obstacle_count)
    batch = torch.arange(batch_size, device=configurations.device)[:, None]
    frames = torch.arange(horizon, device=configurations.device)[None]
    selected_offsets = offsets[batch, frames, point_indices, obstacle_indices]
    active = deepest > 0
    if not bool(active.any()):
        return torch.zeros_like(configurations)

    jacobians = backbone_point_jacobians(
        selected_configurations[active],
        point_indices[active],
        section_lengths,
        points_per_section,
    )
    active_offsets = selected_offsets[active]
    xy = active_offsets[..., :2]
    xy_distance = torch.linalg.vector_norm(xy, dim=-1)
    z_offset = active_offsets[..., 2]
    required_xy = torch.sqrt(
        torch.clamp(float(safe_radius) ** 2 - z_offset.square(), min=0.0)
    )
    xy_error = (
        torch.clamp(required_xy - xy_distance, min=0.0)
        / xy_distance.clamp_min(1e-12)
    )[..., None] * xy
    vertical = xy_distance <= 1e-12
    z_direction = torch.where(z_offset < 0, -1.0, 1.0)
    z_error = torch.where(
        vertical,
        z_direction * torch.clamp(float(safe_radius) - z_offset.abs(), min=0.0),
        torch.zeros_like(z_offset),
    )
    position_error = torch.cat((xy_error, z_error[..., None]), dim=-1)
    active_updates = (
        jacobians.transpose(-1, -2) @ position_error.unsqueeze(-1)
    ).squeeze(-1)
    selected_directions = torch.zeros_like(selected_configurations)
    selected_directions[active] = active_updates
    selected_directions[:, 0] = 0
    directions = torch.zeros_like(configurations)
    directions[has_obstacles] = selected_directions
    return directions


@dataclass(frozen=True)
class DiffusionConfig:
    train_timesteps: int = 1000
    schedule: str = "cosine"
    clip_denoised: bool = True


class GaussianPathDiffusion(nn.Module):
    """Epsilon-prediction diffusion with start-only hard inpainting."""

    def __init__(self, model: nn.Module, config: DiffusionConfig = DiffusionConfig()) -> None:
        super().__init__()
        if config.schedule != "cosine":
            raise ValueError("only the cosine schedule is currently supported")
        self.model = model
        self.config = config
        betas = cosine_beta_schedule(config.train_timesteps)
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", cumulative)
        self.register_buffer("sqrt_alphas_cumprod", cumulative.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - cumulative).sqrt())

    @property
    def horizon(self) -> int:
        return int(self.model.config.horizon)

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    @staticmethod
    def condition_mask(path: torch.Tensor) -> torch.Tensor:
        mask = path.new_zeros((*path.shape[:2], 1))
        mask[:, 0] = 1.0
        return mask

    @staticmethod
    def inpaint_start(path: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        if path.ndim != 3 or path.shape[-1] != 6:
            raise ValueError("path must have shape [B,H,6]")
        if start.shape != (path.shape[0], 6):
            raise ValueError("start must have shape [B,6]")
        result = path.clone()
        result[:, 0] = start
        return result

    def q_sample(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
        start: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(clean)
        if noise.shape != clean.shape:
            raise ValueError("noise and clean path shapes differ")
        if timesteps.shape != (clean.shape[0],):
            raise ValueError("timesteps must have shape [B]")
        noisy = (
            _extract(self.sqrt_alphas_cumprod, timesteps, clean.shape) * clean
            + _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean.shape) * noise
        )
        return self.inpaint_start(noisy, clean[:, 0] if start is None else start)

    def predict_clean(
        self, noisy: torch.Tensor, timesteps: torch.Tensor, predicted_noise: torch.Tensor
    ) -> torch.Tensor:
        alpha_bar = _extract(self.alphas_cumprod, timesteps, noisy.shape)
        return (noisy - (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()

    def training_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        clean = batch["path"]
        start = batch["start"]
        batch_size = clean.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                0, self.config.train_timesteps, (batch_size,), device=clean.device
            )
        if noise is None:
            noise = torch.randn_like(clean)
        noisy = self.q_sample(clean, timesteps, noise=noise, start=start)
        mask = self.condition_mask(clean)
        predicted = self.model(
            noisy,
            timesteps,
            start,
            batch["target_tip"],
            batch["obstacle_graph"],
            mask,
        )
        # Frame zero is a known boundary, not a denoising target.  Excluding it
        # also avoids diluting the loss with an always-restored clean value.
        free = 1.0 - mask
        squared = (predicted - noise).square() * free
        denominator = free.sum() * clean.shape[-1]
        loss = squared.sum() / denominator.clamp_min(1.0)
        return loss, {
            "diffusion_loss": loss.detach(),
            "mean_t": timesteps.float().mean().detach(),
        }

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.training_loss(batch)

    def apply_analytical_path_correction(
        self,
        path: torch.Tensor,
        start: torch.Tensor,
        target_tip: torch.Tensor,
        normaliser: Any,
        section_lengths: tuple[float, float, float],
        tip_weight: float,
        obstacle_centres: torch.Tensor | None,
        obstacle_mask: torch.Tensor | None,
        repulsion_weight: float,
        repulsion_safe_radius: float,
    ) -> torch.Tensor:
        """Apply terminal-tip and backbone-repulsion configuration updates."""
        if path.ndim != 3 or path.shape[-1] != 6:
            raise ValueError("path must have shape [B,H,6]")
        if start.shape != (path.shape[0], 6):
            raise ValueError("start must have shape [B,6]")
        if target_tip.shape != (path.shape[0], 3):
            raise ValueError("target_tip must have shape [B,3]")
        if tip_weight < 0 or repulsion_weight < 0:
            raise ValueError("analytical correction weights must be non-negative")
        if tip_weight == 0 and repulsion_weight == 0:
            return path
        result = path.detach().clone()
        physical = normaliser.denormalise_q(result)
        target_physical = normaliser.denormalise_workspace(target_tip)
        with torch.no_grad():
            update = torch.zeros_like(physical)
            if tip_weight != 0:
                _, gradient = terminal_fk_energy_and_gradient(
                    physical[:, -1],
                    target_physical,
                    section_lengths=section_lengths,
                )
                update[:, -1] -= float(tip_weight) * gradient
            if repulsion_weight != 0:
                if obstacle_centres is None or obstacle_mask is None:
                    raise ValueError("obstacle repulsion requires centres and mask")
                update += float(repulsion_weight) * _obstacle_repulsion_directions(
                    physical,
                    obstacle_centres,
                    obstacle_mask,
                    section_lengths,
                    repulsion_safe_radius,
                )
            result = normaliser.normalise_q(physical + update)
        return self.inpaint_start(result, start)

    def _guided_noise_prediction(
        self,
        path: torch.Tensor,
        timesteps: torch.Tensor,
        start: torch.Tensor,
        target_tip: torch.Tensor,
        obstacle_graph: Any,
        normaliser: Any,
        *,
        tip_weight: float,
        obstacle_centres: torch.Tensor | None,
        obstacle_mask: torch.Tensor | None,
        repulsion_weight: float,
        repulsion_safe_radius: float,
        section_lengths: tuple[float, float, float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tip_weight == 0.0 and repulsion_weight == 0.0:
            with torch.no_grad():
                predicted_noise = self.model(
                    path,
                    timesteps,
                    start,
                    target_tip,
                    obstacle_graph,
                    self.condition_mask(path),
                )
            return predicted_noise, path.new_zeros(path.shape[0])
        with torch.enable_grad():
            state = path.detach().requires_grad_(True)
            predicted_noise = self.model(
                state,
                timesteps,
                start,
                target_tip,
                obstacle_graph,
                self.condition_mask(state),
            )
            clean = self.predict_clean(state, timesteps, predicted_noise)
            terminal_physical = normaliser.denormalise_q(clean[:, -1].detach())
            target_physical = normaliser.denormalise_workspace(target_tip)
            energy_per_sample, physical_gradient = terminal_fk_energy_and_gradient(
                terminal_physical,
                target_physical,
                section_lengths=section_lengths,
                weight=tip_weight,
            )
            path_physical_gradient = torch.zeros_like(clean)
            path_physical_gradient[:, -1] = physical_gradient
            if repulsion_weight != 0:
                if obstacle_centres is None or obstacle_mask is None:
                    raise ValueError("obstacle repulsion requires centres and mask")
                repulsion_direction = _obstacle_repulsion_directions(
                    normaliser.denormalise_q(clean.detach()),
                    obstacle_centres,
                    obstacle_mask,
                    section_lengths,
                    repulsion_safe_radius,
                )
                path_physical_gradient -= float(repulsion_weight) * repulsion_direction
            q_scale = torch.as_tensor(
                normaliser.q_scale,
                dtype=clean.dtype,
                device=clean.device,
            )
            normalised_gradient = path_physical_gradient * q_scale
            energy_gradient = torch.autograd.grad(
                clean,
                state,
                grad_outputs=normalised_gradient,
            )[0]

        # The fixed frame cannot be changed.
        energy_gradient = energy_gradient.detach()
        energy_gradient[:, 0] = 0.0

        # score_guided = score_model + grad(log p(goal|x_t)), where
        # grad(log likelihood) = -grad(E). Since eps=-sqrt(1-a_bar)*score,
        # eps_guided = eps_model + sqrt(1-a_bar)*grad(E), with user weights
        # already included in the combined energy above.
        noise_guided = predicted_noise.detach() + (
            _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, path.shape)
            * energy_gradient
        )
        return noise_guided, energy_per_sample.detach()

    def sample_ddim(
        self,
        start: torch.Tensor,
        target_tip: torch.Tensor,
        obstacle_graph: Any,
        *,
        section_lengths: tuple[float, float, float],
        sample_steps: int,
        eta: float,
        generator: torch.Generator | None,
        initial_noise: torch.Tensor | None,
        normaliser: Any | None,
        guidance_weight: float,
        guidance_fraction: float,
        post_correction: bool,
        post_correction_step_size: float,
        post_correction_fraction: float,
        return_trace: bool,
        obstacle_centres: torch.Tensor | None = None,
        obstacle_mask: torch.Tensor | None = None,
        guidance_repulsion_weight: float = 0.0,
        post_correction_repulsion_weight: float = 0.0,
        repulsion_safe_radius: float = 1.0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Sample paths with deterministic/stochastic DDIM.

        ``eta=0`` is deterministic for a fixed ``initial_noise``.  Different
        initial noise seeds can still select different IK/path modes.
        """
        if eta < 0:
            raise ValueError("eta must be non-negative")
        if guidance_weight < 0:
            raise ValueError("guidance_weight must be non-negative")
        if guidance_repulsion_weight < 0:
            raise ValueError("guidance_repulsion_weight must be non-negative")
        if not 0.0 <= guidance_fraction <= 1.0:
            raise ValueError("guidance_fraction must lie in [0,1]")
        if post_correction_step_size < 0:
            raise ValueError("post_correction_step_size must be non-negative")
        if post_correction_repulsion_weight < 0:
            raise ValueError("post_correction_repulsion_weight must be non-negative")
        if not 0.0 <= post_correction_fraction <= 1.0:
            raise ValueError("post_correction_fraction must lie in [0,1]")
        uses_guidance = guidance_weight != 0.0 or guidance_repulsion_weight != 0.0
        uses_correction = post_correction and (
            post_correction_step_size != 0.0
            or post_correction_repulsion_weight != 0.0
        )
        needs_normaliser = uses_guidance or uses_correction
        if needs_normaliser and normaliser is None:
            raise ValueError("analytic guidance/correction requires the training normaliser")
        uses_repulsion = (
            guidance_repulsion_weight != 0.0
            or (post_correction and post_correction_repulsion_weight != 0.0)
        )
        if uses_repulsion and (obstacle_centres is None or obstacle_mask is None):
            raise ValueError("obstacle repulsion requires centres and mask")
        batch_size = start.shape[0]
        expected = (batch_size, self.horizon, 6)
        if initial_noise is None:
            path = torch.randn(
                expected, device=start.device, dtype=start.dtype, generator=generator
            )
        else:
            if initial_noise.shape != expected:
                raise ValueError(f"initial_noise must have shape {expected}")
            path = initial_noise.to(device=start.device, dtype=start.dtype).clone()
        path = self.inpaint_start(path, start)
        schedule = ddim_timesteps(self.config.train_timesteps, sample_steps).to(start.device)
        guide_after = math.floor(sample_steps * (1.0 - guidance_fraction))
        correction_after = math.floor(
            sample_steps * (1.0 - post_correction_fraction)
        )
        trace = [path.detach().clone()] if return_trace else None

        for step_index, timestep in enumerate(schedule):
            time_batch = torch.full(
                (batch_size,), int(timestep.item()), device=start.device, dtype=torch.long
            )
            should_guide = uses_guidance and step_index >= guide_after
            if should_guide:
                predicted_noise, _ = self._guided_noise_prediction(
                    path,
                    time_batch,
                    start,
                    target_tip,
                    obstacle_graph,
                    normaliser,
                    tip_weight=guidance_weight,
                    obstacle_centres=obstacle_centres,
                    obstacle_mask=obstacle_mask,
                    repulsion_weight=guidance_repulsion_weight,
                    repulsion_safe_radius=repulsion_safe_radius,
                    section_lengths=section_lengths,
                )
            else:
                with torch.no_grad():
                    predicted_noise = self.model(
                        path,
                        time_batch,
                        start,
                        target_tip,
                        obstacle_graph,
                        self.condition_mask(path),
                    )

            alpha = self.alphas_cumprod[timestep]
            clean = (path - (1.0 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
            if self.config.clip_denoised:
                clean = clean.clamp(-1.0, 1.0)
            if uses_correction and step_index >= correction_after:
                clean = self.apply_analytical_path_correction(
                    clean,
                    start,
                    target_tip,
                    normaliser,
                    tip_weight=post_correction_step_size,
                    obstacle_centres=obstacle_centres,
                    obstacle_mask=obstacle_mask,
                    repulsion_weight=post_correction_repulsion_weight,
                    repulsion_safe_radius=repulsion_safe_radius,
                    section_lengths=section_lengths,
                )
            previous_timestep = (
                int(schedule[step_index + 1].item())
                if step_index + 1 < len(schedule)
                else -1
            )
            alpha_previous = (
                self.alphas_cumprod[previous_timestep]
                if previous_timestep >= 0
                else alpha.new_tensor(1.0)
            )
            sigma = float(eta) * torch.sqrt(
                ((1.0 - alpha_previous) / (1.0 - alpha)).clamp_min(0.0)
                * (1.0 - alpha / alpha_previous).clamp_min(0.0)
            )
            direction_scale = (
                1.0 - alpha_previous - sigma.square()
            ).clamp_min(0.0).sqrt()
            if previous_timestep >= 0 and eta > 0:
                noise = torch.randn(
                    path.shape,
                    device=path.device,
                    dtype=path.dtype,
                    generator=generator,
                )
            else:
                noise = torch.zeros_like(path)
            path = (
                alpha_previous.sqrt() * clean
                + direction_scale * predicted_noise
                + sigma * noise
            )
            path = self.inpaint_start(path, start).detach()
            if trace is not None:
                trace.append(path.clone())

        if trace is not None:
            return path, torch.stack(trace, dim=1)
        return path

