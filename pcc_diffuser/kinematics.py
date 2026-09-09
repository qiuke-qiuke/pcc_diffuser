"""Differentiable PCC kinematics in nonsingular exponential coordinates.

For section ``i`` the two configuration variables are

    q_i = L_i kappa_i (-sin(phi_i), cos(phi_i))

and the complete section twist is ``[q_i0, q_i1, 0, 0, 0, L_i]``.  This
module evaluates the corresponding SE(3) exponential directly; it never
recovers bending angle or azimuth, so the straight configuration is finite
and differentiable.
"""

from __future__ import annotations

from typing import Sequence

import torch


def _check_q(q: torch.Tensor, last_dim: int) -> None:
    if not isinstance(q, torch.Tensor):
        raise TypeError("PCC configurations must be torch tensors")
    if not torch.is_floating_point(q):
        raise TypeError("PCC configurations must use a floating-point dtype")
    if q.ndim == 0 or q.shape[-1] != last_dim:
        raise ValueError(f"expected last dimension {last_dim}, got {tuple(q.shape)}")
    # Avoid a device-wide synchronisation in the GPU kinematics hot path.
    # Solver outputs are checked in bulk after iteration.
    if q.device.type == "cpu" and not bool(torch.isfinite(q).all()):
        raise ValueError("PCC configurations must contain only finite values")


def _check_section_lengths(
    section_lengths: Sequence[float | torch.Tensor],
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert three finite, positive scalar lengths to the reference dtype/device."""
    if len(section_lengths) != 3:
        raise ValueError("section_lengths must contain exactly three values")
    checked = []
    for length in section_lengths:
        value = torch.as_tensor(length, dtype=reference.dtype, device=reference.device)
        if value.ndim != 0:
            raise ValueError("each segment length must be a scalar")
        if value.device.type == "cpu" and (
            not bool(torch.isfinite(value)) or not bool(value > 0)
        ):
            raise ValueError("segment lengths must be finite and positive")
        checked.append(value)
    return checked[0], checked[1], checked[2]


def _series_coefficients(s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sin(theta)/theta and (1-cos(theta))/theta^2 for theta^2=s."""
    s2 = s * s
    s3 = s2 * s
    a_series = 1.0 - s / 6.0 + s2 / 120.0 - s3 / 5040.0
    b_series = 0.5 - s / 24.0 + s2 / 720.0 - s3 / 40320.0

    # Both torch.where branches are evaluated.  Clamping makes the regular
    # branch finite at s=0 while the Taylor branch supplies the true limit.
    tiny = torch.finfo(s.dtype).eps
    safe_s = s.clamp_min(tiny)
    theta = torch.sqrt(safe_s)
    a_regular = torch.sin(theta) / theta
    b_regular = (1.0 - torch.cos(theta)) / safe_s
    use_series = s < (1e-3 if s.dtype == torch.float32 else 1e-6)
    return (
        torch.where(use_series, a_series, a_regular),
        torch.where(use_series, b_series, b_regular),
    )


def _section_twist_transform_unchecked(xi: torch.Tensor) -> torch.Tensor:
    a, b, omega_z, v_x, v_y, v_z = xi.unbind(dim=-1)
    s = a.square() + b.square()
    sinc, cosc = _series_coefficients(s)
    zero = torch.zeros_like(a)

    w_hat = torch.stack(
        (zero, zero, b, zero, zero, -a, -b, a, zero), dim=-1
    ).reshape(*xi.shape[:-1], 3, 3)
    eye3 = torch.eye(3, dtype=xi.dtype, device=xi.device)
    eye3 = eye3.expand(*xi.shape[:-1], 3, 3)
    rotation = eye3 + sinc[..., None, None] * w_hat + cosc[..., None, None] * (w_hat @ w_hat)
    # Closed-form translational term after cancelling the bending magnitude.
    position = torch.stack((cosc * b * v_z, -cosc * a * v_z, sinc * v_z), dim=-1)
    top = torch.cat((rotation, position.unsqueeze(-1)), dim=-1)
    bottom = torch.stack((zero, zero, zero, torch.ones_like(zero)), dim=-1).unsqueeze(-2)
    return torch.cat((top, bottom), dim=-2)


def section_twist_transform(xi: torch.Tensor) -> torch.Tensor:
    """Evaluate the homogeneous transform of one PCC section twist.

    The Taylor limits keep the straight pose finite and differentiable.
    """
    _check_q(xi, 6)
    constrained = torch.stack((xi[..., 2], xi[..., 3], xi[..., 4]), dim=-1)
    if constrained.device.type == "cpu" and bool(constrained.abs().gt(1e-12).any()):
        raise ValueError("PCC section twists must have form [q1,q2,0,0,0,L]")
    return _section_twist_transform_unchecked(xi)


def _segment_transform_unchecked(
    q_section: torch.Tensor, length: torch.Tensor
) -> torch.Tensor:
    zero = torch.zeros_like(q_section[..., :1])
    length_column = length.expand_as(zero)
    xi = torch.cat((q_section, zero, zero, zero, length_column), dim=-1)
    return _section_twist_transform_unchecked(xi)


def segment_transform(q_section: torch.Tensor, length: float | torch.Tensor) -> torch.Tensor:
    """Evaluate the homogeneous transform of one PCC section.

    Args:
        q_section: exponential-coordinate pair with shape ``[..., 2]``.
        length: section length in the model's length unit.

    Returns:
        Homogeneous transform with shape ``[..., 4, 4]``.
    """
    _check_q(q_section, 2)
    length_t = torch.as_tensor(length, dtype=q_section.dtype, device=q_section.device)
    if length_t.ndim != 0:
        raise ValueError("section length must be a scalar")
    if not bool(torch.isfinite(length_t)) or not bool(length_t > 0):
        raise ValueError("section length must be finite and positive")
    return _segment_transform_unchecked(q_section, length_t)


def forward_kinematics(
    q: torch.Tensor,
    section_lengths: Sequence[float],
) -> torch.Tensor:
    _check_q(q, 6)
    lengths = _check_section_lengths(section_lengths, q)
    t1 = _segment_transform_unchecked(q[..., 0:2], lengths[0])
    t2 = _segment_transform_unchecked(q[..., 2:4], lengths[1])
    t3 = _segment_transform_unchecked(q[..., 4:6], lengths[2])
    return t1 @ t2 @ t3


def tip_position(
    q: torch.Tensor,
    section_lengths: Sequence[float],
) -> torch.Tensor:
    """Map configurations ``[..., 6]`` to world-frame tip positions ``[..., 3]``."""
    return forward_kinematics(q, section_lengths)[..., :3, 3]


def backbone_points(
    q: torch.Tensor,
    section_lengths: Sequence[float],
    points_per_section: int = 3,
) -> torch.Tensor:
    """Sample the complete PCC centreline, including base and section tips.

    Returns shape ``[..., 1 + 3 * points_per_section, 3]``.
    """
    _check_q(q, 6)
    lengths = _check_section_lengths(section_lengths, q)
    if points_per_section < 1:
        raise ValueError("points_per_section must be positive")
    transform = torch.eye(4, dtype=q.dtype, device=q.device).expand(*q.shape[:-1], 4, 4)
    points = [transform[..., :3, 3]]
    fractions = torch.linspace(
        1.0 / points_per_section,
        1.0,
        points_per_section,
        dtype=q.dtype,
        device=q.device,
    )
    for section, length in enumerate(lengths):
        pair = q[..., 2 * section : 2 * section + 2]
        for fraction in fractions:
            partial = _segment_transform_unchecked(pair * fraction, length * fraction)
            points.append((transform @ partial)[..., :3, 3])
        transform = transform @ _segment_transform_unchecked(pair, length)
    return torch.stack(points, dim=-2)


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(*vector.shape[:-1], 3, 3)


def _segment_kinematics_unchecked(
    q_section: torch.Tensor, length: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one section's transform and analytical body-Jacobian columns."""
    length_values = torch.broadcast_to(length, q_section.shape[:-1])
    a, b = q_section.unbind(dim=-1)
    zero = torch.zeros_like(a)
    squared_angle = a.square() + b.square()
    squared_twice = squared_angle.square()
    squared_thrice = squared_twice * squared_angle
    sinc, cosine_coefficient = _series_coefficients(squared_angle)
    cubic_series = (
        1.0 / 6.0
        - squared_angle / 120.0
        + squared_twice / 5040.0
        - squared_thrice / 362880.0
    )
    derivative_cosine_series = (
        -1.0 / 12.0 + squared_angle / 180.0 - squared_twice / 6720.0
    )
    derivative_cubic_series = (
        -1.0 / 60.0 + squared_angle / 1260.0 - squared_twice / 60480.0
    )

    safe_squared = squared_angle.clamp_min(torch.finfo(q_section.dtype).eps)
    cubic_regular = (1.0 - sinc) / safe_squared
    derivative_cosine_regular = (
        (1.0 - 2.0 * cosine_coefficient) / safe_squared - cubic_regular
    )
    derivative_cubic_regular = (
        cosine_coefficient - 3.0 * cubic_regular
    ) / safe_squared
    use_series = squared_angle < (1e-3 if q_section.dtype == torch.float32 else 1e-6)
    cubic_coefficient = torch.where(use_series, cubic_series, cubic_regular)
    derivative_cosine = torch.where(
        use_series, derivative_cosine_series, derivative_cosine_regular
    )
    derivative_cubic = torch.where(
        use_series, derivative_cubic_series, derivative_cubic_regular
    )

    angular = torch.stack((a, b, zero), dim=-1)
    angular_matrix = _skew(angular)
    angular_squared = angular_matrix @ angular_matrix
    identity = torch.eye(3, dtype=q_section.dtype, device=q_section.device)
    identity = identity.expand(*q_section.shape[:-1], 3, 3)
    rotation = (
        identity
        + sinc[..., None, None] * angular_matrix
        + cosine_coefficient[..., None, None] * angular_squared
    )
    angular_columns = (
        identity
        - cosine_coefficient[..., None, None] * angular_matrix
        + cubic_coefficient[..., None, None] * angular_squared
    )[..., :2]

    derivative_cosine_a = a * derivative_cosine
    derivative_cosine_b = b * derivative_cosine
    derivative_cubic_a = a * derivative_cubic
    derivative_cubic_b = b * derivative_cubic
    position_derivatives = length_values[..., None, None] * torch.stack(
        (
            derivative_cosine_a * b,
            derivative_cosine_b * b + cosine_coefficient,
            -derivative_cosine_a * a - cosine_coefficient,
            -derivative_cosine_b * a,
            -derivative_cubic_a * squared_angle - 2.0 * cubic_coefficient * a,
            -derivative_cubic_b * squared_angle - 2.0 * cubic_coefficient * b,
        ),
        dim=-1,
    ).reshape(*q_section.shape[:-1], 3, 2)
    linear_columns = rotation.transpose(-1, -2) @ position_derivatives
    body_columns = torch.cat((angular_columns, linear_columns), dim=-2)
    position = length_values[..., None] * torch.stack(
        (cosine_coefficient * b, -cosine_coefficient * a, sinc), dim=-1
    )
    top = torch.cat((rotation, position.unsqueeze(-1)), dim=-1)
    bottom = torch.stack(
        (zero, zero, zero, torch.ones_like(zero)), dim=-1
    ).unsqueeze(-2)
    return torch.cat((top, bottom), dim=-2), body_columns


def _transform_body_columns(
    distal_transform: torch.Tensor, columns: torch.Tensor
) -> torch.Tensor:
    """Express body-twist columns after a distal transform."""
    rotation = distal_transform[..., :3, :3]
    position = distal_transform[..., :3, 3]
    angular = columns[..., :3, :]
    linear = columns[..., 3:, :]
    position_cross_angular = torch.linalg.cross(
        position.unsqueeze(-2), angular.transpose(-1, -2), dim=-1
    ).transpose(-1, -2)
    inverse_rotation = rotation.transpose(-1, -2)
    return torch.cat(
        (
            inverse_rotation @ angular,
            inverse_rotation @ (linear - position_cross_angular),
        ),
        dim=-2,
    )


def body_kinematic_jacobian(
    q: torch.Tensor,
    section_lengths: Sequence[float],
) -> torch.Tensor:
    """Return the analytical angular-first body Jacobian ``[...,6,6]``."""
    _check_q(q, 6)
    lengths = _check_section_lengths(section_lengths, q)
    transforms, local_columns = _segment_kinematics_unchecked(
        q.reshape(q.shape[:-1] + (3, 2)), torch.stack(lengths)
    )
    section_2 = transforms[..., 1, :, :]
    section_3 = transforms[..., 2, :, :]
    columns_1 = _transform_body_columns(
        section_2 @ section_3, local_columns[..., 0, :, :]
    )
    columns_2 = _transform_body_columns(
        section_3, local_columns[..., 1, :, :]
    )
    columns_3 = local_columns[..., 2, :, :]
    return torch.cat((columns_1, columns_2, columns_3), dim=-1)


def tip_position_and_jacobian(
    q: torch.Tensor,
    section_lengths: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world-frame tip positions and analytical Jacobians together."""
    _check_q(q, 6)
    lengths = _check_section_lengths(section_lengths, q)
    transforms, local_columns = _segment_kinematics_unchecked(
        q.reshape(q.shape[:-1] + (3, 2)), torch.stack(lengths)
    )
    section_1, section_2, section_3 = transforms.unbind(dim=-3)
    transform = section_1 @ section_2 @ section_3
    columns_1 = _transform_body_columns(
        section_2 @ section_3, local_columns[..., 0, :, :]
    )
    columns_2 = _transform_body_columns(
        section_3, local_columns[..., 1, :, :]
    )
    columns_3 = local_columns[..., 2, :, :]
    body = torch.cat((columns_1, columns_2, columns_3), dim=-1)
    jacobian = transform[..., :3, :3] @ body[..., 3:, :]
    return transform[..., :3, 3], jacobian


def backbone_point_jacobians(
    q: torch.Tensor,
    point_indices: torch.Tensor,
    section_lengths: Sequence[float],
    points_per_section: int = 3,
) -> torch.Tensor:
    """Return analytical Jacobians for selected sampled backbone points."""
    _check_q(q, 6)
    lengths = _check_section_lengths(section_lengths, q)
    if points_per_section < 1:
        raise ValueError("points_per_section must be positive")
    if not isinstance(point_indices, torch.Tensor):
        raise TypeError("point_indices must be a torch tensor")
    if point_indices.shape != q.shape[:-1]:
        raise ValueError("point_indices must match the configuration batch shape")
    if point_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("point_indices must use an integer dtype")
    if point_indices.device != q.device:
        raise ValueError("point_indices and configurations must share a device")
    point_count = 3 * points_per_section
    if point_indices.device.type == "cpu" and bool(
        ((point_indices < 0) | (point_indices >= point_count)).any()
    ):
        raise ValueError(f"point_indices must lie in [0,{point_count})")

    flat_q = q.reshape(-1, 6)
    flat_indices = point_indices.reshape(-1)
    sections = torch.div(flat_indices, points_per_section, rounding_mode="floor")
    fractions = (
        flat_indices.remainder(points_per_section).to(q.dtype) + 1.0
    ) / points_per_section
    pairs = flat_q.reshape(-1, 3, 2)
    rows = torch.arange(len(flat_q), device=q.device)
    selected_pairs = pairs[rows, sections]
    length_values = torch.stack(lengths)
    selected_lengths = length_values[sections]
    partial_lengths = selected_lengths * fractions

    section_pairs = torch.stack(
        (selected_pairs * fractions[:, None], pairs[:, 0], pairs[:, 1]), dim=1
    )
    section_length_batches = torch.stack(
        (
            partial_lengths,
            lengths[0].expand_as(partial_lengths),
            lengths[1].expand_as(partial_lengths),
        ),
        dim=1,
    )
    transforms, local_columns = _segment_kinematics_unchecked(
        section_pairs, section_length_batches
    )
    partial, section_1, section_2 = transforms.unbind(dim=1)
    identity = torch.eye(4, dtype=q.dtype, device=q.device).expand(
        len(flat_q), 4, 4
    )
    is_section_0 = sections == 0
    is_section_1 = sections == 1
    prefix = torch.where(
        is_section_0[:, None, None],
        identity,
        torch.where(
            is_section_1[:, None, None], section_1, section_1 @ section_2
        ),
    )
    transform = prefix @ partial
    selected_columns = local_columns[:, 0] * fractions[:, None, None]
    zero_columns = q.new_zeros((len(flat_q), 6, 2))
    full_columns_0 = local_columns[:, 1]
    full_columns_1 = local_columns[:, 2]
    columns_0 = torch.where(
        is_section_0[:, None, None],
        selected_columns,
        torch.where(
            is_section_1[:, None, None],
            _transform_body_columns(partial, full_columns_0),
            _transform_body_columns(section_2 @ partial, full_columns_0),
        ),
    )
    columns_1 = torch.where(
        is_section_0[:, None, None],
        zero_columns,
        torch.where(
            is_section_1[:, None, None],
            selected_columns,
            _transform_body_columns(partial, full_columns_1),
        ),
    )
    columns_2 = torch.where(
        (sections == 2)[:, None, None], selected_columns, zero_columns
    )
    body = torch.cat((columns_0, columns_1, columns_2), dim=-1)
    result = transform[..., :3, :3] @ body[..., 3:, :]
    return result.reshape(q.shape[:-1] + (3, 6))


def sphere_clearances_and_tips(
    configurations: torch.Tensor,
    obstacle_centres: torch.Tensor,
    section_lengths: Sequence[float],
    sphere_radius: float,
    clearance_margin: float,
    points_per_section: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return clearances and tips from one sampled-backbone evaluation."""
    if configurations.ndim < 1 or configurations.shape[-1] != 6:
        raise ValueError("configurations must have shape [...,6]")
    if obstacle_centres.ndim != 2 or obstacle_centres.shape[-1] != 3:
        raise ValueError("obstacle_centres must have shape [K,3]")
    if sphere_radius <= 0 or clearance_margin <= 0:
        raise ValueError("sphere_radius and clearance_margin must be positive")
    batch_shape = configurations.shape[:-1]
    flattened = configurations.reshape(-1, 6)
    points = backbone_points(
        flattened,
        section_lengths,
        points_per_section,
    )[:, 1:]
    tips = points[:, -1].reshape(batch_shape + (3,))
    if len(obstacle_centres) == 0:
        return configurations.new_full(batch_shape, torch.inf), tips

    distances = torch.linalg.vector_norm(
        points.unsqueeze(-2) - obstacle_centres[None, None, :, :], dim=-1
    )
    clearances = (
        distances.amin(dim=(-1, -2))
        - float(sphere_radius)
        - float(clearance_margin)
    )
    return clearances.reshape(batch_shape), tips
