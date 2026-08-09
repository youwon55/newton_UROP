"""Warp kernels for the educational, rigid-contact PGS solver.

This module deliberately implements the velocity-level Projected
Gauss--Seidel algorithm directly.  It does not call any other Newton solver.

The contact kernel has a launch dimension of one.  That is intentionally slow,
but important: every contact updates ``body_qd`` before the next contact reads
it, which is the defining Gauss--Seidel property.  Parallelising contacts that
share a body would turn this reference implementation into a Jacobi-like
approximation (or introduce a data race).
"""

from __future__ import annotations

import warp as wp

from ...sim import BodyFlags


_EPSILON = wp.constant(1.0e-8)


@wp.func
def _zero_mat33() -> wp.mat33:
    return wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.func
def _point_velocity(qd: wp.spatial_vector, r: wp.vec3) -> wp.vec3:
    """Velocity at a world-space point whose COM offset is ``r``."""
    return wp.spatial_top(qd) + wp.cross(wp.spatial_bottom(qd), r)


@wp.func
def _world_inv_inertia_mul(
    body_q: wp.transform,
    inv_inertia_body: wp.mat33,
    vector_world: wp.vec3,
) -> wp.vec3:
    """Apply a body-frame inverse inertia tensor to a world-frame vector."""
    rotation = wp.transform_get_rotation(body_q)
    vector_body = wp.quat_rotate_inv(rotation, vector_world)
    return wp.quat_rotate(rotation, inv_inertia_body * vector_body)


@wp.func
def _effective_mass_along(
    direction: wp.vec3,
    r0: wp.vec3,
    r1: wp.vec3,
    q0: wp.transform,
    q1: wp.transform,
    inv_mass0: float,
    inv_mass1: float,
    inv_inertia0: wp.mat33,
    inv_inertia1: wp.mat33,
) -> float:
    """Return ``J M^-1 J^T`` for a scalar impulse along ``direction``."""
    angular0 = _world_inv_inertia_mul(q0, inv_inertia0, wp.cross(r0, direction))
    angular1 = _world_inv_inertia_mul(q1, inv_inertia1, wp.cross(r1, direction))

    return (
        inv_mass0
        + inv_mass1
        + wp.dot(direction, wp.cross(angular0, r0))
        + wp.dot(direction, wp.cross(angular1, r1))
    )


@wp.func
def _apply_impulse(
    qd: wp.spatial_vector,
    body_q: wp.transform,
    body_com: wp.vec3,
    inv_mass: float,
    inv_inertia: wp.mat33,
    point_world: wp.vec3,
    impulse_world: wp.vec3,
) -> wp.spatial_vector:
    """Return the velocity after applying a world-space impulse at a point."""
    com_world = wp.transform_point(body_q, body_com)
    r = point_world - com_world
    linear = wp.spatial_top(qd) + inv_mass * impulse_world
    angular = wp.spatial_bottom(qd) + _world_inv_inertia_mul(
        body_q, inv_inertia, wp.cross(r, impulse_world)
    )
    return wp.spatial_vector(linear, angular)


@wp.kernel
def predict_body_velocities(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inertia: wp.array[wp.mat33],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    body_flags: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    angular_damping: float,
    dt: float,
    body_qd_out: wp.array[wp.spatial_vector],
):
    """Compute the unconstrained semi-implicit velocity ``v*``.

    The PGS solve subsequently edits ``body_qd_out`` in place.  Kinematic
    bodies pass through unchanged; their effective inverse mass is handled by
    the solver's persistent buffers during contact resolution.
    """
    body = wp.tid()

    if (body_flags[body] & BodyFlags.KINEMATIC) != 0:
        body_qd_out[body] = body_qd[body]
        return

    q = body_q[body]
    qd = body_qd[body]
    force = body_f[body]
    inv_mass = body_inv_mass[body]

    # Match Newton's model convention, including its global-world index.
    world_gravity = gravity[body_world[body]]

    linear = wp.spatial_top(qd)
    angular = wp.spatial_bottom(qd)
    force_linear = wp.spatial_top(force)
    torque_world = wp.spatial_bottom(force)

    # Linear acceleration is evaluated at the centre of mass.
    linear = linear + (force_linear * inv_mass + world_gravity * wp.nonzero(inv_mass)) * dt

    # Euler's rigid-body equation in the body frame:
    # I w_dot + w x (I w) = tau.
    rotation = wp.transform_get_rotation(q)
    angular_body = wp.quat_rotate_inv(rotation, angular)
    torque_body = wp.quat_rotate_inv(rotation, torque_world)
    gyroscopic_torque = wp.cross(angular_body, body_inertia[body] * angular_body)
    angular_body = angular_body + body_inv_inertia[body] * (torque_body - gyroscopic_torque) * dt
    angular = wp.quat_rotate(rotation, angular_body)
    angular = angular * wp.max(0.0, 1.0 - angular_damping * dt)

    body_qd_out[body] = wp.spatial_vector(linear, angular)


@wp.kernel
def solve_contacts_pgs(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    shape_body: wp.array[int],
    shape_material_mu: wp.array[float],
    contact_count: wp.array[int],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[float],
    contact_margin1: wp.array[float],
    contact_capacity: int,
    baumgarte: float,
    contact_slop: float,
    max_bias_velocity: float,
    dt: float,
    enable_friction: bool,
    normal_lambda: wp.array[float],
    tangent_lambda: wp.array[wp.vec3],
):
    """Run one *sequential* PGS sweep over all rigid contacts.

    ``normal_lambda`` and ``tangent_lambda`` are accumulated over iterations
    in a single time step.  The normal update is projected to ``lambda_n >= 0``;
    the tangential update is projected to the Coulomb disk
    ``|lambda_t| <= mu * lambda_n``.
    """
    # Exactly one thread traverses the constraint rows in a deterministic order.
    if wp.tid() != 0:
        return

    count = wp.min(contact_count[0], contact_capacity)

    for contact in range(count):
        shape0 = contact_shape0[contact]
        shape1 = contact_shape1[contact]

        body0 = -1
        body1 = -1
        if shape0 >= 0:
            body0 = shape_body[shape0]
        if shape1 >= 0:
            body1 = shape_body[shape1]

        # Self contacts and world/world contacts have no usable constraint row.
        if body0 == body1:
            continue

        q0 = wp.transform_identity()
        q1 = wp.transform_identity()
        com0 = wp.vec3(0.0)
        com1 = wp.vec3(0.0)
        inv_mass0 = 0.0
        inv_mass1 = 0.0
        inv_inertia0 = _zero_mat33()
        inv_inertia1 = _zero_mat33()
        qd0 = wp.spatial_vector()
        qd1 = wp.spatial_vector()

        point0 = contact_point0[contact]
        point1 = contact_point1[contact]

        if body0 >= 0:
            q0 = body_q[body0]
            com0 = body_com[body0]
            inv_mass0 = body_inv_mass[body0]
            inv_inertia0 = body_inv_inertia[body0]
            qd0 = body_qd[body0]
            point0 = wp.transform_point(q0, point0)

        if body1 >= 0:
            q1 = body_q[body1]
            com1 = body_com[body1]
            inv_mass1 = body_inv_mass[body1]
            inv_inertia1 = body_inv_inertia[body1]
            qd1 = body_qd[body1]
            point1 = wp.transform_point(q1, point1)

        normal_length = wp.length(contact_normal[contact])
        if normal_length <= _EPSILON:
            continue
        normal = contact_normal[contact] / normal_length

        # Newton's normal convention is shape 0 -> shape 1.
        separation = wp.dot(normal, point1 - point0) - (contact_margin0[contact] + contact_margin1[contact])
        if separation > 0.0:
            continue

        com_world0 = wp.transform_point(q0, com0)
        com_world1 = wp.transform_point(q1, com1)
        r0 = point0 - com_world0
        r1 = point1 - com_world1

        relative_velocity = _point_velocity(qd1, r1) - _point_velocity(qd0, r0)
        normal_velocity = wp.dot(relative_velocity, normal)

        penetration = wp.max(0.0, -separation - contact_slop)
        # Baumgarte turns positional overlap into a separating *speed*.
        # The division by dt is essential: beta is the fraction of the
        # penetration to correct in this time step, not metres per second.
        target_velocity = wp.min(max_bias_velocity, baumgarte * penetration / dt)
        effective_mass = _effective_mass_along(
            normal,
            r0,
            r1,
            q0,
            q1,
            inv_mass0,
            inv_mass1,
            inv_inertia0,
            inv_inertia1,
        )

        if effective_mass <= _EPSILON:
            continue

        # Projected Gauss--Seidel normal update:
        # lambda_new = max(0, lambda_old - (vn - target) / K).
        lambda_old = normal_lambda[contact]
        lambda_new = wp.max(0.0, lambda_old - (normal_velocity - target_velocity) / effective_mass)
        delta_lambda = lambda_new - lambda_old
        normal_lambda[contact] = lambda_new

        if delta_lambda != 0.0:
            normal_impulse = normal * delta_lambda
            if body0 >= 0:
                body_qd[body0] = _apply_impulse(
                    qd0, q0, com0, inv_mass0, inv_inertia0, point0, -normal_impulse
                )
            if body1 >= 0:
                body_qd[body1] = _apply_impulse(
                    qd1, q1, com1, inv_mass1, inv_inertia1, point1, normal_impulse
                )

        if not enable_friction:
            continue

        # Re-read velocities after the normal impulse: this immediate use of
        # corrected state is the Gauss--Seidel part of the friction update too.
        if body0 >= 0:
            qd0 = body_qd[body0]
        if body1 >= 0:
            qd1 = body_qd[body1]
        relative_velocity = _point_velocity(qd1, r1) - _point_velocity(qd0, r0)
        tangent_velocity = relative_velocity - normal * wp.dot(relative_velocity, normal)
        tangent_speed = wp.length(tangent_velocity)

        if tangent_speed <= _EPSILON:
            continue

        tangent = tangent_velocity / tangent_speed
        tangent_effective_mass = _effective_mass_along(
            tangent,
            r0,
            r1,
            q0,
            q1,
            inv_mass0,
            inv_mass1,
            inv_inertia0,
            inv_inertia1,
        )

        if tangent_effective_mass <= _EPSILON:
            continue

        material_count = 0
        friction = 0.0
        if shape0 >= 0:
            friction = friction + shape_material_mu[shape0]
            material_count += 1
        if shape1 >= 0:
            friction = friction + shape_material_mu[shape1]
            material_count += 1
        if material_count > 0:
            friction = friction / float(material_count)

        old_tangent_lambda = tangent_lambda[contact]
        candidate_tangent_lambda = old_tangent_lambda - tangent_velocity / tangent_effective_mass
        max_tangent_lambda = friction * lambda_new
        candidate_length = wp.length(candidate_tangent_lambda)
        new_tangent_lambda = candidate_tangent_lambda
        if candidate_length > max_tangent_lambda and candidate_length > _EPSILON:
            new_tangent_lambda = candidate_tangent_lambda * (max_tangent_lambda / candidate_length)

        delta_tangent_lambda = new_tangent_lambda - old_tangent_lambda
        tangent_lambda[contact] = new_tangent_lambda

        if body0 >= 0:
            body_qd[body0] = _apply_impulse(
                body_qd[body0], q0, com0, inv_mass0, inv_inertia0, point0, -delta_tangent_lambda
            )
        if body1 >= 0:
            body_qd[body1] = _apply_impulse(
                body_qd[body1], q1, com1, inv_mass1, inv_inertia1, point1, delta_tangent_lambda
            )


@wp.kernel
def integrate_body_positions(
    body_q_in: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_flags: wp.array[wp.int32],
    dt: float,
    body_q_out: wp.array[wp.transform],
):
    """Advance poses using the PGS-corrected COM velocities."""
    body = wp.tid()

    if (body_flags[body] & BodyFlags.KINEMATIC) != 0:
        body_q_out[body] = body_q_in[body]
        return

    q = body_q_in[body]
    qd = body_qd[body]
    translation = wp.transform_get_translation(q)
    rotation = wp.transform_get_rotation(q)
    com = body_com[body]

    com_world = translation + wp.quat_rotate(rotation, com)
    com_world = com_world + wp.spatial_top(qd) * dt

    angular = wp.spatial_bottom(qd)
    rotation = wp.normalize(rotation + wp.quat(angular, 0.0) * rotation * (0.5 * dt))

    body_q_out[body] = wp.transform(com_world - wp.quat_rotate(rotation, com), rotation)
