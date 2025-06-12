import numpy as np
from scipy.integrate import solve_ivp

def ball_dynamics_cv(t, state, omega, dynamics):
    # Constant velocity model: state = [x, y, z, vx, vy, vz]
    x, y, z, vx, vy, vz = state
    v = np.array([vx, vy, vz])
    v_mag = np.linalg.norm(v)
    m = dynamics["mass"]
    rho = dynamics["air_density"]
    Cd = dynamics["drag_coefficient"]
    A = np.pi * (dynamics["radius"] ** 2)
    Cl = dynamics["magnus_coefficient"]
    g = np.array(dynamics["gravity"])
    if v_mag == 0:
        return [vx, vy, vz, 0, 0, 0]
    Fd = -0.5 * rho * Cd * A * v_mag * v / m
    Fm = Cl * np.cross(omega, v) / m
    ax, ay, az = Fd + Fm + g
    return [vx, vy, vz, ax, ay, az]

def ball_dynamics_ca(t, state, omega, dynamics):
    # Constant acceleration model: state = [x, y, z, vx, vy, vz, ax, ay, az]
    x, y, z, vx, vy, vz, ax, ay, az = state
    v = np.array([vx, vy, vz])
    v_mag = np.linalg.norm(v)
    m = dynamics["mass"]
    rho = dynamics["air_density"]
    Cd = dynamics["drag_coefficient"]
    A = np.pi * (dynamics["radius"] ** 2)
    Cl = dynamics["magnus_coefficient"]
    g = np.array(dynamics["gravity"])
    if v_mag == 0:
        Fd = np.zeros(3)
        Fm = np.zeros(3)
    else:
        Fd = -0.5 * rho * Cd * A * v_mag * v / m
        Fm = Cl * np.cross(omega, v) / m
    a_total = Fd + Fm + g
    return [vx, vy, vz, ax, ay, az, a_total[0], a_total[1], a_total[2]]

def ekf_predict(state, P, dt, omega, Q, table_height, has_bounced, dynamics, model_type, bounce_factor):
    if model_type == "constant_acceleration":
        sol = solve_ivp(ball_dynamics_ca, [0, dt], state, args=(omega, dynamics), t_eval=[dt])
        state_pred = sol.y[:, -1]
        # Bounce check for CA model
        if state_pred[2] <= table_height and state_pred[5] < 0 and not has_bounced:
            state_pred[5] *= -1
            state_pred[3:6] = state_pred[3:6] * bounce_factor
            state_pred[6:9] = state_pred[6:9] * bounce_factor
            has_bounced = True
        # Jacobian for CA model
        F = np.eye(9)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        F[0, 6] = 0.5 * dt**2
        F[1, 7] = 0.5 * dt**2
        F[2, 8] = 0.5 * dt**2
        F[3, 6] = dt
        F[4, 7] = dt
        F[5, 8] = dt
        P_pred = F @ P @ F.T + Q
        return state_pred, P_pred, has_bounced
    else:
        sol = solve_ivp(ball_dynamics_cv, [0, dt], state, args=(omega, dynamics), t_eval=[dt])
        state_pred = sol.y[:, -1]
        # Bounce check for CV model
        if state_pred[2] <= table_height and state_pred[-1] < 0 and not has_bounced:
            state_pred[-1] *= -1
            state_pred[-3:] = state_pred[-3:] * bounce_factor
            has_bounced = True
        # Jacobian for CV model
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        P_pred = F @ P @ F.T + Q
        return state_pred, P_pred, has_bounced

def ekf_update(state_pred, P_pred, measurement, R, model_type):
    if model_type == "constant_acceleration":
        H = np.zeros((6, 9))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        H[3, 3] = 1
        H[4, 4] = 1
        H[5, 5] = 1
    else:
        H = np.eye(6)
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    state_upd = state_pred + K @ (measurement - H @ state_pred)
    P_upd = (np.eye(P_pred.shape[0]) - K @ H) @ P_pred
    return state_upd, P_upd

def bootstrap_initial_state_from_EKF(measurements, dt, dynamics_config, ekf_config):
    model_type = "constant_acceleration" if not ekf_config["constant_velocity_motion_model"] else "constant_velocity"
    sigma_P = ekf_config["sigma_P"]
    sigma_Q = ekf_config["sigma_Q"]
    sigma_R = ekf_config["sigma_R"]
    bounce_factor = np.array(dynamics_config["bounce_factor"])
    omega = np.array(dynamics_config.get("omega", [0, 0, 0]))

    state_positions = []
    state_velocities = []
    covariances = []
    timestamps = []
    time_from_start = 0
    N = measurements.shape[0]

    if model_type == "constant_acceleration":
        v = np.array([measurements[0][3], measurements[0][4], measurements[0][5]])
        v_norm = np.linalg.norm(v)
        m = dynamics_config["mass"]
        rho = dynamics_config["air_density"]
        Cd = dynamics_config["drag_coefficient"]
        A = np.pi * (dynamics_config["radius"] ** 2)
        g = np.array(dynamics_config["gravity"])
        Fd = -0.5 * rho * Cd * A * v_norm * v / m
        a_0 = Fd + g
        state = np.hstack([measurements[0], a_0])
        P = np.eye(9) * sigma_P
        Q = np.eye(9) * sigma_Q
        R = np.eye(6) * sigma_R
    else:
        state = measurements[0]
        P = np.eye(6) * sigma_P
        Q = np.eye(6) * sigma_Q
        R = np.eye(6) * sigma_R

    has_bounced = False
    for i in range(1, N):
        state, P, has_bounced = ekf_predict(state, P, dt, omega, Q, dynamics_config["height_table"], has_bounced, dynamics_config, model_type, bounce_factor)
        state, P = ekf_update(state, P, measurements[i], R, model_type)
        state_positions.append(state[:3])
        state_velocities.append(state[-3:])
        covariances.append(np.diag(P)[:3])
        timestamps.append(time_from_start)
        time_from_start += dt

    return np.array(state_positions), np.array(state_velocities)