"""
Constrained Kalman Filter (CKF) for rotor state estimation.

State: x = [omega, alpha]^T  (RPM, RPM/s)
Measurement: z = omega (RPM only)
Input: u = cmd_rpm

Dynamics model:
  jerk = -(p1 + p2*omega)*alpha + p3*(u - omega)
  omega_next = omega + alpha*dt + 0.5*jerk*dt^2
  alpha_next = alpha + jerk*dt

Constraints enforced via iterative MAP projection:
  min_speed <= omega <= max_speed
  -max_accel <= alpha <= max_accel
"""

import numpy as np


class CKFParams:
    """Physical and constraint parameters for the rotor CKF."""

    def __init__(
        self,
        p1=25.16687,
        p2=0.003933,
        p3=515.605,
        min_speed=2000.0,
        max_speed=7300.0,
        max_accel=15e3,
        max_jerk=250e3,
    ):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_jerk = max_jerk


def ckf_step(x_prev, P_prev, z_meas, u_cmd, dt, param, Q, R):
    """
    Single step of the Constrained Kalman Filter.

    Parameters
    ----------
    x_prev : ndarray (2,)  – previous state [omega, alpha]
    P_prev : ndarray (2,2) – previous covariance
    z_meas : float          – measured omega (RPM)
    u_cmd  : float          – command RPM
    dt     : float          – time step (s)
    param  : CKFParams      – model / constraint parameters
    Q      : ndarray (2,2)  – process noise covariance
    R      : float           – measurement noise variance

    Returns
    -------
    x_hat  : ndarray (2,)   – updated state
    P      : ndarray (2,2)  – updated covariance
    x_pred : ndarray (2,)   – predicted state (before update)
    P_pred : ndarray (2,2)  – predicted covariance (before update)
    A      : ndarray (2,2)  – linearized state transition matrix
    """
    omega_est = x_prev[0]
    alpha_est = x_prev[1]

    # ── Prediction ──────────────────────────────────────────────
    # Physical jerk
    j_temp = (-(param.p1 + param.p2 * omega_est) * alpha_est
              + param.p3 * (u_cmd - omega_est))
    j_clamp = np.clip(j_temp, -param.max_jerk, param.max_jerk)

    # Check saturation
    if alpha_est >= param.max_accel and j_clamp > 0:
        alpha_eff = param.max_accel
        j_eff = 0.0
        A = np.array([[1.0, 0.0],
                       [0.0, 0.0]])
    elif alpha_est <= -param.max_accel and j_clamp < 0:
        alpha_eff = -param.max_accel
        j_eff = 0.0
        A = np.array([[1.0, 0.0],
                       [0.0, 0.0]])
    else:
        alpha_eff = alpha_est
        j_eff = j_clamp
        A = np.array([
            [1.0, dt],
            [(-param.p2 * alpha_est - param.p3) * dt,
             1.0 - (param.p1 + param.p2 * omega_est) * dt]
        ])

    # State prediction
    x_pred = np.array([
        omega_est + alpha_eff * dt + 0.5 * j_eff * dt**2,
        alpha_est + j_eff * dt,
    ])

    # Clamp predicted acceleration
    x_pred[1] = np.clip(x_pred[1], -param.max_accel, param.max_accel)

    # Covariance prediction
    P_pred = A @ P_prev @ A.T + Q

    # ── Update ──────────────────────────────────────────────────
    C = np.array([[1.0, 0.0]])
    innovation = z_meas - C @ x_pred
    S = (C @ P_pred @ C.T + R).item()
    K = (P_pred @ C.T) / S  # (2,1)

    x_hat = x_pred + (K * innovation).ravel()

    # Joseph form covariance update
    I = np.eye(2)
    IKC = I - K @ C
    P = IKC @ P_pred @ IKC.T + K @ (R * np.eye(1)) @ K.T

    # ── Constraint projection (iterative MAP) ───────────────────
    D = np.array([
        [1.0, 0.0],    # omega <= max_speed
        [-1.0, 0.0],   # omega >= min_speed
        [0.0, 1.0],    # alpha <= max_accel
        [0.0, -1.0],   # alpha >= -max_accel
    ])
    d = np.array([
        param.max_speed,
        -param.min_speed,
        param.max_accel,
        param.max_accel,
    ])

    for _ in range(2):
        violations = D @ x_hat - d
        if np.any(violations > 0):
            idx = np.argmax(violations)
            D_i = D[idx]
            alpha_proj = (D_i @ x_hat - d[idx]) / (D_i @ P @ D_i)
            x_hat = x_hat - alpha_proj * (P @ D_i)

            denom = D_i @ P @ D_i
            if denom > 1e-10:
                P = P - np.outer(P @ D_i, D_i @ P) / denom

            # Ensure symmetry & positive-definiteness
            P = (P + P.T) / 2.0
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.maximum(eigvals, 1e-10)
            P = eigvecs @ np.diag(eigvals) @ eigvecs.T
        else:
            break

    return x_hat, P, x_pred, P_pred, A


def rts_smooth(x_filt, P_filt, x_pred_all, P_pred_all, A_all, param=None):
    """
    Rauch-Tung-Striebel (RTS) fixed-interval smoother.

    Runs a backward pass over the forward-filtered CKF results to
    eliminate phase lag in state estimates. This is the Extended RTS
    variant that uses the linearized transition matrices.

    Parameters
    ----------
    x_filt     : ndarray (N, 2)    – filtered states from forward pass
    P_filt     : ndarray (N, 2, 2) – filtered covariances
    x_pred_all : ndarray (N, 2)    – predicted states (before update)
    P_pred_all : ndarray (N, 2, 2) – predicted covariances (before update)
    A_all      : ndarray (N, 2, 2) – linearized transition matrices
    param      : CKFParams or None – if provided, clip smoothed states

    Returns
    -------
    x_smooth : ndarray (N, 2)    – smoothed states
    P_smooth : ndarray (N, 2, 2) – smoothed covariances
    """
    N = x_filt.shape[0]
    x_smooth = np.zeros_like(x_filt)
    P_smooth = np.zeros_like(P_filt)

    # Initialize with last filtered estimate
    x_smooth[-1] = x_filt[-1].copy()
    P_smooth[-1] = P_filt[-1].copy()

    for k in range(N - 2, -1, -1):
        # Smoother gain: G_k = P_filt[k] @ A_{k+1}^T @ inv(P_pred[k+1])
        # A_all[k+1] is the transition from x_filt[k] to x_pred[k+1]
        P_pred_kp1 = P_pred_all[k + 1]

        # Use solve for numerical stability: G @ P_pred = P_filt @ A^T
        # => G = P_filt[k] @ A^T @ inv(P_pred)
        try:
            G = P_filt[k] @ A_all[k + 1].T @ np.linalg.inv(P_pred_kp1)
        except np.linalg.LinAlgError:
            G = P_filt[k] @ A_all[k + 1].T @ np.linalg.pinv(P_pred_kp1)

        x_smooth[k] = (x_filt[k]
                        + G @ (x_smooth[k + 1] - x_pred_all[k + 1]))
        P_smooth[k] = (P_filt[k]
                        + G @ (P_smooth[k + 1] - P_pred_kp1) @ G.T)

        # Ensure symmetry and positive-definiteness
        P_smooth[k] = (P_smooth[k] + P_smooth[k].T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(P_smooth[k])
        eigvals = np.maximum(eigvals, 1e-10)
        P_smooth[k] = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Clip smoothed state to constraints
        if param is not None:
            x_smooth[k][0] = np.clip(
                x_smooth[k][0], param.min_speed, param.max_speed)
            x_smooth[k][1] = np.clip(
                x_smooth[k][1], -param.max_accel, param.max_accel)

    return x_smooth, P_smooth


def run_ckf(t_meas, z_meas, t_cmd, u_cmd_rpm, param, Q, R,
            x0=None, P0=None, smooth=False):
    """
    Run the CKF over full measurement and command time series.

    Parameters
    ----------
    t_meas     : ndarray (N,)  – measurement timestamps (s, absolute)
    z_meas     : ndarray (N,)  – measured omega (RPM)
    t_cmd      : ndarray (M,)  – command timestamps (s, absolute)
    u_cmd_rpm  : ndarray (M,)  – command RPM
    param      : CKFParams
    Q          : ndarray (2,2) – process noise
    R          : float          – measurement noise variance
    x0         : ndarray (2,) or None – initial state
    P0         : ndarray (2,2) or None – initial covariance
    smooth     : bool – if True, apply RTS smoother after forward pass

    Returns
    -------
    x_est : ndarray (N, 2) – estimated [omega, alpha] at each measurement time
    P_est : ndarray (N, 2, 2) – covariance at each time
    """
    N = len(t_meas)

    if x0 is None:
        x0 = np.array([z_meas[0], 0.0])
    if P0 is None:
        P0 = np.diag([10.0, 100.0])

    x_est = np.zeros((N, 2))
    P_est = np.zeros((N, 2, 2))

    # Storage for smoother intermediates
    x_pred_all = np.zeros((N, 2))
    P_pred_all = np.zeros((N, 2, 2))
    A_all = np.zeros((N, 2, 2))

    x_hat = x0.copy()
    P_hat = P0.copy()

    # Pre-interpolate command to measurement times
    u_interp = np.interp(t_meas, t_cmd, u_cmd_rpm)

    for k in range(N):
        if k == 0:
            dt = np.median(np.diff(t_meas[:min(20, N)]))
        else:
            dt = t_meas[k] - t_meas[k - 1]

        if dt <= 0 or dt > 1.0:
            dt = 0.01  # fallback

        x_hat, P_hat, x_pred, P_pred, A = ckf_step(
            x_hat, P_hat, z_meas[k], u_interp[k], dt, param, Q, R
        )
        x_est[k] = x_hat
        P_est[k] = P_hat
        x_pred_all[k] = x_pred
        P_pred_all[k] = P_pred
        A_all[k] = A

    if smooth:
        x_est, P_est = rts_smooth(
            x_est, P_est, x_pred_all, P_pred_all, A_all, param)

    return x_est, P_est
