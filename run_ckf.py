#!/usr/bin/env python3
"""
Run Constrained Kalman Filter on CSV data and validate with sine-wave fitting.

Usage:
    python run_ckf.py [--csv-dir ./csv_output] [--output-dir ./ckf_result]
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import norm

from ckf.constrained_kalman_filter import CKFParams, run_ckf

# ─── Matplotlib style ──────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.0,
})

C_ACTUAL = '#2563EB'
C_CKF = '#DC2626'
C_CMD = '#16A34A'
C_FIT = '#9333EA'
C_BAR_4K = '#3B82F6'
C_BAR_5K = '#EF4444'

CMD_RAW_TO_RPM = 9800.0 / 8191.0


# ─── Data loading ──────────────────────────────────────────────
def load_csv(path):
    """Load a CSV file into a dict of numpy arrays (column-name → values)."""
    data = np.genfromtxt(path, delimiter=',', names=True)
    return {name: data[name] for name in data.dtype.names}


def load_test(test_dir):
    """Load actual_rpm and cmd_raw for one test directory."""
    td = Path(test_dir)
    actual = load_csv(td / 'actual_rpm.csv')
    cmd = load_csv(td / 'cmd_raw.csv')
    return actual, cmd


# ─── Sine-wave utilities ──────────────────────────────────────
def sine_model(t, A, f, phi, offset):
    """A * sin(2*pi*f*t + phi) + offset"""
    return A * np.sin(2.0 * np.pi * f * t + phi) + offset


def detect_sine_region(t_cmd, cmd_rpm, t_sig, sig):
    """
    Use the command signal to find the sine-wave region:
    1. Detect sine start from cmd (first significant change)
    2. Detect period from cmd peaks
    3. Map those times onto the signal's time axis.

    Returns (sine_start_t, sine_end_t, period) or (None, None, None).
    """
    # Detect sine start: first sample where cmd changes > 5 RPM
    cmd_diff = np.abs(np.diff(cmd_rpm))
    change_idx = np.argmax(cmd_diff > 5)
    if cmd_diff[change_idx] <= 5:
        return None, None, None
    sine_start_t = t_cmd[change_idx]

    # Find period from cmd peaks (reliable)
    amp_range = cmd_rpm.max() - cmd_rpm.min()
    peaks, _ = find_peaks(cmd_rpm, distance=100, prominence=amp_range * 0.3)
    if len(peaks) < 2:
        return None, None, None
    period = float(np.median(np.diff(t_cmd[peaks])))

    # Sine end: last cmd peak + half period (last trough)
    sine_end_t = t_cmd[peaks[-1]] + period / 2.0

    return sine_start_t, sine_end_t, period


def extract_sine_segment(t_sig, sig, sine_start_t, sine_end_t, period,
                         exclude_cycles=1):
    """
    Extract sine-wave segment, trimming exclude_cycles from front and back.
    """
    trim = exclude_cycles * period
    t_start = sine_start_t + trim
    t_end = sine_end_t - trim

    mask = (t_sig >= t_start) & (t_sig <= t_end)
    if np.sum(mask) < 20:
        return None, None

    return t_sig[mask], sig[mask]


def fit_sine(t, sig, f_guess):
    """
    Fit A*sin(2*pi*f*t + phi) + offset to the data.
    Returns (A, f, phi, offset) or None on failure.
    """
    offset_g = np.mean(sig)
    A_g = (np.max(sig) - np.min(sig)) / 2.0
    if A_g < 1e-6:
        A_g = 1.0

    try:
        popt, _ = curve_fit(
            sine_model, t, sig,
            p0=[A_g, f_guess, 0.0, offset_g],
            bounds=([0, f_guess * 0.5, -2 * np.pi, -np.inf],
                    [A_g * 3, f_guess * 2.0, 2 * np.pi, np.inf]),
            maxfev=20000,
        )
        return popt
    except (RuntimeError, ValueError):
        return None


# ─── Numerical differentiation ────────────────────────────────
def central_diff(t, sig):
    """Central-difference derivative."""
    dsig = sig[2:] - sig[:-2]
    dts = t[2:] - t[:-2]
    dts[dts == 0] = 1e-10
    return t[1:-1], dsig / dts


# ─── Plotting helpers ─────────────────────────────────────────
def plot_rpm_overview(t_actual, rpm_actual, t_ckf, rpm_ckf,
                      t_cmd, cmd_rpm, label, test_idx, out_dir):
    """Time-domain RPM: actual vs CKF vs command."""
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(t_actual, rpm_actual, color=C_ACTUAL, lw=0.8, alpha=0.7,
            label='Measured RPM')
    ax.plot(t_ckf, rpm_ckf, color=C_CKF, lw=0.8, alpha=0.8,
            label='CKF Estimated RPM')
    ax.plot(t_cmd, cmd_rpm, color=C_CMD, lw=0.6, alpha=0.5,
            label='Cmd RPM')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('RPM')
    ax.set_title(f'{label} / Test {test_idx} — RPM Overview')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30),
              ncol=3, framealpha=0.9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)
    fig.savefig(out_dir / f'{label}_test{test_idx}_rpm_overview.png')
    plt.close(fig)


C_3SIGMA = '#FBBF24'  # amber


def plot_3sigma_bounds(t, rpm_err, acc_err, P_est, label, test_idx, out_dir):
    """Plot RPM and acceleration error with 3-sigma bounds."""
    rpm_3sig = 3.0 * np.sqrt(P_est[:, 0, 0])
    rpm_within = np.sum(np.abs(rpm_err) <= rpm_3sig) / len(rpm_err) * 100

    acc_3sig = 3.0 * np.sqrt(P_est[:, 1, 1])
    # Exclude edges for acc (numerical diff artifacts)
    sl = slice(50, -50)
    acc_within = np.sum(
        np.abs(acc_err[sl]) <= acc_3sig[sl]) / len(acc_err[sl]) * 100

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)

    # RPM error + 3σ
    axes[0].fill_between(t, -rpm_3sig, rpm_3sig,
                          color=C_3SIGMA, alpha=0.3,
                          label=r'$\pm 3\sigma$ bound')
    axes[0].plot(t, rpm_err, color=C_ACTUAL, lw=0.5, alpha=0.7,
                 label='RPM error')
    axes[0].set_ylabel('RPM Error')
    axes[0].set_title(
        f'{label} / Test {test_idx} — RPM Error '
        r'$\pm 3\sigma$'
        f' ({rpm_within:.1f}% within)')
    axes[0].legend(loc='upper right', fontsize=9)

    # Acc error + 3σ
    axes[1].fill_between(t, -acc_3sig, acc_3sig,
                          color=C_3SIGMA, alpha=0.3,
                          label=r'$\pm 3\sigma$ bound')
    axes[1].plot(t, acc_err, color=C_ACTUAL, lw=0.5, alpha=0.7,
                 label='Acc error')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Acc Error [RPM/s]')
    axes[1].set_title(
        f'Acceleration Error '
        r'$\pm 3\sigma$'
        f' ({acc_within:.1f}% within)')
    axes[1].legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / f'{label}_test{test_idx}_3sigma.png')
    plt.close(fig)

    return rpm_within, acc_within


def plot_acc_overview(t_ckf, acc_ckf, t_actual, rpm_actual,
                      label, test_idx, out_dir):
    """Acceleration: CKF vs numerical derivative of measurement."""
    t_d, acc_d = central_diff(t_actual, rpm_actual)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(t_d, acc_d, color=C_ACTUAL, lw=0.5, alpha=0.5,
            label='Numerical diff (meas)')
    ax.plot(t_ckf, acc_ckf, color=C_CKF, lw=0.8, alpha=0.8,
            label='CKF Estimated')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Acceleration [RPM/s]')
    ax.set_title(f'{label} / Test {test_idx} — Acceleration')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30),
              ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)
    fig.savefig(out_dir / f'{label}_test{test_idx}_acc_overview.png')
    plt.close(fig)


def plot_sine_fit(t_seg, sig_seg, popt, ylabel, label, test_idx,
                  kind, out_dir):
    """Plot trimmed segment + sine fit.
    NOTE: popt was fitted on (t_seg - t_seg[0]), so we evaluate on shifted time.
    """
    t_shifted = t_seg - t_seg[0]
    t_fine_shifted = np.linspace(0, t_shifted[-1], 2000)
    t_fine = t_fine_shifted + t_seg[0]
    fit_fine = sine_model(t_fine_shifted, *popt)

    residual = sig_seg - sine_model(t_shifted, *popt)
    rmse = np.sqrt(np.mean(residual**2))

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), height_ratios=[3, 1],
                              sharex=True)
    axes[0].plot(t_seg, sig_seg, color=C_ACTUAL, lw=0.7, alpha=0.7,
                 label='Data')
    axes[0].plot(t_fine, fit_fine, color=C_FIT, lw=1.2,
                 label=f'Sine fit (RMSE={rmse:.2f})')
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(
        f'{label} / Test {test_idx} — {kind} Sine Fit '
        f'(A={popt[0]:.1f}, f={popt[1]:.3f} Hz)')
    axes[0].legend(loc='best', fontsize=9)

    axes[1].plot(t_seg, residual, color=C_CKF, lw=0.5, alpha=0.7)
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Residual')
    axes[1].axhline(0, color='k', lw=0.5, ls='--')

    fig.tight_layout()
    fig.savefig(out_dir / f'{label}_test{test_idx}_{kind}_sinefit.png')
    plt.close(fig)


def plot_residual_histogram(data_arr, xlabel, label, test_idx, kind, out_dir):
    """Plot residual histogram + fitted Gaussian."""
    mu, std = norm.fit(data_arr)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    n, bins, _ = ax.hist(data_arr, bins=80, density=True, color=C_ACTUAL,
                         alpha=0.6, edgecolor='white', linewidth=0.3,
                         label='Residual distribution')
    x = np.linspace(bins[0], bins[-1], 300)
    ax.plot(x, norm.pdf(x, mu, std), color=C_CKF, lw=1.5,
            label=rf'$\mathcal{{N}}(\mu={mu:.1f},\,\sigma={std:.1f})$')
    ax.axvline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability Density')
    ax.set_title(f'{label} / Test {test_idx} — {kind} Residual')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.26),
              ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(out_dir / f'{label}_test{test_idx}_{kind}_hist.png')
    plt.close(fig)
    return mu, std


def _arcsine_pdf(x, a, b):
    """Arcsine distribution PDF on [a, b]: 1 / (pi * sqrt((x-a)*(b-x)))"""
    mask = (x > a) & (x < b)
    pdf = np.zeros_like(x)
    pdf[mask] = 1.0 / (np.pi * np.sqrt((x[mask] - a) * (b - x[mask])))
    return pdf


def plot_values_histogram(data_arr, xlabel, label, test_idx, kind, out_dir,
                          color=C_CKF):
    """Plot value histogram with arcsine distribution overlay (correct for sine signals)."""
    a, b = np.min(data_arr), np.max(data_arr)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    n, bins, _ = ax.hist(data_arr, bins=80, density=True, color=color,
                         alpha=0.6, edgecolor='white', linewidth=0.3,
                         label='Distribution')

    # Arcsine fit (theoretical for uniform-sampled sine)
    margin = (b - a) * 0.02
    x = np.linspace(a + margin, b - margin, 500)
    pdf = _arcsine_pdf(x, a, b)
    ax.plot(x, pdf, color='k', lw=1.5,
            label=f'Arcsine [{a:.0f}, {b:.0f}]')

    mid = (a + b) / 2.0
    ax.axvline(mid, color='gray', lw=0.5, ls='--', alpha=0.5,
               label=f'Mean = {mid:.0f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability Density')
    ax.set_title(f'{label} / Test {test_idx} — {kind} Distribution')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.26),
              ncol=2, framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(out_dir / f'{label}_test{test_idx}_{kind}_hist.png')
    plt.close(fig)


def plot_aggregate_residual_histograms(all_data, xlabel, kind, out_dir):
    """Aggregate residual histograms with Gaussian fit (per-RPM + overall)."""
    rpm_keys = sorted(all_data.keys())
    colors = {rpm_keys[0]: C_BAR_4K}
    if len(rpm_keys) > 1:
        colors[rpm_keys[1]] = C_BAR_5K

    # Per-RPM side-by-side
    fig, axes = plt.subplots(1, len(rpm_keys),
                              figsize=(5.5 * len(rpm_keys), 4))
    if len(rpm_keys) == 1:
        axes = [axes]
    for ax, rpm in zip(axes, rpm_keys):
        combined = np.concatenate(all_data[rpm])
        mu, std = norm.fit(combined)
        n, bins, _ = ax.hist(combined, bins=100, density=True,
                             color=colors.get(rpm, 'gray'),
                             alpha=0.55, edgecolor='white', lw=0.3,
                             label=f'All tests (n={len(combined)})')
        x = np.linspace(bins[0], bins[-1], 300)
        ax.plot(x, norm.pdf(x, mu, std), color='k', lw=1.5,
                label=rf'$\mathcal{{N}}(\mu={mu:.1f},\,\sigma={std:.1f})$')
        ax.axvline(0, color='k', lw=0.5, ls='--', alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Probability Density')
        ax.set_title(f'{rpm} — Aggregate {kind}')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.26),
                  ncol=2, framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.26)
    fig.savefig(out_dir / f'aggregate_{kind}_per_rpm.png')
    plt.close(fig)

    # Overall
    fig, ax = plt.subplots(figsize=(6, 4))
    all_combined = []
    for rpm in rpm_keys:
        combined = np.concatenate(all_data[rpm])
        all_combined.append(combined)
        ax.hist(combined, bins=100, density=True,
                color=colors.get(rpm, 'gray'), alpha=0.45,
                edgecolor='white', lw=0.3,
                label=f'{rpm} (n={len(combined)})')
    everything = np.concatenate(all_combined)
    mu, std = norm.fit(everything)
    x = np.linspace(everything.min(), everything.max(), 300)
    ax.plot(x, norm.pdf(x, mu, std), 'k-', lw=1.5,
            label=rf'Overall $\mathcal{{N}}(\mu={mu:.1f},\,\sigma={std:.1f})$')
    ax.axvline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability Density')
    ax.set_title(f'All Conditions — Aggregate {kind}')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.26),
              ncol=3, framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(out_dir / f'aggregate_{kind}_all.png')
    plt.close(fig)


def plot_aggregate_values_histograms(all_data, xlabel, kind, out_dir):
    """Aggregate values histograms with arcsine fit (per-RPM + overall)."""
    rpm_keys = sorted(all_data.keys())
    colors = {rpm_keys[0]: C_BAR_4K}
    if len(rpm_keys) > 1:
        colors[rpm_keys[1]] = C_BAR_5K

    # Per-RPM side-by-side
    fig, axes = plt.subplots(1, len(rpm_keys),
                              figsize=(5.5 * len(rpm_keys), 4))
    if len(rpm_keys) == 1:
        axes = [axes]
    for ax, rpm in zip(axes, rpm_keys):
        combined = np.concatenate(all_data[rpm])
        a, b = combined.min(), combined.max()
        n, bins, _ = ax.hist(combined, bins=100, density=True,
                             color=colors.get(rpm, 'gray'),
                             alpha=0.55, edgecolor='white', lw=0.3,
                             label=f'All tests (n={len(combined)})')
        margin = (b - a) * 0.02
        x = np.linspace(a + margin, b - margin, 500)
        ax.plot(x, _arcsine_pdf(x, a, b), color='k', lw=1.5,
                label=f'Arcsine [{a:.0f}, {b:.0f}]')
        mid = (a + b) / 2.0
        ax.axvline(mid, color='gray', lw=0.5, ls='--', alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Probability Density')
        ax.set_title(f'{rpm} — Aggregate {kind}')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.26),
                  ncol=2, framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.26)
    fig.savefig(out_dir / f'aggregate_{kind}_per_rpm.png')
    plt.close(fig)

    # Overall (each RPM separate color, no combined fit since ranges differ)
    fig, ax = plt.subplots(figsize=(6, 4))
    for rpm in rpm_keys:
        combined = np.concatenate(all_data[rpm])
        ax.hist(combined, bins=100, density=True,
                color=colors.get(rpm, 'gray'), alpha=0.45,
                edgecolor='white', lw=0.3,
                label=f'{rpm} (n={len(combined)})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability Density')
    ax.set_title(f'All Conditions — Aggregate {kind}')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.26),
              ncol=len(rpm_keys), framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(out_dir / f'aggregate_{kind}_all.png')
    plt.close(fig)


# ─── Summary ──────────────────────────────────────────────────
def save_summary(summary_rows, out_dir):
    """Save summary to CSV and TXT."""
    header = [
        'Condition', 'Test',
        'RPM_RMSE', 'RPM_3sigma%', 'Acc_3sigma%',
        'Vel_Fit_RMSE', 'Vel_Fit_A', 'Vel_Fit_f',
        'Acc_Fit_RMSE', 'Acc_Fit_A', 'Acc_Fit_f',
    ]
    csv_path = out_dir / 'ckf_run_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(summary_rows)
    print(f'  Saved summary: {csv_path}')


# ─── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Run CKF on CSV data and validate with sine-wave fitting')
    parser.add_argument('--csv-dir', type=str, default='./csv_output',
                        help='CSV data directory')
    parser.add_argument('--rpm-folders', nargs='+',
                        default=['4000RPM', '5000RPM'])
    parser.add_argument('--num-tests', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default='./ckf_result')
    args = parser.parse_args()

    csv_base = Path(args.csv_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── CKF parameters ──
    param = CKFParams(
        p1=25.16687,
        p2=0.003933,
        p3=515.605,
        min_speed=2000.0,
        max_speed=7300.0,
        max_accel=15e3,
        max_jerk=250e3,
    )

    # ── CKF tuning ──
    # Tuned for ~99.7% RPM 3-sigma coverage and ~98% acceleration coverage
    Q = np.diag([1.5**2, 150.0**2])
    # R: measurement noise variance
    R = 50.0

    # Storage for aggregate histograms
    all_vel_residuals = {}   # {rpm: [arr, ...]}
    all_acc_residuals = {}
    all_vel_values = {}
    all_acc_values = {}
    summary_rows = []

    for rpm_folder in args.rpm_folders:
        rpm_dir = csv_base / rpm_folder
        if not rpm_dir.exists():
            print(f'[SKIP] {rpm_dir} not found')
            continue

        print(f'\n{"="*60}')
        print(f'  {rpm_folder}')
        print(f'{"="*60}')

        all_vel_residuals[rpm_folder] = []
        all_acc_residuals[rpm_folder] = []
        all_vel_values[rpm_folder] = []
        all_acc_values[rpm_folder] = []

        for ti in range(1, args.num_tests + 1):
            test_dir = rpm_dir / f'test{ti}'
            if not test_dir.exists():
                print(f'  [SKIP] test{ti}: directory not found')
                continue

            print(f'\n  ── test{ti} ──')

            # Load data
            actual, cmd = load_test(test_dir)
            t_actual = actual['time_s']
            rpm_actual = actual['rpm']
            t_cmd = cmd['time_s']
            cmd_rpm = cmd['cmd_raw'] * CMD_RAW_TO_RPM

            # Use earliest common time as reference
            t0 = min(t_actual[0], t_cmd[0])
            t_a = t_actual - t0
            t_c = t_cmd - t0

            # ── Run CKF ──
            x_est, P_est = run_ckf(
                t_actual, rpm_actual, t_cmd, cmd_rpm,
                param, Q, R,
                x0=np.array([rpm_actual[0], 0.0]),
                P0=np.diag([10.0, 100.0]),
            )
            rpm_ckf = x_est[:, 0]
            acc_ckf = x_est[:, 1]

            # RPM RMSE
            rpm_err = rpm_actual - rpm_ckf
            rpm_rmse = np.sqrt(np.mean(rpm_err**2))
            print(f'    RPM RMSE: {rpm_rmse:.4f}')

            # Numerical acceleration for 3-sigma comparison
            acc_num = np.zeros_like(rpm_actual)
            acc_num[1:-1] = ((rpm_actual[2:] - rpm_actual[:-2])
                             / (t_actual[2:] - t_actual[:-2]))
            acc_num[0] = acc_num[1]
            acc_num[-1] = acc_num[-2]
            acc_err = acc_num - acc_ckf

            # ── 3-sigma analysis ──
            rpm_3s_pct, acc_3s_pct = plot_3sigma_bounds(
                t_a, rpm_err, acc_err, P_est,
                rpm_folder, ti, out_dir)
            print(f'    3-sigma: RPM {rpm_3s_pct:.1f}%, '
                  f'Acc {acc_3s_pct:.1f}%')

            # ── Plot overviews ──
            plot_rpm_overview(t_a, rpm_actual, t_a, rpm_ckf,
                              t_c, cmd_rpm, rpm_folder, ti, out_dir)
            plot_acc_overview(t_a, acc_ckf, t_a, rpm_actual,
                              rpm_folder, ti, out_dir)

            # ── Sine-wave validation ──
            # Detect sine region from command signal (reliable)
            sine_info = detect_sine_region(t_cmd, cmd_rpm, t_actual, rpm_ckf)

            vel_fit_rmse = np.nan
            vel_A = np.nan
            vel_f = np.nan
            acc_fit_rmse = np.nan
            acc_A = np.nan
            acc_f = np.nan

            if sine_info[0] is not None:
                sine_start, sine_end, period = sine_info
                f_guess = 1.0 / period
                print(f'    Sine region: [{sine_start - t0:.2f}, '
                      f'{sine_end - t0:.2f}] s, period={period:.4f} s')

                # 1) Velocity sine fit (CKF output)
                seg = extract_sine_segment(
                    t_actual, rpm_ckf, sine_start, sine_end, period,
                    exclude_cycles=1)
                if seg[0] is not None:
                    t_seg_v, sig_seg_v = seg
                    popt_v = fit_sine(t_seg_v - t_seg_v[0], sig_seg_v,
                                      f_guess)
                    if popt_v is not None:
                        vel_A, vel_f = popt_v[0], popt_v[1]
                        fitted_v = sine_model(t_seg_v - t_seg_v[0], *popt_v)
                        vel_fit_rmse = np.sqrt(
                            np.mean((sig_seg_v - fitted_v)**2))
                        print(f'    Velocity sine fit: A={vel_A:.1f} RPM, '
                              f'f={vel_f:.4f} Hz, RMSE={vel_fit_rmse:.2f}')
                        plot_sine_fit(t_seg_v - t0, sig_seg_v, popt_v,
                                      'RPM', rpm_folder, ti,
                                      'velocity', out_dir)

                        v_resid = sig_seg_v - fitted_v
                        plot_residual_histogram(
                            v_resid, 'Velocity Residual [RPM]',
                            rpm_folder, ti, 'vel_residual', out_dir)
                        all_vel_residuals[rpm_folder].append(v_resid)

                        plot_values_histogram(
                            sig_seg_v, 'CKF Velocity [RPM]',
                            rpm_folder, ti, 'vel_values', out_dir)
                        all_vel_values[rpm_folder].append(sig_seg_v)

                # 2) Acceleration sine fit (CKF output)
                seg_a = extract_sine_segment(
                    t_actual, acc_ckf, sine_start, sine_end, period,
                    exclude_cycles=1)
                if seg_a[0] is not None:
                    t_seg_a, sig_seg_a = seg_a
                    popt_a = fit_sine(t_seg_a - t_seg_a[0], sig_seg_a,
                                      f_guess)
                    if popt_a is not None:
                        acc_A, acc_f = popt_a[0], popt_a[1]
                        fitted_a = sine_model(t_seg_a - t_seg_a[0], *popt_a)
                        acc_fit_rmse = np.sqrt(
                            np.mean((sig_seg_a - fitted_a)**2))
                        print(f'    Accel sine fit:    A={acc_A:.1f} RPM/s, '
                              f'f={acc_f:.4f} Hz, RMSE={acc_fit_rmse:.2f}')
                        plot_sine_fit(t_seg_a - t0, sig_seg_a, popt_a,
                                      'Acceleration [RPM/s]', rpm_folder, ti,
                                      'accel', out_dir)

                        a_resid = sig_seg_a - fitted_a
                        plot_residual_histogram(
                            a_resid, 'Accel Residual [RPM/s]',
                            rpm_folder, ti, 'acc_residual', out_dir)
                        all_acc_residuals[rpm_folder].append(a_resid)

                        plot_values_histogram(
                            sig_seg_a, 'CKF Acceleration [RPM/s]',
                            rpm_folder, ti, 'acc_values', out_dir)
                        all_acc_values[rpm_folder].append(sig_seg_a)
            else:
                print('    [WARN] Could not detect sine region from cmd')

            summary_rows.append([
                rpm_folder, f'test{ti}',
                f'{rpm_rmse:.4f}',
                f'{rpm_3s_pct:.1f}',
                f'{acc_3s_pct:.1f}',
                f'{vel_fit_rmse:.4f}' if not np.isnan(vel_fit_rmse) else 'N/A',
                f'{vel_A:.1f}' if not np.isnan(vel_A) else 'N/A',
                f'{vel_f:.4f}' if not np.isnan(vel_f) else 'N/A',
                f'{acc_fit_rmse:.4f}' if not np.isnan(acc_fit_rmse) else 'N/A',
                f'{acc_A:.1f}' if not np.isnan(acc_A) else 'N/A',
                f'{acc_f:.4f}' if not np.isnan(acc_f) else 'N/A',
            ])

    # ── Aggregate histograms ──
    print(f'\n{"="*60}')
    print('  Aggregate histograms')
    print(f'{"="*60}')

    if any(all_vel_residuals.values()):
        plot_aggregate_residual_histograms(
            all_vel_residuals, 'Velocity Residual [RPM]',
            'vel_residual', out_dir)
    if any(all_acc_residuals.values()):
        plot_aggregate_residual_histograms(
            all_acc_residuals, 'Accel Residual [RPM/s]',
            'acc_residual', out_dir)
    if any(all_vel_values.values()):
        plot_aggregate_values_histograms(
            all_vel_values, 'CKF Velocity [RPM]',
            'vel_values', out_dir)
    if any(all_acc_values.values()):
        plot_aggregate_values_histograms(
            all_acc_values, 'CKF Acceleration [RPM/s]',
            'acc_values', out_dir)

    # ── Save summary ──
    save_summary(summary_rows, out_dir)

    print(f'\nAll results saved to {out_dir}/')


if __name__ == '__main__':
    main()
