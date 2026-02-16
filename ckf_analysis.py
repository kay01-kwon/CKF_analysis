import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm
import csv
import argparse
from pathlib import Path
from utils.bag_extractor import extract_all_bags

# ============================================================
# Paper-quality matplotlib style (LaTeX-like)
# ============================================================
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
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.0,
})

# Color palette
C_ACTUAL = '#2563EB'   # blue
C_CKF = '#DC2626'      # red
C_ERROR = '#6B7280'     # gray
C_3SIGMA = '#FBBF24'    # amber
C_BAR_4K = '#3B82F6'    # blue
C_BAR_5K = '#EF4444'    # red


# ============================================================
# Utility
# ============================================================
def compute_numerical_acceleration(t, rpm):
    """Central difference: a[i] = (rpm[i+1] - rpm[i-1]) / (t[i+1] - t[i-1])"""
    acc = (rpm[2:] - rpm[:-2]) / (t[2:] - t[:-2])
    return t[1:-1], acc


def get_errors(data, t0):
    """Return (rpm_error, acc_error) arrays for a single test."""
    t_actual = data['actual_rpm']['t'] - t0
    rpm_actual = data['actual_rpm']['rpm']
    t_ckf = data['rotor_state']['t'] - t0
    rpm_ckf = data['rotor_state']['rpm']
    acc_ckf = data['rotor_state']['acceleration']

    # RPM error
    rpm_ckf_interp = np.interp(t_actual, t_ckf, rpm_ckf)
    rpm_err = rpm_actual - rpm_ckf_interp

    # Acc error
    t_diff, acc_diff = compute_numerical_acceleration(t_actual, rpm_actual)
    acc_ckf_interp = np.interp(t_diff, t_ckf, acc_ckf)
    acc_err = acc_diff - acc_ckf_interp

    return rpm_err, acc_err


def rmse_from_error(err):
    """RMSE from an error array."""
    return np.sqrt(np.mean(err ** 2))


# ============================================================
# Per-test plots
# ============================================================
def plot_rpm_comparison(data, t0, rpm_label, test_idx, output_dir):
    t_actual = data['actual_rpm']['t'] - t0
    rpm_actual = data['actual_rpm']['rpm']
    t_ckf = data['rotor_state']['t'] - t0
    rpm_ckf = data['rotor_state']['rpm']

    rpm_ckf_interp = np.interp(t_actual, t_ckf, rpm_ckf)
    rmse = np.sqrt(np.mean((rpm_actual - rpm_ckf_interp) ** 2))

    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(t_actual, rpm_actual, color=C_ACTUAL, linewidth=0.8, alpha=0.7, label='Measured RPM')
    ax.plot(t_ckf, rpm_ckf, color=C_CKF, linewidth=0.8, alpha=0.8, label='Bag Estimated RPM')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('RPM')
    ax.set_title(f'{rpm_label} / Test {test_idx} — RPM (RMSE = {rmse:.2f} RPM)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(output_dir / f'{rpm_label}_test{test_idx}_rpm.png')
    plt.close(fig)
    return rmse


def plot_acc_comparison(data, t0, rpm_label, test_idx, output_dir):
    t_actual = data['actual_rpm']['t'] - t0
    rpm_actual = data['actual_rpm']['rpm']
    t_diff, acc_diff = compute_numerical_acceleration(t_actual, rpm_actual)
    t_ckf = data['rotor_state']['t'] - t0
    acc_ckf = data['rotor_state']['acceleration']

    acc_ckf_interp = np.interp(t_diff, t_ckf, acc_ckf)
    rmse = np.sqrt(np.mean((acc_diff - acc_ckf_interp) ** 2))

    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.plot(t_diff, acc_diff, color=C_ACTUAL, linewidth=0.5, alpha=0.5,
            label='Numerical diff (measured)')
    ax.plot(t_ckf, acc_ckf, color=C_CKF, linewidth=0.8, alpha=0.8,
            label='Bag Estimated')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Acceleration [RPM/s]')
    ax.set_title(f'{rpm_label} / Test {test_idx} — Acceleration (RMSE = {rmse:.2f} RPM/s)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(output_dir / f'{rpm_label}_test{test_idx}_acceleration.png')
    plt.close(fig)
    return rmse


def plot_error_with_3sigma(data, t0, rpm_label, test_idx, output_dir):
    t_actual = data['actual_rpm']['t'] - t0
    rpm_actual = data['actual_rpm']['rpm']
    t_diff, acc_diff = compute_numerical_acceleration(t_actual, rpm_actual)
    t_ckf = data['rotor_state']['t'] - t0
    acc_ckf = data['rotor_state']['acceleration']

    acc_ckf_interp = np.interp(t_diff, t_ckf, acc_ckf)
    error = acc_diff - acc_ckf_interp

    t_cov = data['rotor_state_cov']['t'] - t0
    cov_acc = data['rotor_state_cov']['cov'][:, 1]
    sigma_3 = 3.0 * np.sqrt(np.abs(cov_acc))
    sigma_3_interp = np.interp(t_diff, t_cov, sigma_3)

    within_3sigma = np.sum(np.abs(error) <= sigma_3_interp) / len(error) * 100

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.fill_between(t_diff, -sigma_3_interp, sigma_3_interp,
                    color=C_3SIGMA, alpha=0.3, label=r'$\pm 3\sigma$ bound')
    ax.plot(t_diff, error, color=C_ERROR, linewidth=0.5, alpha=0.7, label='Acc. error')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Error [RPM/s]')
    ax.set_title(f'{rpm_label} / Test {test_idx} — Acc. Error with '
                 r'$\pm 3\sigma$'
                 f' ({within_3sigma:.1f}% within bound)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(output_dir / f'{rpm_label}_test{test_idx}_error_3sigma.png')
    plt.close(fig)

    print(f'    -> {within_3sigma:.1f}% within 3σ bound')
    return within_3sigma


def plot_error_histogram(data, t0, rpm_label, test_idx, output_dir):
    """Per-test acceleration error histogram."""
    _, acc_err = get_errors(data, t0)
    mu, std = norm.fit(acc_err)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    n, bins, _ = ax.hist(acc_err, bins=80, density=True, color=C_ACTUAL,
                         alpha=0.6, edgecolor='white', linewidth=0.3,
                         label='Error distribution')
    x = np.linspace(bins[0], bins[-1], 300)
    ax.plot(x, norm.pdf(x, mu, std), color=C_CKF, linewidth=1.5,
            label=rf'$\mathcal{{N}}(\mu={mu:.1f},\,\sigma={std:.1f})$')
    ax.axvline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_xlabel('Acceleration Error [RPM/s]')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'{rpm_label} / Test {test_idx} — Acc. Error Distribution')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(output_dir / f'{rpm_label}_test{test_idx}_acc_error_hist.png')
    plt.close(fig)
    return mu, std


def plot_rpm_error_histogram(data, t0, rpm_label, test_idx, output_dir):
    """Per-test RPM error histogram."""
    rpm_err, _ = get_errors(data, t0)
    mu, std = norm.fit(rpm_err)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    n, bins, _ = ax.hist(rpm_err, bins=80, density=True, color=C_ACTUAL,
                         alpha=0.6, edgecolor='white', linewidth=0.3,
                         label='Error distribution')
    x = np.linspace(bins[0], bins[-1], 300)
    ax.plot(x, norm.pdf(x, mu, std), color=C_CKF, linewidth=1.5,
            label=rf'$\mathcal{{N}}(\mu={mu:.1f},\,\sigma={std:.1f})$')
    ax.axvline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_xlabel('RPM Error [RPM]')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'{rpm_label} / Test {test_idx} — RPM Error Distribution')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(output_dir / f'{rpm_label}_test{test_idx}_rpm_error_hist.png')
    plt.close(fig)
    return mu, std


# ============================================================
# Aggregate histograms
# ============================================================
def _plot_aggregate_per_rpm(all_errors, ylabel, xlabel, title_suffix, fname, output_dir):
    """Generic per-RPM aggregate histogram (side-by-side subplots)."""
    rpm_keys = sorted(all_errors.keys())
    n_rpm = len(rpm_keys)

    fig, axes = plt.subplots(1, n_rpm, figsize=(5.5 * n_rpm, 4))
    if n_rpm == 1:
        axes = [axes]

    for ax, rpm in zip(axes, rpm_keys):
        combined = np.concatenate(all_errors[rpm])
        mu, std = norm.fit(combined)

        n, bins, _ = ax.hist(combined, bins=100, density=True, color=C_ACTUAL,
                             alpha=0.55, edgecolor='white', linewidth=0.3,
                             label=f'All tests (n={len(combined)})')
        x = np.linspace(bins[0], bins[-1], 300)
        ax.plot(x, norm.pdf(x, mu, std), color=C_CKF, linewidth=1.5,
                label=rf'$\mathcal{{N}}(\mu={mu:.1f},\,\sigma={std:.1f})$')
        ax.axvline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{rpm} — Aggregate {title_suffix} (Test 1–{len(all_errors[rpm])})')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2,
                  framealpha=0.9, fontsize=9)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(output_dir / fname)
    plt.close(fig)


def _plot_aggregate_all(all_errors, xlabel, title_suffix, fname, output_dir):
    """Generic all-conditions aggregate histogram."""
    rpm_keys = sorted(all_errors.keys())
    rpm_colors = {rpm_keys[0]: C_BAR_4K}
    if len(rpm_keys) > 1:
        rpm_colors[rpm_keys[1]] = C_BAR_5K

    fig, ax = plt.subplots(figsize=(6, 4))
    all_combined = []
    for rpm in rpm_keys:
        combined = np.concatenate(all_errors[rpm])
        all_combined.append(combined)
        ax.hist(combined, bins=100, density=True,
                color=rpm_colors.get(rpm, 'gray'), alpha=0.45,
                edgecolor='white', linewidth=0.3,
                label=f'{rpm} (n={len(combined)})')

    everything = np.concatenate(all_combined)
    mu, std = norm.fit(everything)
    x = np.linspace(np.min(everything), np.max(everything), 300)
    ax.plot(x, norm.pdf(x, mu, std), 'k-', linewidth=1.5,
            label=rf'Overall $\mathcal{{N}}(\mu={mu:.1f},\,\sigma={std:.1f})$')
    ax.axvline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability Density')
    ax.set_title(f'All Conditions — Aggregate {title_suffix}')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=3,
              framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(output_dir / fname)
    plt.close(fig)


def plot_aggregate_histograms(all_acc_errors, all_rpm_errors, output_dir):
    """Plot all 4 aggregate histograms (acc + rpm) × (per-RPM + all)."""
    # Acceleration
    _plot_aggregate_per_rpm(all_acc_errors, 'Probability Density',
                            'Acceleration Error [RPM/s]', 'Acc. Error',
                            'acc_error_hist_per_rpm.png', output_dir)
    _plot_aggregate_all(all_acc_errors, 'Acceleration Error [RPM/s]',
                        'Acc. Error', 'acc_error_hist_all.png', output_dir)
    # RPM
    _plot_aggregate_per_rpm(all_rpm_errors, 'Probability Density',
                            'RPM Error [RPM]', 'RPM Error',
                            'rpm_error_hist_per_rpm.png', output_dir)
    _plot_aggregate_all(all_rpm_errors, 'RPM Error [RPM]',
                        'RPM Error', 'rpm_error_hist_all.png', output_dir)
    print('  Saved aggregate histograms (acc + rpm)')


# ============================================================
# RMSE bar chart
# ============================================================
def plot_rmse_bar_chart(rmse_results, output_dir):
    rpm_keys = sorted(rmse_results.keys())
    num_tests = max(len(rmse_results[k]) for k in rpm_keys)
    colors = {rpm_keys[0]: C_BAR_4K}
    if len(rpm_keys) > 1:
        colors[rpm_keys[1]] = C_BAR_5K

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    x = np.arange(num_tests)
    width = 0.35

    for idx, rpm in enumerate(rpm_keys):
        vals = [r['rpm_rmse'] for r in rmse_results[rpm]]
        offset = (idx - (len(rpm_keys) - 1) / 2) * width
        bars = ax1.bar(x[:len(vals)] + offset, vals, width * 0.9,
                       label=rpm, color=colors.get(rpm, f'C{idx}'), alpha=0.85)
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    ax1.set_xlabel('Test')
    ax1.set_ylabel('RMSE [RPM]')
    ax1.set_title('RPM RMSE')
    ax1.set_xticks(x[:num_tests])
    ax1.set_xticklabels([f'T{i+1}' for i in range(num_tests)])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28),
               ncol=len(rpm_keys), framealpha=0.9)

    for idx, rpm in enumerate(rpm_keys):
        vals = [r['acc_rmse'] for r in rmse_results[rpm]]
        offset = (idx - (len(rpm_keys) - 1) / 2) * width
        bars = ax2.bar(x[:len(vals)] + offset, vals, width * 0.9,
                       label=rpm, color=colors.get(rpm, f'C{idx}'), alpha=0.85)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{v:.1f}', ha='center', va='bottom', fontsize=8)

    ax2.set_xlabel('Test')
    ax2.set_ylabel('RMSE [RPM/s]')
    ax2.set_title('Acceleration RMSE')
    ax2.set_xticks(x[:num_tests])
    ax2.set_xticklabels([f'T{i+1}' for i in range(num_tests)])
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28),
               ncol=len(rpm_keys), framealpha=0.9)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(output_dir / 'rmse_comparison_bar.png')
    plt.close(fig)
    print(f'\n  Saved RMSE bar chart')


# ============================================================
# Summary: print + txt + csv  (pooled errors → RMSE, Mean, Std)
# ============================================================
def compute_pooled_stats(error_lists):
    """Concatenate all error arrays → RMSE, Mean, Std from the pooled errors."""
    pooled = np.concatenate(error_lists)
    return rmse_from_error(pooled), np.mean(pooled), np.std(pooled)


def print_and_save_summary(per_test_stats, all_rpm_errors, all_acc_errors,
                           sigma_results, output_dir):
    """
    Print and save summary.
    per_test_stats: {rpm: [(rpm_rmse, acc_rmse, 3sig, rpm_mu, rpm_std, acc_mu, acc_std), ...]}
    all_rpm_errors / all_acc_errors: {rpm: [err_array_t1, ...]}
    """
    lines = []

    def log(s=''):
        print(s)
        lines.append(s)

    hdr1 = (f'{"":>22} {"RPM RMSE":>10} {"RPM Mean":>10} {"RPM Std":>10}'
            f' {"Acc RMSE":>10} {"Acc Mean":>10} {"Acc Std":>10} {"3σ cov":>8}')
    hdr2 = (f'{"":>22} {"[RPM]":>10} {"[RPM]":>10} {"[RPM]":>10}'
            f' {"[RPM/s]":>10} {"[RPM/s]":>10} {"[RPM/s]":>10} {"[%]":>8}')

    log()
    log('=' * 102)
    log(hdr1)
    log(hdr2)
    log('-' * 102)

    csv_rows = []
    csv_header = ['Condition', 'Test',
                  'RPM_RMSE [RPM]', 'RPM_Mean [RPM]', 'RPM_Std [RPM]',
                  'Acc_RMSE [RPM/s]', 'Acc_Mean [RPM/s]', 'Acc_Std [RPM/s]',
                  '3sigma_coverage [%]']

    for rpm in sorted(per_test_stats.keys()):
        tests = per_test_stats[rpm]
        sigs = sigma_results[rpm]

        for i, (r_rmse, a_rmse, sig, r_mu, r_std, a_mu, a_std) in enumerate(tests):
            log(f'  {rpm}/test{i+1:>2}       '
                f'{r_rmse:>10.4f} {r_mu:>10.4f} {r_std:>10.4f}'
                f' {a_rmse:>10.4f} {a_mu:>10.4f} {a_std:>10.4f} {sig:>8.1f}')
            csv_rows.append([rpm, f'test{i+1}',
                             f'{r_rmse:.6f}', f'{r_mu:.6f}', f'{r_std:.6f}',
                             f'{a_rmse:.6f}', f'{a_mu:.6f}', f'{a_std:.6f}',
                             f'{sig:.2f}'])

        # Per-RPM pooled stats
        rpm_rmse_p, rpm_mean_p, rpm_std_p = compute_pooled_stats(all_rpm_errors[rpm])
        acc_rmse_p, acc_mean_p, acc_std_p = compute_pooled_stats(all_acc_errors[rpm])
        sig_mean = np.mean(sigs)

        log(f'  {rpm} Pooled       '
            f'{rpm_rmse_p:>10.4f} {rpm_mean_p:>10.4f} {rpm_std_p:>10.4f}'
            f' {acc_rmse_p:>10.4f} {acc_mean_p:>10.4f} {acc_std_p:>10.4f} {sig_mean:>8.1f}')
        log('-' * 102)

        csv_rows.append([rpm, 'Pooled',
                         f'{rpm_rmse_p:.6f}', f'{rpm_mean_p:.6f}', f'{rpm_std_p:.6f}',
                         f'{acc_rmse_p:.6f}', f'{acc_mean_p:.6f}', f'{acc_std_p:.6f}',
                         f'{sig_mean:.2f}'])

    # Overall pooled
    all_rpm_flat = [e for v in all_rpm_errors.values() for e in v]
    all_acc_flat = [e for v in all_acc_errors.values() for e in v]
    rpm_rmse_o, rpm_mean_o, rpm_std_o = compute_pooled_stats(all_rpm_flat)
    acc_rmse_o, acc_mean_o, acc_std_o = compute_pooled_stats(all_acc_flat)
    all_sig = [s for v in sigma_results.values() for s in v]

    log(f'  {"Overall Pooled":>20} '
        f'{rpm_rmse_o:>10.4f} {rpm_mean_o:>10.4f} {rpm_std_o:>10.4f}'
        f' {acc_rmse_o:>10.4f} {acc_mean_o:>10.4f} {acc_std_o:>10.4f} {np.mean(all_sig):>8.1f}')
    log('=' * 102)

    csv_rows.append(['Overall', 'Pooled',
                     f'{rpm_rmse_o:.6f}', f'{rpm_mean_o:.6f}', f'{rpm_std_o:.6f}',
                     f'{acc_rmse_o:.6f}', f'{acc_mean_o:.6f}', f'{acc_std_o:.6f}',
                     f'{np.mean(all_sig):.2f}'])

    # Save .txt
    txt_path = output_dir / 'ckf_summary.txt'
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Saved TXT: {txt_path}')

    # Save .csv
    csv_path = output_dir / 'ckf_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)
    print(f'  Saved CSV: {csv_path}')


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Bag Acceleration Estimation Analysis')

    parser.add_argument('base_dir', type=str,
                        help='Base directory (e.g., Bag)')
    parser.add_argument('--rpm-folders', nargs='+', type=str,
                        default=['4000RPM', '5000RPM'],
                        help='RPM folder names (default: 4000RPM 5000RPM)')
    parser.add_argument('--num-tests', type=int, default=5,
                        help='Number of test bags per RPM (default: 5)')
    parser.add_argument('--output-dir', type=str, default='./result',
                        help='Output directory (default: ./result)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = extract_all_bags(
        base_dir=args.base_dir,
        rpm_folders=args.rpm_folders,
        num_tests=args.num_tests
    )

    # Storage
    per_test_stats = {}   # {rpm: [(rpm_rmse, acc_rmse, 3sig, rpm_mu, rpm_std, acc_mu, acc_std)]}
    sigma_results = {}    # {rpm: [within_3sigma, ...]}
    all_rpm_errors = {}   # {rpm: [err_array_t1, ...]}
    all_acc_errors = {}   # {rpm: [err_array_t1, ...]}
    rmse_results = {}     # for bar chart: {rpm: [{rpm_rmse, acc_rmse}]}

    for rpm, tests in all_data.items():
        print(f'\n=== {rpm}: {len(tests)} bags ===')
        per_test_stats[rpm] = []
        sigma_results[rpm] = []
        all_rpm_errors[rpm] = []
        all_acc_errors[rpm] = []
        rmse_results[rpm] = []

        for i, data in enumerate(tests):
            if len(data['actual_rpm']['t']) < 3 or len(data['rotor_state']['t']) < 3:
                print(f'  [SKIP] test{i+1}: not enough data')
                continue

            t0 = min(data['actual_rpm']['t'][0], data['rotor_state']['t'][0])
            print(f'  test{i+1}:')

            # Get error arrays
            rpm_err, acc_err = get_errors(data, t0)
            all_rpm_errors[rpm].append(rpm_err)
            all_acc_errors[rpm].append(acc_err)

            # Per-test stats from the error arrays
            r_rmse = rmse_from_error(rpm_err)
            a_rmse = rmse_from_error(acc_err)
            r_mu, r_std = np.mean(rpm_err), np.std(rpm_err)
            a_mu, a_std = np.mean(acc_err), np.std(acc_err)

            # Plots
            plot_rpm_comparison(data, t0, rpm, i + 1, output_dir)
            plot_acc_comparison(data, t0, rpm, i + 1, output_dir)
            within_3s = plot_error_with_3sigma(data, t0, rpm, i + 1, output_dir)
            plot_error_histogram(data, t0, rpm, i + 1, output_dir)
            plot_rpm_error_histogram(data, t0, rpm, i + 1, output_dir)

            per_test_stats[rpm].append((r_rmse, a_rmse, within_3s,
                                        r_mu, r_std, a_mu, a_std))
            sigma_results[rpm].append(within_3s)
            rmse_results[rpm].append({'rpm_rmse': r_rmse, 'acc_rmse': a_rmse})

            print(f'    -> RPM  RMSE={r_rmse:.4f}, Mean={r_mu:.4f}, Std={r_std:.4f}')
            print(f'    -> Acc  RMSE={a_rmse:.4f}, Mean={a_mu:.4f}, Std={a_std:.4f}')

    # RMSE bar chart
    plot_rmse_bar_chart(rmse_results, output_dir)

    # Aggregate histograms (acc + rpm)
    print('\n=== Aggregate Histograms ===')
    plot_aggregate_histograms(all_acc_errors, all_rpm_errors, output_dir)

    # Summary (console + txt + csv)
    print_and_save_summary(per_test_stats, all_rpm_errors, all_acc_errors,
                           sigma_results, output_dir)

    print(f'\nAll plots saved to {output_dir}/')


if __name__ == "__main__":
    main()