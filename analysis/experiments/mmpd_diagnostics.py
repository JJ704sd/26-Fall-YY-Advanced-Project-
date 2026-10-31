"""Audit physicality and Lu-Chipman parameters on a reproducible ROI sample.

The unfiltered decomposition values are numerical diagnostics, not validated
depolarization or retardance measurements. Input masks remain unreviewed.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import muellerkit
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
NAMES = [f'FinalM{i}{j}' for i in range(1, 5) for j in range(1, 5)]


def run(rois, stride, mask_run, run_name):
    start = time.perf_counter()
    out = ROOT / 'analysis' / 'runs' / run_name
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for roi in rois:
        path = ROOT / '类器官' / str(roi) / 'FinalMM.mat'
        data = loadmat(path, variable_names=NAMES)
        prepared = np.load(ROOT / mask_run / f'roi_{roi}_prepared.npz')
        mask = prepared['mask']
        height, width = data['FinalM11'].shape
        y, x = np.arange(0, height, stride), np.arange(0, width, stride)
        chosen = mask[(y * mask.shape[0] // height)[:, None],
                      (x * mask.shape[1] // width)[None, :]]
        matrices = np.stack([data[name][::stride, ::stride][chosen]
                             for name in NAMES], axis=-1).reshape(-1, 4, 4)
        if not np.isfinite(matrices).all():
            raise ValueError(f'ROI {roi}: nonfinite matrix in sampled tissue')
        physicality = muellerkit.physical_realizability(matrices)
        dep, ret, dia = muellerkit.decompose_depolarizer_retarder_diattenuator(matrices)
        params = muellerkit.extract_parameters(dep, ret, dia)
        residual = np.max(np.abs(dep @ ret @ dia - matrices), axis=(1, 2))
        ret_block = ret[:, 1:, 1:]
        orthogonality_error = np.linalg.norm(
            np.swapaxes(ret_block, -1, -2) @ ret_block - np.eye(3), axis=(1, 2))
        purity_squared = (np.sum(matrices * matrices, axis=(1, 2)) - matrices[:, 0, 0] ** 2) / (3 * matrices[:, 0, 0] ** 2)
        purity_index = np.sqrt(np.maximum(purity_squared, 0))
        eig = physicality.minimum_eigenvalue
        valid = physicality.is_realizable
        valid_delta = params.total_depolarization[valid]
        valid_retardance = params.linear_retardance[valid]
        row = {'roi': roi, 'sample_count': len(matrices), 'valid_count': int(valid.sum()),
               'valid_fraction': float(valid.mean()), 'min_eigen_median': float(np.median(eig)),
               'min_eigen_p05': float(np.quantile(eig, .05)),
               'min_eigen_p95': float(np.quantile(eig, .95)),
               'D_median_unfiltered': float(np.median(params.diattenuation)),
               'delta_rad_median_unfiltered': float(np.median(params.linear_retardance)),
               'depolarization_median_unfiltered': float(np.median(params.total_depolarization)),
               'delta_rad_median_valid_only': float(np.median(valid_retardance)) if len(valid_retardance) else None,
               'depolarization_median_valid_only': float(np.median(valid_delta)) if len(valid_delta) else None,
               'unfiltered_depolarization_below_zero_fraction': float(np.mean(params.total_depolarization < 0)),
               'purity_index_above_one_fraction': float(np.mean(purity_index > 1)),
               'retarder_orthogonality_error_above_001_fraction': float(np.mean(orthogonality_error > .001)),
               'retarder_orthogonality_error_p95': float(np.quantile(orthogonality_error, .95)),
               'reconstruction_abs_error_p99': float(np.quantile(residual, .99))}
        rows.append(row)
        np.savez_compressed(out / f'roi_{roi}_sample.npz', y=y, x=x, selected=chosen,
                            sample_matrices=matrices, physical_valid=valid, minimum_eigenvalue=eig,
                            purity_index=purity_index, retarder_orthogonality_error=orthogonality_error,
                            diagnostic_D=params.diattenuation,
                            diagnostic_delta_rad=params.linear_retardance,
                            diagnostic_depolarization=params.total_depolarization)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
        values = [eig, valid.astype(float), params.linear_retardance]
        titles = ['Minimum coherency eigenvalue', 'Physicality pass (1=yes)', 'Unfiltered linear retardance (rad)']
        for ax, value, title in zip(axes, values, titles):
            image = np.full(chosen.shape, np.nan)
            image[chosen] = value
            im = ax.imshow(image, cmap='viridis', interpolation='nearest')
            ax.set_title(title, fontsize=9); ax.axis('off'); fig.colorbar(im, ax=ax, fraction=.045)
        fig.suptitle(f'ROI {roi}; sampled every {stride} px; draft tissue mask; pass={valid.mean():.2%}', fontsize=10)
        fig.tight_layout(); fig.savefig(out / f'roi_{roi}_physicality.png', dpi=130); plt.close(fig)
        print(f"ROI {roi}: {valid.sum()}/{len(valid)} physicality pass; median eigenvalue={np.median(eig):.6f}", flush=True)
    with (out / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    meta = {'rois': rois, 'stride_native_pixels': stride, 'mask_source': mask_run,
            'mask_status': 'auto_draft_unreviewed', 'muellerkit_version': importlib.metadata.version('muellerkit'),
            'assumed_stokes_order': ['I', 'Q', 'U', 'V'],
            'physicality_tolerances': {'atol': 1e-12, 'rtol': 1e-10},
            'decomposition_order': 'M_delta @ M_R @ M_D',
            'matrix_source': 'FinalM11..FinalM44 as stored; muellerkit accounts for M11 in the decomposition',
            'orthogonality_diagnostic_threshold': .001,
            'warning': 'Strict physicality is a mathematical criterion under the stated convention. Measurement noise, calibration, and Stokes convention require independent investigation. Unfiltered MMPD values are diagnostic only.',
            'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'seconds': time.perf_counter() - start}
    (out / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    content = ['<!doctype html><meta charset="utf-8"><title>Mueller physicality audit</title>',
               '<style>body{font:16px sans-serif;max-width:1100px;margin:30px auto}img{max-width:100%}</style>',
               '<h1>缪勒矩阵抽样与分解诊断</h1>',
               f'<p>每隔{stride}个原始像素抽样，组织掩膜尚未人工核对。严格物理性检查使用 I,Q,U,V 约定；'
               '原始数值的分解结果仅用于数值诊断，不能直接作为已验证的退偏或延迟。</p>']
    for roi in rois:
        content.append(f'<h2>ROI {roi}</h2><img src="roi_{roi}_physicality.png">')
    (out / 'report.html').write_text(''.join(content), encoding='utf-8')
    print(f'OUTPUT={out}', flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rois', nargs='+', type=int, default=list(range(1, 8)))
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--mask-run', default='analysis/runs/roi1_7_exploratory_20261030_1')
    parser.add_argument('--run-name', default=None)
    args = parser.parse_args()
    if args.stride < 1 or not args.rois or len(args.rois) != len(set(args.rois)):
        parser.error('Use a positive stride and distinct ROI numbers')
    name = args.run_name or datetime.now().strftime('mmpd_audit_%Y%m%d_%H%M%S')
    if Path(name).name != name or name in {'.', '..'}:
        parser.error('run-name must be a single directory name')
    run(args.rois, args.stride, args.mask_run, name)


if __name__ == '__main__':
    main()
