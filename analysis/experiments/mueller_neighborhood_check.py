"""Compare native and locally averaged matrices at the same sampled pixels."""
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
WINDOWS = (1, 3, 7)


def local_mean_at(array, y, x, size):
    radius = size // 2
    values = np.zeros(len(y), dtype=np.float64)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            values += array[y + dy, x + dx]
    return values / (size * size)


def run(rois, stride, mask_run, name):
    started = time.perf_counter()
    out = ROOT / 'analysis' / 'runs' / name
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for roi in rois:
        source = ROOT / '类器官' / str(roi) / 'FinalMM.mat'
        data = loadmat(source, variable_names=NAMES)
        mask = np.load(ROOT / mask_run / f'roi_{roi}_prepared.npz')['mask']
        h, w = data['FinalM11'].shape
        ys, xs = np.arange(0, h, stride), np.arange(0, w, stride)
        selected = mask[(ys * mask.shape[0] // h)[:, None], (xs * mask.shape[1] // w)[None, :]]
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        y, x = yy[selected], xx[selected]
        margin = max(WINDOWS) // 2
        interior = (y >= margin) & (y < h - margin) & (x >= margin) & (x < w - margin)
        y, x = y[interior], x[interior]
        raw = np.stack([data[n][y, x] for n in NAMES], axis=-1).reshape(-1, 4, 4)
        sample_data = {'y': y, 'x': x}
        for size in WINDOWS:
            if size == 1:
                matrices = raw
            else:
                matrices = np.stack([local_mean_at(data[n], y, x, size) for n in NAMES], axis=-1).reshape(-1, 4, 4)
            result = muellerkit.physical_realizability(matrices)
            difference = np.linalg.norm(matrices - raw, axis=(1, 2)) / np.linalg.norm(raw, axis=(1, 2))
            rows.append({'roi': roi, 'window': f'{size}x{size}', 'sample_count': len(raw),
                         'valid_count': int(result.is_realizable.sum()),
                         'valid_fraction': float(result.is_realizable.mean()),
                         'minimum_eigenvalue_median': float(np.median(result.minimum_eigenvalue)),
                         'relative_matrix_change_median': float(np.median(difference)),
                         'relative_matrix_change_p95': float(np.quantile(difference, .95))})
            sample_data[f'valid_{size}'] = result.is_realizable
            sample_data[f'minimum_eigenvalue_{size}'] = result.minimum_eigenvalue
            sample_data[f'relative_change_{size}'] = difference
        np.savez_compressed(out / f'roi_{roi}_sample.npz', **sample_data)
        r = rows[-3:]
        print(f"ROI {roi}: pass 1x1={r[0]['valid_fraction']:.2%}, 3x3={r[1]['valid_fraction']:.2%}, 7x7={r[2]['valid_fraction']:.2%}", flush=True)
    with (out / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    counts = [sum(r['sample_count'] for r in rows if r['window'] == f'{size}x{size}') for size in WINDOWS]
    fraction = [sum(r['valid_count'] for r in rows if r['window'] == f'{size}x{size}') / n
                for size, n in zip(WINDOWS, counts)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(s) + 'x' + str(s) for s in WINDOWS], np.asarray(fraction) * 100, color=['#547AA3','#55A57D','#DC9950'])
    ax.set_ylabel('Strict physicality pass rate (%)')
    ax.set_xlabel('Native-pixel averaging window')
    ax.set_title('Same sampled locations; draft masks; I,Q,U,V convention')
    for i, value in enumerate(fraction): ax.text(i, value * 100, f'{value:.2%}', ha='center', va='bottom')
    fig.tight_layout(); fig.savefig(out / 'physicality_by_window.png', dpi=150); plt.close(fig)
    meta = {'rois': rois, 'stride_native_pixels': stride, 'windows': list(WINDOWS),
            'mask_source': mask_run, 'mask_status': 'auto_draft_unreviewed',
            'selection': 'Identical sampled mask locations for all windows; exclude image margins of 3 native pixels',
            'averaging': 'Unweighted square neighborhood on each Mueller element; may mix tissue, lumen and background',
            'relative_change': 'Frobenius norm of (averaged matrix - native matrix) / Frobenius norm of native matrix',
            'physicality_tolerances': {'atol': 1e-12, 'rtol': 1e-10},
            'assumed_stokes_order': ['I','Q','U','V'], 'muellerkit_version': importlib.metadata.version('muellerkit'),
            'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'seconds': time.perf_counter() - started}
    (out/'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    lines=['局部平均对矩阵物理性检查的影响',
           f'ROI={rois}；每{stride}个原始像素取样；组织掩膜尚未人工核对。',
           '每种窗口使用相同的取样点，排除距图像边界小于3像素的位置。',
           '窗口 | 抽样数 | 严格物理性检查通过数 | 抽样加权通过率']
    for size,n,rate in zip(WINDOWS,counts,fraction):
        lines.append(f'{size}×{size} | {n} | {sum(r["valid_count"] for r in rows if r["window"] == f"{size}x{size}")} | {rate:.2%}')
    lines.extend(['局部平均按16个矩阵元素分别计算；7×7可能跨越组织、腔隙或背景。',
                  '通过率上升只能说明该严格数值条件的变化，不能证明平均后矩阵已校准，也不能替代结构边界检查。',
                  '逐ROI的矩阵相对改变量中位数和95%分位数见summary.csv。'])
    (out/'结果说明.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Mueller averaging audit</title>'
        '<style>body{font:16px sans-serif;max-width:900px;margin:30px auto}img{max-width:100%}</style>'
        '<h1>矩阵局部平均对物理性检查的影响</h1><p>不同窗口使用相同采样点。平均可能混合组织、腔隙与背景，结果仅供诊断。</p>'
        '<img src="physicality_by_window.png"><p>逐ROI数值与矩阵改变量见 summary.csv 和 结果说明.txt。</p>', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    print(f'OUTPUT={out}', flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rois', nargs='+', type=int, default=list(range(1,8)))
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--mask-run', default='analysis/runs/ppt_reference_4channels_20261031')
    parser.add_argument('--run-name')
    args=parser.parse_args()
    if args.stride < 1 or not args.rois or len(args.rois) != len(set(args.rois)):
        parser.error('Use a positive stride and distinct ROI numbers')
    name=args.run_name or datetime.now().strftime('mueller_neighborhood_%Y%m%d_%H%M%S')
    if Path(name).name != name or name in {'.','..'}:
        parser.error('run-name must be a single directory name')
    run(args.rois,args.stride,args.mask_run,name)


if __name__ == '__main__':
    main()
