"""Summarize the controlled seven-ROI method and feature experiments."""
import argparse
import csv
from collections import defaultdict
from datetime import datetime
import hashlib
from pathlib import Path
import json
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = ['ppt_reference_4channels_20261031', 'ppt_totalD_5channels_20261031',
                'ppt_similarity_edges_20261031']
METHODS = ['baseline', 'propagation', 'gae', 'dgi']


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def experiment_name(cfg):
    if cfg['weighted']:
        return 'core4_weighted'
    return 'core4_plus_D' if cfg.get('include_total_diattenuation', False) else 'core4'


def run(names, audit_name, neighborhood_name, output_name, ppt_file):
    paths = [ROOT / 'analysis' / 'runs' / name for name in names]
    ppt_path = ROOT / ppt_file
    ppt_sha256 = hashlib.sha256(ppt_path.read_bytes()).hexdigest()
    configs = [json.loads((p / 'config.json').read_text(encoding='utf-8')) for p in paths]
    comparable_keys = ['rois', 'normalization', 'max_side', 'n_segments', 'compactness',
                       'hops', 'k', 'epochs', 'seeds', 'device', 'mask_dir', 'methods']
    for key in comparable_keys:
        if any(cfg[key] != configs[0][key] for cfg in configs[1:]):
            raise ValueError(f'Experiments differ in {key}; do not compare as a controlled run')
    labels = [experiment_name(cfg) for cfg in configs]
    if labels != ['core4', 'core4_plus_D', 'core4_weighted']:
        raise ValueError('Expected core4, core4 + D, and core4 with weighted edges')
    output = ROOT / 'analysis' / 'reports' / output_name
    output.mkdir(parents=True, exist_ok=False)
    equality = []
    for roi in configs[0]['rois']:
        arrays = [np.load(p / f'roi_{roi}_prepared.npz') for p in paths]
        for label, a in zip(labels[1:], arrays[1:]):
            equality.append({'roi': roi, 'comparison': label,
                'same_mask': bool(np.array_equal(arrays[0]['mask'], a['mask'])),
                'same_superpixels': bool(np.array_equal(arrays[0]['superpixels'], a['superpixels'])),
                'same_core4_maps': bool(np.array_equal(arrays[0]['parameters'], a['parameters'][..., :4]))})
    write_csv(output / 'data_alignment.csv', equality)
    if not all(row['same_mask'] and row['same_superpixels'] and row['same_core4_maps'] for row in equality):
        raise ValueError('Segmentation or input maps differ; inspect data_alignment.csv before comparison')
    per_roi, aggregates = [], []
    seed_count = len(configs[0]['seeds'])
    pair_count = seed_count * (seed_count - 1) // 2
    for label, path in zip(labels, paths):
        metrics = read_csv(path / 'metrics.csv')
        stability = read_csv(path / 'stability_ari.csv')
        grouped_metrics, grouped_ari = defaultdict(list), defaultdict(list)
        for row in metrics:
            grouped_metrics[(int(row['roi']), row['method'])].append(row)
        for row in stability:
            grouped_ari[(int(row['roi']), row['method'])].append(float(row['ARI']))
        for method in METHODS:
            for roi in configs[0]['rois']:
                rows = grouped_metrics[(roi, method)]
                ari = grouped_ari[(roi, method)]
                if len(rows) != seed_count or len(ari) != pair_count:
                    raise ValueError(f'Missing metrics for {label} ROI {roi} {method}')
                scores = [float(row['silhouette_core4']) for row in rows if row['silhouette_core4']]
                if len(scores) != len(rows):
                    raise ValueError(f'Undefined silhouette for {label} ROI {roi} {method}')
                per_roi.append({'experiment': label, 'roi': roi, 'method': method,
                    'silhouette_core4_mean': mean(scores), 'stability_ari_mean': mean(ari),
                    'mixed_neighbor_fraction_mean': mean(float(row['mixed_neighbor_fraction']) for row in rows),
                    'connected_components_mean': mean(float(row['connected_components']) for row in rows)})
            rr = [row for row in per_roi if row['experiment'] == label and row['method'] == method]
            a = [row['stability_ari_mean'] for row in rr]
            s = [row['silhouette_core4_mean'] for row in rr]
            b = [row['mixed_neighbor_fraction_mean'] for row in rr]
            aggregates.append({'experiment': label, 'method': method,
                'silhouette_core4_mean_roi': mean(s), 'silhouette_core4_min_roi': min(s),
                'silhouette_core4_max_roi': max(s), 'stability_ari_mean_roi': mean(a),
                'stability_ari_min_roi': min(a), 'stability_ari_max_roi': max(a),
                'mixed_neighbor_fraction_mean_roi': mean(b)})
    write_csv(output / 'comparison_per_roi.csv', per_roi)
    write_csv(output / 'comparison_summary.csv', aggregates)
    node_rows = read_csv(paths[1] / 'nodes.csv')
    corr = []
    for roi in configs[0]['rois']:
        rows = [row for row in node_rows if int(row['roi']) == roi]
        dl = np.asarray([float(row['DL_mean']) for row in rows])
        d = np.asarray([float(row['D_mean']) for row in rows])
        corr.append({'roi': roi, 'nodes': len(rows), 'pearson_DL_mean_vs_D_mean': float(np.corrcoef(dl, d)[0, 1])})
    write_csv(output / 'diattenuation_feature_correlation.csv', corr)
    audit = read_csv(ROOT / 'analysis' / 'runs' / audit_name / 'summary.csv')
    audit_metadata = json.loads((ROOT / 'analysis' / 'runs' / audit_name / 'metadata.json').read_text(encoding='utf-8'))
    neighborhood = read_csv(ROOT / 'analysis' / 'runs' / neighborhood_name / 'summary.csv')
    audit_alignment = []
    for roi in configs[0]['rois']:
        reference = np.load(paths[0] / f'roi_{roi}_prepared.npz')['mask']
        audit_mask = np.load(ROOT / audit_metadata['mask_source'] / f'roi_{roi}_prepared.npz')['mask']
        a = np.load(ROOT / 'analysis' / 'runs' / audit_name / f'roi_{roi}_sample.npz')
        b = np.load(ROOT / 'analysis' / 'runs' / neighborhood_name / f'roi_{roi}_sample.npz')
        yy, xx = np.meshgrid(a['y'], a['x'], indexing='ij')
        audit_ids = yy[a['selected']].astype(np.int64) * 2**32 + xx[a['selected']]
        neighborhood_ids = b['y'].astype(np.int64) * 2**32 + b['x']
        indexes = np.searchsorted(audit_ids, neighborhood_ids)
        same_coordinates = bool(np.all(indexes < len(audit_ids)) and
                                np.array_equal(audit_ids[indexes], neighborhood_ids))
        same_validity = bool(same_coordinates and np.array_equal(a['physical_valid'][indexes], b['valid_1']))
        audit_alignment.append({'roi': roi, 'same_tissue_mask': bool(np.array_equal(reference, audit_mask)),
                                'neighborhood_samples_are_subset': same_coordinates,
                                'same_raw_physicality_flags': same_validity})
    write_csv(output / 'audit_alignment.csv', audit_alignment)
    if not all(row['same_tissue_mask'] and row['neighborhood_samples_are_subset'] and
               row['same_raw_physicality_flags'] for row in audit_alignment):
        raise ValueError('Matrix audits use different masks or sample coordinates; inspect audit_alignment.csv')
    valid_total = sum(int(row['valid_count']) for row in audit)
    samples_total = sum(int(row['sample_count']) for row in audit)
    orth_fraction = sum(int(row['sample_count']) * float(row['retarder_orthogonality_error_above_001_fraction'])
                        for row in audit) / samples_total
    neighborhood_rates = {}
    for size in ('1x1', '3x3', '7x7'):
        selected = [row for row in neighborhood if row['window'] == size]
        count = sum(int(row['sample_count']) for row in selected)
        neighborhood_rates[size] = (sum(int(row['valid_count']) for row in selected) / count, count)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    positions = np.arange(len(METHODS))
    colors = ['#3465a4', '#e0933c', '#4b9b68']
    titles = {'core4': 'Core 4 / uniform edges', 'core4_plus_D': 'Core 4 + D / uniform edges',
              'core4_weighted': 'Core 4 / similarity-weighted edges'}
    for j, label in enumerate(labels):
        rows = [next(r for r in aggregates if r['experiment'] == label and r['method'] == m) for m in METHODS]
        for ax, field, low, high in [
            (axes[0], 'silhouette_core4_mean_roi', 'silhouette_core4_min_roi', 'silhouette_core4_max_roi'),
            (axes[1], 'stability_ari_mean_roi', 'stability_ari_min_roi', 'stability_ari_max_roi')]:
            mid = np.array([r[field] for r in rows]); lo = np.array([r[low] for r in rows]); hi = np.array([r[high] for r in rows])
            ax.bar(positions + (j-1)*.23, mid, width=.23, color=colors[j], label=titles[label],
                   yerr=[mid-lo, hi-mid], capsize=2, error_kw={'linewidth': .7})
    axes[0].set_title('Silhouette in common core feature space')
    axes[1].set_title('ARI between random seeds')
    for ax in axes:
        ax.set_xticks(positions, METHODS); ax.axhline(0, color='#444', linewidth=.6)
        ax.set_ylabel('Mean of ROI means; whiskers show ROI range')
    axes[1].legend(fontsize=7, loc='lower left')
    fig.suptitle('Exploratory comparison; 7 ROIs, 5 seeds; unreviewed masks', fontsize=11)
    fig.tight_layout(); fig.savefig(output / 'method_comparison.png', dpi=150); plt.close(fig)
    lines = [
        '乳腺癌类器官偏振分析 对照实验记录',
        '',
        f'项目PPT：{ppt_path}；SHA-256：{ppt_sha256}',
        f"数据与设置：{len(configs[0]['rois'])}个ROI；自动掩膜草稿；最长边{configs[0]['max_side']}像素；"
        f"每个ROI约{configs[0]['n_segments']}个超像素；K={configs[0]['k']}；{seed_count}个随机种子；"
        f"GAE/DGI各{configs[0]['epochs']}轮；设备={configs[0]['device']}。",
        '三组分别为：core4=原4参数+等权空间边；core4_plus_D=增加总二向色性D+等权空间边；core4_weighted=原4参数+偏振相似度边权。',
        '三组逐ROI核对后，组织掩膜、超像素和4个基础参数图均逐像素相同；对应关系见 data_alignment.csv。',
        f'每种方法先在每个ROI内对{seed_count}个种子的轮廓系数取均值，对{pair_count}组种子对的ARI取均值，再计算各ROI的描述性平均。ROI之间是否生物学独立尚不明确，这些统计量不是置信区间。',
        '共同特征空间为4个原始偏振参数的均值和标准差，共8维；加入D的实验也使用同一8维空间评价silhouette_core4。',
        '',
        '实验组 | 方法 | 共同空间轮廓系数均值 | 随机种子ARI均值 | 不同类相邻比例均值',
    ]
    for row in aggregates:
        lines.append(f"{row['experiment']} | {row['method']} | {row['silhouette_core4_mean_roi']:.4f} | "
                     f"{row['stability_ari_mean_roi']:.4f} | {row['mixed_neighbor_fraction_mean_roi']:.4f}")
    lines += [
        '',
        f"完整缪勒矩阵诊断：在自动掩膜内按原始图像每{audit_metadata['stride_native_pixels']}像素抽样，共{samples_total}点；"
        f'采用 I,Q,U,V 与严格的相干矩阵半正定条件，{valid_total}点通过（{valid_total/samples_total:.2%}）。',
        '检查条件、逐ROI结果和样本见对应 mmpd_audit 目录。该结果依赖掩膜、采样、Stokes约定和容差，不等同于整个数据集的比例；测量噪声或标定等原因仍需排查。',
        f"排除靠近图像边缘的采样点后，在同一批{neighborhood_rates['1x1'][1]}个取样点上，原始、3×3平均、7×7平均的严格检查通过率分别为"
        f"{neighborhood_rates['1x1'][0]:.2%}、{neighborhood_rates['3x3'][0]:.2%}、{neighborhood_rates['7x7'][0]:.2%}。"
        '局部平均可能混合组织、腔隙和背景，不能证明数据已经校准。',
        f'未经物理性筛选的分解中，{orth_fraction:.2%}的抽样矩阵得到的延迟子矩阵正交性误差超过0.001（此阈值仅作数值诊断）。矩阵重构误差小并不意味着分解具有可靠的物理解释。',
        '诊断程序的未经筛选Lu-Chipman数值仅用于观察；当前未将退偏与线性相位延迟并入模型或视为组织结论。',
        f"在这批数据的超像素均值上，D与DL逐ROI的Pearson相关系数为{min(r['pearson_DL_mean_vs_D_mean'] for r in corr):.4f}至{max(r['pearson_DL_mean_vs_D_mean'] for r in corr):.4f}，其新增信息需要在不同样本中继续验证。",
        '类别编号不对应具体病理结构；掩膜低纹理部分和腔隙需要人工核对。面积与边界指标依赖分析网格和掩膜，不推断患者间泛化或临床效果。',
        '',
        '资料：DGI 原文 https://arxiv.org/abs/1809.10341',
        'Lu-Chipman 分解 https://opg.optica.org/josaa/abstract.cfm?uri=josaa-13-5-1106',
        '物理可实现性检查 https://academic.oup.com/bioinformatics/article/40/7/btae348/7687046',
        '依赖实现与版本 https://pypi.org/project/muellerkit/0.1.2/',
    ]
    (output / '实验结果说明.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (output / 'sources.json').write_text(json.dumps({'runs': names, 'audit': audit_name, 'neighborhood': neighborhood_name,
        'config_controls': comparable_keys, 'experiments': labels,
        'ppt_path': str(ppt_path), 'ppt_sha256': ppt_sha256,
        'report_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n'.join(lines), flush=True)
    print(f'OUTPUT={output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs=3, default=DEFAULT_RUNS)
    parser.add_argument('--audit', default='mmpd_audit_7roi_20261031_v2')
    parser.add_argument('--neighborhood', default='mueller_neighborhood_7roi_20261031')
    parser.add_argument('--ppt', default='相关资料/2026高项偏振介绍_副本(1).pptx')
    parser.add_argument('--name')
    args = parser.parse_args()
    name = args.name or datetime.now().strftime('ppt_comparison_%Y%m%d_%H%M%S_%f')
    if Path(name).name != name or name in {'.', '..'}:
        parser.error('name must be a single directory name')
    run(args.runs, args.audit, args.neighborhood, name, args.ppt)


if __name__ == '__main__':
    main()
