"""Polarization superpixel baselines. Run with --help; originals are read only."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from skimage.filters import sobel, threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects
from skimage.segmentation import mark_boundaries, relabel_sequential, slic
from skimage.transform import resize
from threadpoolctl import threadpool_limits
from graph_models import dgi_embedding

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ('PL', 'DL', 'qL', 'rL')
PAIRS = (('21', '31'), ('12', '13'), ('42', '43'), ('24', '34'))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def parameters(matrix, normalization, include_total_diattenuation=False):
    """Use one-based Mueller indices; no clipping or absolute-value substitution."""
    m11 = matrix['FinalM11']
    needed = ['FinalM' + x for pair in PAIRS for x in pair]
    for name in ['FinalM11', *needed]:
        a = matrix[name]
        if a.ndim != 2 or a.shape != m11.shape or not np.isfinite(a).all():
            raise ValueError(f'{name}: expected matching finite 2D arrays, got {a.shape}')
    if normalization == 'as_stored':
        error = float(np.quantile(np.abs(m11 - 1), .999))
        if error > .01:
            raise ValueError(f'M11 differs from unity (99.9% error={error:.6g}); confirm normalization before continuing')
        divisor = 1.
    elif normalization == 'divide_by_m11':
        if np.any(m11 <= 0):
            raise ValueError('Cannot divide by nonpositive M11; inspect invalid pixels')
        divisor = m11
    else:
        raise ValueError('normalization must explicitly be as_stored or divide_by_m11')
    maps = [np.hypot(matrix['FinalM' + a] / divisor,
                     matrix['FinalM' + b] / divisor) for a, b in PAIRS]
    if include_total_diattenuation:
        m14 = matrix['FinalM14']
        if m14.shape != m11.shape or not np.isfinite(m14).all():
            raise ValueError('FinalM14 must be finite and match FinalM11')
        maps.append(np.hypot(maps[1], m14 / divisor))
    return np.stack(maps, axis=-1).astype(np.float32)


def auto_mask(intensity):
    """Texture-based draft; never treated as a reviewed tissue/lumen mask."""
    smooth = ndimage.gaussian_filter(intensity, 1.)
    local_var = ndimage.gaussian_filter(smooth * smooth, 3.) - ndimage.gaussian_filter(smooth, 3.) ** 2
    score = np.sqrt(np.maximum(local_var, 0)) + sobel(smooth)
    mask = score > threshold_otsu(score)
    mask = binary_closing(mask, footprint=disk(4))
    mask = remove_small_holes(mask, area_threshold=128)
    return remove_small_objects(mask, min_size=128)


def extract_features(maps, labels):
    n = int(labels.max())
    count = np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(np.float64)
    if np.any(count == 0):
        raise ValueError('Superpixel labels must be contiguous and nonempty')
    features = []
    for c in range(maps.shape[-1]):
        values = maps[..., c].astype(np.float64)
        mean = np.bincount(labels.ravel(), weights=values.ravel(), minlength=n + 1)[1:] / count
        second = np.bincount(labels.ravel(), weights=(values * values).ravel(), minlength=n + 1)[1:] / count
        features.extend([mean, np.sqrt(np.maximum(second - mean * mean, 0))])
    return np.stack(features, axis=1).astype(np.float32), count


def spatial_edges(labels):
    pairs = []
    for a, b in [(labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:])]:
        keep = (a > 0) & (b > 0) & (a != b)
        pairs.append(np.sort(np.stack([a[keep] - 1, b[keep] - 1], axis=1), axis=1))
    return np.unique(np.concatenate(pairs), axis=0, return_counts=True)


def normalized_graph(n, edges, weights):
    rows = np.concatenate([edges[:, 0], edges[:, 1], np.arange(n)])
    cols = np.concatenate([edges[:, 1], edges[:, 0], np.arange(n)])
    vals = np.concatenate([weights, weights, np.ones(n)])
    a = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    inv = np.asarray(a.sum(axis=1)).ravel() ** -.5
    return (sparse.diags(inv) @ a @ sparse.diags(inv)).astype(np.float32)


def sample_negatives(graphs, rng):
    """Sample only within the same ROI; exclude real edges and self-pairs."""
    result, offset = [], 0
    for n, edges in graphs:
        forbidden = {tuple(x) for x in edges.tolist()}
        available = n * (n - 1) // 2 - len(forbidden)
        target = min(len(edges), available)
        if target < 1:
            raise ValueError('GAE requires spatial edges and available nonedges in every ROI graph')
        selected = set()
        while len(selected) < target:
            candidates = np.sort(rng.integers(0, n, size=(max(64, 2 * (target - len(selected))), 2)), axis=1)
            for a, b in candidates:
                pair = (int(a), int(b))
                if a != b and pair not in forbidden:
                    selected.add(pair)
                    if len(selected) == target:
                        break
        result.append(np.asarray(sorted(selected)) + offset)
        offset += n
    return np.concatenate(result)


def gae_embedding(x, adjacency, graphs, seed, epochs, device, folder):
    import torch
    from torch import nn
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    coo = adjacency.tocoo()
    a = torch.sparse_coo_tensor(np.stack([coo.row, coo.col]), coo.data,
                                size=coo.shape, device=device).coalesce()
    tx = torch.as_tensor(x, dtype=torch.float32, device=device)
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.first, self.second = nn.Linear(x.shape[1], 32), nn.Linear(32, 16)
        def forward(self):
            h = torch.relu(self.first(torch.sparse.mm(a, tx)))
            return self.second(torch.sparse.mm(a, h))
    model = Encoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    positives, offset = [], 0
    for n, edges in graphs:
        positives.append(edges + offset)
        offset += n
    positive = torch.as_tensor(np.concatenate(positives).T, device=device, dtype=torch.long)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        negative = torch.as_tensor(sample_negatives(graphs, rng).T, device=device, dtype=torch.long)
        optimizer.zero_grad()
        z = model()
        pos = (z[positive[0]] * z[positive[1]]).sum(1)
        neg = (z[negative[0]] * z[negative[1]]).sum(1)
        loss = .5 * (nn.functional.softplus(-pos).mean() + nn.functional.softplus(neg).mean())
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite GAE loss at epoch {epoch}')
        loss.backward()
        optimizer.step()
        history.append({'epoch': epoch + 1, 'loss': float(loss.detach().cpu())})
    model.eval()
    with torch.no_grad():
        embedding = model().cpu().numpy()
    write_csv(folder / f'gae_loss_seed{seed}.csv', history)
    torch.save(model.state_dict(), folder / f'gae_weights_seed{seed}.pt')
    return embedding, device


def spatial_metrics(cluster_map, k):
    valid = cluster_map >= 0
    counts = np.bincount(cluster_map[valid], minlength=k)
    p = counts / counts.sum()
    entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    contacts = np.zeros((k, k), dtype=np.int64)
    mixed = total = 0
    for a, b in [(cluster_map[:-1], cluster_map[1:]), (cluster_map[:, :-1], cluster_map[:, 1:])]:
        keep = (a >= 0) & (b >= 0)
        aa, bb = a[keep], b[keep]
        total += int(keep.sum())
        mixed += int((aa != bb).sum())
        np.add.at(contacts, (aa, bb), 1)
        np.add.at(contacts, (bb, aa), 1)
    return {'shannon_entropy': entropy, 'normalized_entropy': entropy / np.log(k),
            'mixed_neighbor_fraction': mixed / total if total else 0.,
            'connected_components': int(sum(ndimage.label(cluster_map == c)[1] for c in range(k)))}, p, contacts


def silhouette(x, labels):
    if not 1 < np.unique(labels).size < len(labels):
        return None
    return float(silhouette_score(x, labels, sample_size=min(2000, len(x)), random_state=0))


def load_roi(roi, cfg, folder):
    source = ROOT / cfg['data_root'] / str(roi) / 'FinalMM.mat'
    names = ['FinalM11', 'CalibratedM11', *['FinalM' + c for pair in PAIRS for c in pair]]
    include_d = cfg.get('include_total_diattenuation', False)
    channels = (*CHANNELS, 'D') if include_d else CHANNELS
    if include_d:
        names.append('FinalM14')
    d = loadmat(source, variable_names=names)
    missing = [name for name in names if name not in d]
    if missing:
        raise ValueError(f'{source}: missing required variables: {missing}')
    original = d['FinalM11'].shape
    m11_finite = d['FinalM11'][np.isfinite(d['FinalM11'])]
    source_stat = source.stat()
    q = {'roi': roi, 'source': str(source), 'source_bytes': source_stat.st_size,
         'source_mtime_ns': source_stat.st_mtime_ns,
         'original_shape': list(original), 'normalization': cfg['normalization'],
         'required_array_nonfinite_counts': {n: int((~np.isfinite(d[n])).sum()) for n in names},
         'm11_abs_error_p999': float(np.quantile(np.abs(m11_finite - 1), .999)) if m11_finite.size else None,
         'm11_min': float(m11_finite.min()) if m11_finite.size else None,
         'm11_max': float(m11_finite.max()) if m11_finite.size else None}
    save_json(folder / f'roi_{roi}_qc.json', q)
    maps = parameters(d, cfg['normalization'], include_total_diattenuation=include_d)
    intensity = d['CalibratedM11'].astype(np.float32)
    if not np.isfinite(intensity).all() or intensity.shape != original:
        raise ValueError('CalibratedM11 must match the matrix and contain finite values')
    ratio = min(1., cfg['max_side'] / max(original))
    shape = tuple(max(1, round(v * ratio)) for v in original)
    if shape != original:
        maps = resize(maps, (*shape, len(channels)), preserve_range=True, anti_aliasing=True).astype(np.float32)
        intensity = resize(intensity, shape, preserve_range=True, anti_aliasing=True).astype(np.float32)
    mask_dir = cfg.get('mask_dir')
    if mask_dir:
        mask_path = ROOT / mask_dir / f'roi_{roi}.png'
        mask = np.asarray(Image.open(mask_path))
        if mask.ndim != 2 or mask.shape != original or not set(np.unique(mask)).issubset({0, 1, 255}):
            raise ValueError(f'{mask_path}: expected original-resolution binary PNG')
        mask = resize(mask > 0, shape, order=0, preserve_range=True, anti_aliasing=False).astype(bool)
        status = 'external_mask_provided'
    else:
        mask, status = auto_mask(intensity), 'auto_draft_unreviewed'
    if mask.sum() < cfg['n_segments'] * 4:
        raise ValueError(f'ROI {roi}: tissue draft too small for requested superpixels; inspect mask')
    q.update(analysis_shape=list(shape), mask_status=status, mask_fraction=float(mask.mean()),
             parameter_quantiles={c: np.quantile(maps[..., i], [.01, .5, .99]).tolist() for i, c in enumerate(channels)})
    save_json(folder / f'roi_{roi}_qc.json', q)
    return {'roi': roi, 'maps': maps, 'channels': channels, 'intensity': intensity, 'mask': mask,
            'original_shape': original, 'qc': q}


def prepare_rois(cfg, folder):
    rois = []
    for roi in cfg['rois']:
        print(f'Loading ROI {roi}', flush=True)
        rois.append(load_roi(roi, cfg, folder))
    pooled = np.concatenate([r['maps'][r['mask']][::16] for r in rois])
    low, high = np.quantile(pooled, [.01, .99], axis=0)
    if np.any(high <= low):
        raise ValueError('Constant polarization channel; inspect parameters before segmentation')
    save_json(folder / 'slic_scaling.json', {'low': low.tolist(), 'high': high.tolist(), 'quantiles': [.01, .99]})
    for r in rois:
        image = np.clip((r['maps'] - low) / (high - low), 0, 1).astype(np.float32)
        # Keep SLIC input fixed when adding total diattenuation to node features.
        labels = slic(image[..., :len(CHANNELS)], n_segments=cfg['n_segments'], compactness=cfg['compactness'],
                      mask=r['mask'], convert2lab=False, channel_axis=-1,
                      start_label=1, enforce_connectivity=True)
        labels = relabel_sequential(labels)[0].astype(np.int32)
        r['labels'] = labels
        r['features'], r['areas'] = extract_features(r['maps'], labels)
        r['edges'], r['shared_boundaries'] = spatial_edges(labels)
        np.savez_compressed(folder / f"roi_{r['roi']}_prepared.npz", parameters=r['maps'],
                            intensity=r['intensity'], mask=r['mask'], superpixels=labels,
                            features=r['features'], areas=r['areas'], edges=r['edges'],
                            shared_boundaries=r['shared_boundaries'], original_shape=r['original_shape'],
                            channels=np.asarray(r['channels']))
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes[0, 0].imshow(r['intensity'], cmap='gray'); axes[0, 0].set_title('CalibratedM11')
        axes[0, 1].imshow(r['mask'], cmap='gray'); axes[0, 1].set_title(r['qc']['mask_status'])
        axes[0, 2].imshow(mark_boundaries(np.clip(r['intensity'], 0, 1), labels)); axes[0, 2].set_title('SLIC boundaries')
        for i, ax in enumerate(axes[1]):
            im = ax.imshow(r['maps'][..., i], vmin=low[i], vmax=high[i], cmap='viridis')
            ax.set_title(CHANNELS[i]); fig.colorbar(im, ax=ax, fraction=.04)
        for ax in axes.ravel(): ax.axis('off')
        fig.suptitle(f"ROI {r['roi']} - draft mask; analysis grid {labels.shape}")
        fig.tight_layout(); fig.savefig(folder / f"roi_{r['roi']}_overview.png", dpi=130); plt.close(fig)
        fig, axes = plt.subplots(1, len(r['channels']), figsize=(3 * len(r['channels']), 3))
        for i, ax in enumerate(axes):
            im = ax.imshow(r['maps'][..., i], vmin=low[i], vmax=high[i], cmap='viridis')
            ax.set_title(r['channels'][i]); ax.axis('off'); fig.colorbar(im, ax=ax, fraction=.04)
        fig.tight_layout(); fig.savefig(folder / f"roi_{r['roi']}_parameters.png", dpi=130); plt.close(fig)
        print(f"ROI {r['roi']}: {len(r['features'])} nodes, {len(r['edges'])} edges; mask={r['qc']['mask_fraction']:.3f}", flush=True)
    return rois


def run(cfg):
    start = time.perf_counter()
    run_id = cfg.get('run_name') or datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise ValueError('run_name must be a single directory name')
    folder = ROOT / 'analysis' / 'runs' / run_id
    folder.mkdir(parents=True, exist_ok=False)
    save_json(folder / 'config.json', cfg)
    save_json(folder / 'environment.json', {name: importlib.metadata.version(name) for name in
              ['numpy', 'scipy', 'scikit-image', 'scikit-learn', 'torch', 'matplotlib']})
    save_json(folder / 'code_provenance.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [Path(__file__), Path(__file__).with_name('graph_models.py')]})
    rois = prepare_rois(cfg, folder)
    channels = rois[0]['channels']
    scaler = StandardScaler()
    x = scaler.fit_transform(np.concatenate([r['features'] for r in rois])).astype(np.float32)
    save_json(folder / 'feature_scaler.json', {'features': [f'{c}_{s}' for c in channels for s in ['mean', 'std']],
              'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()})
    graph_parts, graphs = [], []
    offset = 0
    node_rows = []
    for r in rois:
        n, edges = len(r['features']), r['edges']
        weights = np.ones(len(edges))
        if cfg['weighted']:
            distance = np.linalg.norm(x[offset + edges[:, 0]] - x[offset + edges[:, 1]], axis=1)
            sigma = np.median(distance)
            if sigma <= 0: raise ValueError('Degenerate feature distances for weighted graph')
            weights = np.exp(-.5 * (distance / sigma) ** 2)
        graph_parts.append(normalized_graph(n, edges, weights))
        graphs.append((n, edges))
        for i, area in enumerate(r['areas']):
            row = {'roi': r['roi'], 'node': i, 'area_analysis_pixels': int(area)}
            row.update({f'{c}_{s}': float(r['features'][i, j]) for j, (c, s) in enumerate((c, s) for c in channels for s in ['mean', 'std'])})
            node_rows.append(row)
        offset += n
    write_csv(folder / 'nodes.csv', node_rows)
    adjacency = sparse.block_diag(graph_parts, format='csr')
    sparse.save_npz(folder / 'normalized_adjacency.npz', adjacency)
    propagated = x.copy()
    for _ in range(cfg['hops']): propagated = adjacency @ propagated
    metrics, proportions, predictions, devices_used = [], [], {}, {}
    for seed in cfg['seeds']:
        for method in cfg['methods']:
            if method == 'baseline': z = x
            elif method == 'propagation': z = propagated
            elif method == 'gae':
                z, devices_used['gae'] = gae_embedding(x, adjacency, graphs, seed, cfg['epochs'], cfg['device'], folder)
            elif method == 'dgi':
                z, devices_used['dgi'] = dgi_embedding(x, adjacency, graphs, seed, cfg['epochs'], cfg['device'], folder)
            else: raise ValueError(f'Unknown method: {method}')
            cluster_space = StandardScaler().fit_transform(z)
            with threadpool_limits(limits=4):
                labels = KMeans(n_clusters=cfg['k'], n_init=20, random_state=seed).fit_predict(cluster_space)
            np.savez_compressed(folder / f'{method}_seed{seed}_embedding.npz', embedding=z, labels=labels)
            offset = 0
            for r in rois:
                n = len(r['features']); node_labels = labels[offset:offset+n]
                prediction = np.full(r['labels'].shape, -1, dtype=np.int16)
                prediction[r['mask']] = node_labels[r['labels'][r['mask']] - 1]
                spatial_result, p, contacts = spatial_metrics(prediction, cfg['k'])
                metric = {'roi': r['roi'], 'method': method, 'seed': seed,
                          'mask_status': r['qc']['mask_status'], 'n_nodes': n,
                          'silhouette_base': silhouette(x[offset:offset+n], node_labels),
                          'silhouette_core4': silhouette(x[offset:offset+n, :8], node_labels),
                          'silhouette_embedding': silhouette(cluster_space[offset:offset+n], node_labels), **spatial_result}
                metrics.append(metric)
                for c in range(cfg['k']):
                    proportions.append({'roi': r['roi'], 'method': method, 'seed': seed,
                                        'cluster': c, 'area_fraction': float(p[c])})
                stem = f"roi_{r['roi']}_{method}_seed{seed}"
                np.save(folder / f'{stem}_labels.npy', prediction)
                np.savetxt(folder / f'{stem}_contacts.csv', contacts, delimiter=',', fmt='%d')
                if seed in cfg.get('plot_seeds', cfg['seeds']):
                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.imshow(r['intensity'], cmap='gray')
                    im = ax.imshow(np.ma.masked_less(prediction, 0), cmap=plt.get_cmap('tab10', cfg['k']),
                                   vmin=-.5, vmax=cfg['k'] - .5, alpha=.8)
                    fig.colorbar(im, ax=ax, ticks=list(range(cfg['k'])), fraction=.04)
                    ax.set_title(f"ROI {r['roi']} | {method} | seed {seed}\nExploratory clusters; method-specific IDs; {r['qc']['mask_status']}", fontsize=9)
                    ax.axis('off'); fig.tight_layout(); fig.savefig(folder / f'{stem}.png', dpi=130); plt.close(fig)
                predictions[(r['roi'], method, seed)] = node_labels.copy()
                offset += n
            print(f'{method} seed={seed} completed', flush=True)
    stability = []
    for r in rois:
        for method in cfg['methods']:
            for i, seed_a in enumerate(cfg['seeds']):
                for seed_b in cfg['seeds'][i+1:]:
                    stability.append({'roi': r['roi'], 'method': method, 'seed_a': seed_a, 'seed_b': seed_b,
                        'ARI': float(adjusted_rand_score(predictions[(r['roi'], method, seed_a)], predictions[(r['roi'], method, seed_b)]))})
    write_csv(folder / 'metrics.csv', metrics)
    write_csv(folder / 'area_fractions.csv', proportions)
    write_csv(folder / 'stability_ari.csv', stability)
    mask_statuses = sorted({r['qc']['mask_status'] for r in rois})
    normalization_evidence = ('Stored M11 passed the near-unity numerical check; calibration provenance remains unverified'
                              if cfg['normalization'] == 'as_stored'
                              else 'Parameters were divided by the stored M11; calibration provenance remains unverified')
    limitations = ['Mask provenance and tissue/lumen interpretation must be reviewed',
                   'No pathology ground truth or accuracy claims',
                   'BR-029 ROI source relationship not independently confirmed',
                   'Downsampling changes fine structure; full-resolution validation remains pending']
    save_json(folder / 'summary.json', {'status': 'completed_exploratory_run', 'rois': cfg['rois'],
              'methods': cfg['methods'], 'seeds': cfg['seeds'], 'seconds': time.perf_counter()-start,
              'graph_devices': devices_used, 'node_count': len(x), 'mask_statuses': mask_statuses,
              'feature_channels': list(channels), 'slic_channels': list(CHANNELS),
              'normalization_evidence': normalization_evidence,
              'spatial_units': 'downsampled_analysis_grid_pixels',
              'limitations': limitations})
    cards = ''.join(f'<figure><img src="{html.escape(p.name)}"><figcaption>{html.escape(p.stem)}</figcaption></figure>' for p in sorted(folder.glob('*.png')))
    (folder / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Polarization pilot</title>'
        '<style>body{font:16px sans-serif;max-width:1200px;margin:32px auto;color:#20324b}img{max-width:100%}figure{margin:24px 0}</style>'
        '<h1>类器官偏振分析试跑</h1><p>结果为探索性输出。请核对掩膜及组织、腔隙范围；类别编号不代表组织名称。'
        '请先检查掩膜和超像素边界，再解释指标。详见同目录 config.json、summary.json 和 CSV。</p>' + cards, encoding='utf-8')
    print(f'OUTPUT={folder}', flush=True)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--rois', type=int, nargs='+')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--mask-dir')
    parser.add_argument('--run-name')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    for key in ['rois', 'seeds', 'epochs', 'device', 'mask_dir', 'run_name']:
        value = getattr(args, key)
        if value is not None: cfg[key] = value
    if not cfg['rois'] or len(set(cfg['rois'])) != len(cfg['rois']):
        parser.error('rois must be distinct and nonempty')
    if not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']):
        parser.error('seeds must be distinct and nonempty')
    if cfg['k'] < 2 or cfg['epochs'] < 1 or cfg['max_side'] < 32 or cfg['n_segments'] < cfg['k']:
        parser.error('Invalid k, epochs, max_side or n_segments')
    run(cfg)


if __name__ == '__main__':
    main()
