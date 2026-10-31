"""Export native-size reference images and draft masks for review."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='analysis/runs/ppt_reference_4channels_20261031')
    parser.add_argument('--name', default='ppt_20261031')
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {'.', '..'}:
        parser.error('name must be a single directory name')
    source = ROOT / args.run
    config = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    out = ROOT / 'analysis' / 'mask_review' / args.name
    out.mkdir(parents=True, exist_ok=False)
    entries = []
    for roi in config['rois']:
        prepared = np.load(source / f'roi_{roi}_prepared.npz')
        mat = ROOT / config['data_root'] / str(roi) / 'm11.mat'
        intensity = loadmat(mat, variable_names=['CalibratedM11'])['CalibratedM11']
        if tuple(prepared['original_shape']) != intensity.shape:
            raise ValueError(f'ROI {roi}: m11.mat and saved source shape differ')
        low, high = np.quantile(intensity, [.01, .99])
        if high <= low:
            raise ValueError(f'ROI {roi}: intensity display range is constant')
        display = (np.clip((intensity - low) / (high - low), 0, 1) * 255).astype(np.uint8)
        reference = out / f'roi_{roi}_reference.png'
        draft = out / f'roi_{roi}_draft.png'
        Image.fromarray(display).save(reference)
        small_mask = Image.fromarray(prepared['mask'].astype(np.uint8) * 255)
        small_mask.resize((intensity.shape[1], intensity.shape[0]), Image.Resampling.NEAREST).save(draft)
        entries.append({'roi': roi, 'shape': list(intensity.shape), 'source': str(mat),
                        'reference': reference.name, 'draft': draft.name,
                        'display_quantiles': [.01, .99], 'display_range': [float(low), float(high)],
                        'draft_status': 'upsampled_auto_mask_not_reviewed'})
        print(f'ROI {roi}: reference and unreviewed draft exported', flush=True)
    (out / 'manifest.json').write_text(json.dumps({'source_run': str(source), 'entries': entries},
        ensure_ascii=False, indent=2), encoding='utf-8')
    (out / '使用说明.txt').write_text(
        '每个reference.png是原始尺寸的灰度显示图，亮度按1%和99%分位缩放，仅用于对照，不能替代原始数值。\n'
        '每个draft.png是将自动掩膜最近邻放大后的二值草稿，尚未核对。\n'
        '在相同坐标和尺寸下检查组织、背景、腔隙和切片边缘，修订后另存为roi_1.png等，放入单独目录。\n'
        '修订掩膜为单通道PNG，0为背景，255为组织；处理7个ROI时应提供全部7张。\n'
        '命令示例见analysis\\操作指南.txt。不要把带标题与色条的图表当作掩膜。\n', encoding='utf-8')
    print(f'OUTPUT={out}', flush=True)


if __name__ == '__main__':
    main()
