"""Summarize completed runs and plot their recorded training diagnostics."""
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
SUITE = OUT.parent
RUNS = sorted(SUITE.glob('0[1-4]_*'))
LABELS = ['Country multi', 'Latent multi', 'Country K-way', 'Latent K-way']
DOMAINS = ['Japan', 'India', 'United_States', 'Norway', 'new_Czech',
           'new_China_Drone', 'new_China_MotorBike']
COLORS = ['#3573b9', '#da8133', '#458c61', '#ad549a']


def read_json(path):
    return json.loads(path.read_text())


def main():
    metrics, audits, results, summary = [], [], [], []
    for run in RUNS:
        rows = [{k: float(v) for k, v in row.items()} for row in
                csv.DictReader((run / 'metrics/metrics_epoch.csv').open())]
        audit = [json.loads(line) for line in (run / 'epoch_audit.jsonl').read_text().splitlines()]
        test = {d: read_json(run / 'testing' / d / 'metrics.json') for d in DOMAINS}
        # Detect incomplete runs before including their means in the comparison.
        assert [r['epoch'] for r in rows] == list(range(1, 61)), run.name
        assert len(audit) == 60 and all(a['images_seen'] == 12742 and
                                       a['validation_images'] == 1592 for a in audit), run.name
        assert sum(t['images'] for t in test.values()) == 9433, run.name
        best = min(rows, key=lambda r: r['val_loss'])
        checkpoint = read_json(run / 'best_checkpoint.json')
        assert checkpoint['epoch'] == best['epoch'], run.name
        assert abs(checkpoint['val_loss'] - best['val_loss']) < 1e-10, run.name
        metrics.append(rows)
        audits.append(audit)
        results.append(test)
        summary.append(dict(
            experiment=run.name, selected_epoch=int(best['epoch']),
            target_ap50=100 * statistics.mean(test[d]['coco']['map50'] for d in DOMAINS[3:]),
            target_ap50_95=100 * statistics.mean(test[d]['coco']['map50_95'] for d in DOMAINS[3:]),
            best_val_loss=best['val_loss'], last_val_loss=rows[-1]['val_loss'],
            val_loss_rise_pct=100 * (rows[-1]['val_loss'] / best['val_loss'] - 1),
            sampled_domain_det_ratio_median_epoch20_60=statistics.median(
                a['gradient_checks'][0]['domain_det_ratio'] for a in audit[19:]),
            selected_domain_balanced_accuracy=audit[int(best['epoch'])-1].get('domain_balanced_accuracy'),
            last_domain_balanced_accuracy=audit[-1].get('domain_balanced_accuracy'),
        ))
    with (OUT / 'diagnostics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for i, rows in enumerate(metrics):
        epochs = [r['epoch'] for r in rows]
        axes[0, 0].plot(epochs[1:], [r['loss_yolo'] for r in rows[1:]], color=COLORS[i], label=LABELS[i])
        axes[0, 1].plot(epochs, [r['val_loss'] for r in rows], color=COLORS[i])
        best = min(rows, key=lambda r: r['val_loss'])
        axes[0, 1].scatter(best['epoch'], best['val_loss'], color=COLORS[i], s=30)
        axes[1, 0].plot(epochs, [100*r['map_50'] for r in rows], color=COLORS[i])
        values = [a['gradient_checks'][0]['domain_det_ratio'] for a in audits[i]]
        axes[1, 1].plot(epochs, [v if v > 0 else np.nan for v in values], color=COLORS[i])
    titles = ['Training detection loss (epochs 2-60)', 'Source validation detection loss',
              'Source validation AP50 (training callback)', 'Domain / detection feature gradient norm']
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.2)
    axes[1, 0].set_ylabel('AP50 (%)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_ylabel('Ratio; first batch of each epoch')
    axes[0, 0].legend()
    fig.savefig(OUT / 'training_diagnostics.png', dpi=180)
    fig.savefig(OUT / 'training_diagnostics.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    x = np.arange(4)
    for i, test in enumerate(results):
        vals = [100*test[d]['coco']['map50'] for d in DOMAINS[3:]]
        axes[0].bar(x + (i-1.5)*.2, vals, width=.2, color=COLORS[i], label=LABELS[i])
    axes[0].set_xticks(x, ['Norway', 'Czech', 'China-Drone', 'China-MotorBike'])
    axes[0].set_ylabel('COCO AP50 (%)')
    axes[0].set_title('Unseen domains; selected checkpoints')
    axes[0].legend(fontsize=8)
    for i in (2, 3):
        axes[1].plot(range(1, 61), [100*a['domain_balanced_accuracy'] for a in audits[i]],
                     label=LABELS[i], color=COLORS[i])
    axes[1].axhline(100/3, color=COLORS[2], linestyle=':', label='K=3 chance (balanced)')
    axes[1].axhline(50, color=COLORS[3], linestyle=':', label='K=2 chance (balanced)')
    axes[1].set_ylim(25, 102)
    axes[1].set_title('Online training domain classification')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Balanced accuracy (%)')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(OUT / 'target_and_domain_accuracy.png', dpi=180)
    fig.savefig(OUT / 'target_and_domain_accuracy.pdf')
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
