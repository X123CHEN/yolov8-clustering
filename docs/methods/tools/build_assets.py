"""Read archived experiment records and rebuild documentation tables and figures."""
import ast
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

DOCS = Path(__file__).resolve().parents[1]
ROOT = DOCS.parents[1]
S0 = ROOT / 'stage0-latent_domain_analysis'
S1 = ROOT / 'stage1-latent-domain'
S2 = ROOT / 'stage2-k-way'
SUITE = S2 / 'kway_comparison_100ep_seed11_20261015'
TABLES = DOCS / 'tables'
FIGURES = DOCS / 'figures'
RUNS = ['01_country_multi_unique', '02_latent_multi_unique',
        '03_country_kway_unique', '04_latent_kway_unique']
LABELS = ['Country multi', 'Latent multi', 'Country K-way', 'Latent K-way']
FIRST = ['baseline', 'country_repeat', 'latent_repeat', 'country_unique', 'latent_unique']
DOMAINS = ['Japan', 'India', 'United_States', 'Norway', 'new_Czech',
           'new_China_Drone', 'new_China_MotorBike']
COLORS = ['#3268a8', '#d07a2f', '#278369', '#9960a3']
METRICS = ['voc_map50', 'coco_map50', 'coco_map50_95']


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def relative(path):
    return str(path.relative_to(ROOT))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def archive_metrics():
    first_train = S1 / 'comparison_20261005_100ep'
    older = S2 / 'kway_comparison_60ep_seed42_20261014'
    protocols = [
        ('stage1_original_score', S1 / 'test_comparison_20261006', first_train, FIRST, 11, 100, 'uncertainty'),
        ('stage1_standard_score', S1 / 'test_comparison_standard_conf_20261006_201514', first_train, FIRST, 11, 100, 'max_class_sigmoid'),
        ('stage2_60_original_score', older, older, RUNS, 42, 60, 'uncertainty'),
        ('stage2_60_standard_score', older / 'testing_standard_conf_20261015_115715', older, RUNS, 42, 60, 'max_class_sigmoid'),
        ('stage2_100_standard_score', SUITE, SUITE, RUNS, 11, 100, 'max_class_sigmoid'),
    ]
    macro, per_domain = [], []
    for protocol, test_root, train_root, runs, seed, epochs, confidence in protocols:
        for run in runs:
            folder = test_root / run
            if not (folder / 'summary.csv').exists():
                folder = folder / 'testing'
            completion = read_json(folder / 'completion.json')
            assert completion['images'] == 9433 and completion['domains_completed'] == 7
            assert not completion['smoke_test']
            rows = read_csv(folder / 'summary.csv')
            assert [row['domain'] for row in rows] == DOMAINS
            config = read_json(train_root / run / 'run_config.json')
            assert config['seed'] == seed and config['epochs'] == epochs
            info = dict(protocol=protocol, experiment=run, seed=seed, epochs=epochs,
                        confidence=confidence, source=relative(folder / 'summary.csv'))
            for row in rows:
                per_domain.append(dict(**info, domain=row['domain'], images=int(row['images']),
                                       **{key: 100 * float(row[key]) for key in METRICS}))
            for split, domains in [('source', DOMAINS[:3]), ('target', DOMAINS[3:]), ('all', DOMAINS)]:
                selected = [row for row in rows if row['domain'] in domains]
                macro.append(dict(**info, group=split, domains=len(domains),
                                  **{key: 100 * np.mean([float(row[key]) for row in selected]) for key in METRICS}))
    write_csv(TABLES / 'experiment_macro_metrics.csv', macro)
    write_csv(TABLES / 'experiment_domain_metrics.csv', per_domain)
    cost = read_csv(SUITE / 'comparison.csv')
    # Detect accidental mixing of an old-score summary with the standard-score main table.
    for row in cost:
        recorded = next(item for item in macro if item['protocol'] == 'stage2_100_standard_score'
                        and item['experiment'] == row['experiment'] and item['group'] == 'all')
        for key in METRICS:
            assert abs(recorded[key] - 100 * float(row['all_' + key])) < 1e-9
    write_csv(TABLES / 'stage2_100_cost_and_accuracy.csv', cost)
    pairs = []
    for left, right in [(2, 0), (3, 1), (1, 0), (3, 2)]:
        a, b = cost[left], cost[right]
        pairs.append(dict(comparison=RUNS[left] + ' minus ' + RUNS[right],
                          voc50_difference_pp=100 * (float(a['all_voc_map50']) - float(b['all_voc_map50'])),
                          coco50_difference_pp=100 * (float(a['all_coco_map50']) - float(b['all_coco_map50'])),
                          train_time_reduction_pct=100 * (1 - float(a['train_wall_time_s']) / float(b['train_wall_time_s'])),
                          total_training_parameter_reduction_pct=100 * (1 - int(a['total_training_parameters']) / int(b['total_training_parameters']))))
    write_csv(TABLES / 'stage2_100_paired_differences.csv', pairs)
    write_csv(TABLES / 'k_selection_metrics.csv', read_csv(S0 / 'results/k_selection_metrics.csv'))
    return macro, cost


def index_files():
    entries = []
    for p in sorted(S0.glob('*.py')):
        entries.append((p, 'stage0', '离线表示、K选择、分配、导出或校验'))
    for pattern in ['results/final_k_decision.json', 'results/final_cluster_labels.csv',
                    'results/final_val_cluster_labels.csv', 'results/cluster_centers.npy',
                    'results/scaler_mean.npy', 'results/scaler_scale.npy', 'results/K选择报告.md',
                    'domain_splits/split_summary.json', 'domain_splits/combined_train.txt',
                    'domain_splits/combined_val.txt', 'domain_splits/Cluster_0_train.txt',
                    'domain_splits/Cluster_1_train.txt', 'domain_splits/Cluster_0_val.txt',
                    'domain_splits/Cluster_1_val.txt']:
        entries.append((S0 / pattern, 'stage0', '固定域artifact或说明'))
    main_roles = {
        'utils/comparison_training.py': '五组共享数据/Triplet/多专家训练；仍含旧数据路径',
        'utils/comparison_evaluation.py': '严格加载及VOC/COCO计分；解码器须显式确认',
        'nets/yolo.py': 'Student骨干/检测前向和特征返回',
        'nets/backbone.py': '骨干及冻结专家构建',
        'nets/yolo_training.py': '检测Loss、EMA和学习率日程',
        'nets/maddg.py': '逐域Patch判别器卷积主干',
        'utils/dataloader.py': '单图增强、数据集、标注格式',
        'utils/callbacks.py': '损失与源验证AP记录',
        'utils/latent_adversarial.py': '主线复用KWayDiscriminator；其余为扩展分支',
        'train_style_adv.py': '当前旧K=3入口；与S1分发快照不同',
        'README_style.md': '早期K=3软路由说明',
        'dg_config.py': '早期K=3软路由配置',
        'latent_dg_config.py': 'K=2扩展hard/soft/profile配置',
        'train_latent_ablation.py': '扩展消融训练入口',
        'utils/latent_membership.py': '离线hard/soft membership缓存',
        'utils/latent_ablation_training.py': 'membership消融训练器',
        'utils/latent_v2.py': 'V2 GRL/pooled/balanced CE分支',
        'utils/latent_structure.py': '条件式对抗和归一化变体',
        'utils/online_latent_centers.py': '在线中心分支；主四组未用',
        'utils/latent_validation.py': '扩展验证口径',
        'tools/run_latent_suite.py': '扩展任务套件',
        'tools/run_latent_screen_worker.py': '扩展screen任务worker',
        'tools/run_kway_stage_worker.py': '历史K-way v2阶段worker',
        'tools/calibrate_latent_structure.py': '扩展结构系数标定',
        'tools/run_deterministic_parity.py': '确定性对照包装器',
        'tools/audit_latent_training.py': '历史训练审计',
        'tools/audit_latent_evaluation.py': '历史测试口径审计',
        'tools/summarize_evaluation_audit.py': '历史复测汇总',
        'utils/style_router.py': '早期soft路由',
        'utils/style_assignment.py': '早期soft分配',
        'utils/style_adv_trainer.py': '早期加权对抗',
        'get_map_new.py': '通用测试入口；确认运行解码器',
    }
    for path, role in main_roles.items():
        entries.append((ROOT / path, 'shared_or_historical', role))
    for p in sorted((ROOT / 'new_experiments/kway_domain').glob('*.py')):
        entries.append((p, 'stage2_live', '根目录版本；默认60轮/seed42，区别于100轮快照'))
    frozen = SUITE / 'source_snapshot'
    for p in sorted((frozen / 'new_experiments/kway_domain').glob('*.py')):
        entries.append((p, 'stage2_100_frozen', '100轮实际执行快照'))
    entries += [(frozen / 'run_experiment.py', 'stage2_100_frozen', '快照引导；搬迁后需数据路径适配'),
                (frozen / 'standard_utils_bbox.py', 'stage2_100_frozen', '最终标准类别置信度解码器'),
                (S1 / 'comparison_20261005_100ep/source_snapshot/train_style_adv.py', 'stage1_frozen', '五组提前分发入口'),
                (S1 / 'comparison_20261005_100ep/source_snapshot/utils/comparison_training.py', 'stage1_frozen', '第一阶段实际核心训练'),
                (S1 / 'evaluation_audit_20261007/结果分析.md', 'historical_audit', '其他hard/soft扩展分支结果')]
    files, symbols = [], []
    for path, stage, role in entries:
        assert path.exists(), path
        files.append(dict(path=relative(path), stage=stage, role=role))
        if path.suffix == '.py':
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    symbols.append(dict(path=relative(path), symbol=node.name,
                                        kind=type(node).__name__, line=node.lineno, stage=stage))
    write_csv(TABLES / 'file_inventory.csv', files)
    write_csv(TABLES / 'code_index.csv', symbols)
    return len(files), len(symbols)


def save_figure(fig, name):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(FIGURES / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def box(ax, x, y, text, width=.16, height=.18, color='#eaf0f7', fontsize=10):
    ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height,
                               boxstyle='round,pad=0.012', facecolor=color,
                               edgecolor='#58718a', linewidth=1.1))
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize)


def arrow(ax, a, b, label=None, dashed=False, color='#526778'):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=13,
                                color=color, linewidth=1.3, linestyle='--' if dashed else '-'))
    if label:
        ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+.03, label, ha='center', fontsize=8, color=color)


def flow_figures():
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    xs = [.10, .30, .5, .70, .90]
    texts = ['Source train\n12,742 images', 'Frozen backbone\nfeat1: 128 x 80 x 80',
             'Channel mean/std\n256D + train Z-score', 'K = 2..8\nstructure + stability', 'Fixed K = 2\ncentres + hard IDs']
    for x, text in zip(xs, texts):
        box(ax, x, .79, text, width=.145, height=.20, fontsize=9)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a+.085, .79), (b-.085, .79))
    box(ax, .14, .27, 'Source images + boxes\nfixed domain IDs', width=.22, color='#eaf3ed', fontsize=9)
    box(ax, .43, .27, 'Student + domain module\ndetection + object Triplet', width=.23, color='#eaf3ed', fontsize=9)
    box(ax, .70, .27, 'EMA detector\nmin source-val loss', width=.17, color='#fff1df', fontsize=9)
    box(ax, .90, .27, 'Seven-domain test\nstandard class score', width=.14, color='#f0eaf4', fontsize=8)
    arrow(ax, (.90, .68), (.14, .39), 'fixed assignment')
    arrow(ax, (.265, .27), (.30, .27))
    arrow(ax, (.56, .27), (.60, .27))
    arrow(ax, (.80, .27), (.815, .27))
    ax.text(.5, .02, 'Source validation: assignment / checkpoint selection only. Target tests: evaluation only.',
            ha='center', fontsize=10)
    ax.set_title('Offline latent domains and detector training', fontsize=15, pad=15)
    save_figure(fig, '01_method_overview')

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    a = axes[0]
    a.set_title('(a) Frozen experts + per-domain critics', fontsize=13)
    box(a, .15, .56, 'Source x\nhard ID k', width=.19)
    box(a, .44, .76, 'Expert E_k\nfrozen, eval', width=.23, color='#fff1df')
    box(a, .44, .36, 'Student F\nfeat3', width=.23, color='#eaf3ed')
    box(a, .82, .56, 'Critic D_k\nLSGAN', width=.22)
    arrow(a, (.25, .61), (.315, .73))
    arrow(a, (.25, .51), (.315, .39))
    arrow(a, (.57, .76), (.70, .62), 'real = 1')
    arrow(a, (.57, .36), (.70, .50), 'fake = 0')
    arrow(a, (.70, .43), (.57, .27), 'student aims at 1', dashed=True, color='#b54f4f')
    a.text(.5, .08, 'Student step: D frozen\nThen per-domain D steps: expert no-grad; student detached',
           ha='center', fontsize=10)
    b = axes[1]
    b.set_title('(b) Single K-way domain classifier', fontsize=13)
    for x, text, color, width in [(.13, 'Source x', '#eaf0f7', .16),
                                 (.38, 'Student F\nfeat3', '#eaf3ed', .20),
                                 (.63, 'GRL\nforward = identity', '#fff1df', .17),
                                 (.88, 'K logits\nHard CE', '#eaf0f7', .17)]:
        box(b, x, .56, text, width=width, fontsize=9)
    arrow(b, (.22, .56), (.27, .56))
    arrow(b, (.49, .56), (.535, .56))
    arrow(b, (.725, .56), (.785, .56))
    box(b, .88, .86, 'Fixed domain ID', width=.19, height=.11, fontsize=9)
    arrow(b, (.88, .79), (.88, .67))
    arrow(b, (.79, .40), (.49, .40), r'feature gradient: $-\alpha r$', dashed=True, color='#b54f4f')
    b.text(.5, .08, r'One joint backward: Student SGD + classifier Adam' + '\n' +
           r'No experts; classifier gradient: $+\alpha\nabla L_{dom}$', ha='center', fontsize=10)
    fig.text(.5, -.015, 'Shared task path: detection + 0.3 object Triplet (feat3 / feat2). Inference: detector only.',
             ha='center', fontsize=11)
    fig.tight_layout(w_pad=2)
    save_figure(fig, '02_adversarial_comparison')


def data_figures(macro, cost):
    k = read_csv(S0 / 'results/k_selection_metrics.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    axes[0].errorbar([int(r['K']) for r in k], [float(r['silhouette_mean']) for r in k],
                     yerr=[float(r['silhouette_std']) for r in k], marker='o', capsize=4, color=COLORS[0])
    axes[0].set(xlabel='K', ylabel='Silhouette', title='Source-train K comparison', xticks=range(2, 9))
    composition = np.array([[306, 5055, 552], [2272, 1265, 3292]])
    bottoms = np.zeros(2)
    for i, country in enumerate(['India', 'Japan', 'United States']):
        values = composition[:, i]
        axes[1].bar(['Cluster 0', 'Cluster 1'], values, bottom=bottoms,
                    color=[COLORS[1], COLORS[0], COLORS[2]][i], label=country, width=.55)
        for j, value in enumerate(values):
            axes[1].text(j, bottoms[j]+value/2, str(value), ha='center', va='center', fontsize=9, color='white')
        bottoms += values
    axes[1].set(ylabel='Training images', title='Fixed K=2 country composition', ylim=(0, 8000))
    axes[1].legend(frameon=False, fontsize=9)
    save_figure(fig, '03_latent_evidence')

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    for i, row in enumerate(cost):
        x = float(row['train_wall_time_s']) / 3600
        y = float(row['all_voc_map50']) * 100
        axes[0].scatter(x, y, s=85, color=COLORS[i], marker=['o', 'o', 's', 's'][i])
        offset = (8, 10) if i != 1 else (8, -18)
        axes[0].annotate(LABELS[i] + f'\n{y:.4f}%', (x, y), xytext=offset, textcoords='offset points', fontsize=9)
    axes[0].set(xlabel='Training wall time (hours)', ylabel='Seven-domain VOC AP50 (%)',
                title='One paired seed; 100-epoch protocol', xlim=(12.5, 20.9), ylim=(40.2, 41.0))
    values = [int(r['total_training_parameters'])/1e6 for r in cost]
    axes[1].bar(range(4), values, color=COLORS, width=.65)
    axes[1].set(xticks=range(4), xticklabels=['Country\nmulti', 'Latent\nmulti', 'Country\nK-way', 'Latent\nK-way'],
                ylabel='Total resident training parameters (M)', title='Student + heads + frozen experts', ylim=(0, 42))
    for i, value in enumerate(values):
        axes[1].text(i, value+.8, f'{value:.2f}', ha='center')
    save_figure(fig, '04_accuracy_cost')

    fig, ax = plt.subplots(figsize=(8.8, 4.5), constrained_layout=True)
    protocols = ['stage1_standard_score', 'stage2_60_standard_score', 'stage2_100_standard_score']
    for side, label in enumerate(['Country multi', 'Latent multi']):
        ys = []
        for protocol in protocols:
            run = ['country_unique', 'latent_unique'][side] if protocol.startswith('stage1_') else RUNS[side]
            ys.append(next(r['voc_map50'] for r in macro if r['protocol']==protocol and r['experiment']==run and r['group']=='all'))
        xs = np.arange(3) + (side-.5)*.13
        ax.plot(xs, ys, 'o-', color=COLORS[side], label=label)
        for x, y in zip(xs, ys):
            ax.annotate(f'{y:.4f}', (x, y), xytext=(0, 10 if side else -17), textcoords='offset points', ha='center', fontsize=9)
    ax.set(xticks=range(3), xticklabels=['Stage 1\nseed 11 / 100 ep', 'Four-way\nseed 42 / 60 ep', 'Four-way replay\nseed 11 / 100 ep'],
           ylabel='Seven-domain VOC AP50 (%)', title='Standard confidence; protocols shown separately', ylim=(40.05, 41.12))
    ax.legend(frameon=False)
    save_figure(fig, '05_protocol_comparison')


def main():
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    macro, cost = archive_metrics()
    files, symbols = index_files()
    flow_figures()
    data_figures(macro, cost)
    print(f'Documentation assets: {len(macro)} macro rows, {files} files, {symbols} symbols, 5 figures (PNG/PDF/SVG).')


if __name__ == '__main__':
    main()
