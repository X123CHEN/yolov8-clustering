import csv
import datetime
import os
import time

import torch

import dg_config

SUMMARY_FIELDS = [
    'results_path', 'timestamp',
    'style_experiment', 'routing_mode', 'soft_temperature', 'confidence_floor', 'adv_weight',
    'phi', 'input_shape', 'num_classes', 'batch_size',
    'num_train', 'num_val', 'total_epochs',
    'best_epoch', 'best_map_50', 'best_val_loss', 'total_train_time_h',
    'avg_train_images_per_s', 'avg_epoch_time_s',
    'epoch_to_map45', 'train_time_to_map45_h',
    'peak_train_mem_GB', 'gpu_hours',
    'stable_loss_yolo', 'stable_loss_triplet', 'stable_loss_critic', 'stable_loss_generator',
    'params_M', 'gflops', 'model_size_MB', 'infer_latency_ms', 'infer_fps',
]

_PROFILE_NA = {
    'model_size_MB': 'N/A',
    'params_M': 'N/A',
    'gflops': 'N/A',
    'infer_latency_ms': 'N/A',
    'infer_fps': 'N/A',
}

STABLE_EPOCH_RATIO = 0.1
STABLE_EPOCH_MIN = 5
DEFAULT_MAP_THRESHOLD = 0.45


def _load_checkpoint(weights_path, device):
    try:
        return torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(weights_path, map_location=device)


def _checkpoint_params_m(state):
    param_keys = (
        k for k in state.keys()
        if not k.endswith('total_ops') and not k.endswith('total_params')
    )
    return f'{sum(state[k].numel() for k in param_keys) / 1e6:.3f}'


def _build_loaded_model(state, phi, input_shape, num_classes, device):
    from nets.yolo import YoloBody

    model = YoloBody(input_shape, num_classes, phi, pretrained=False).to(device)
    model_sd = model.state_dict()
    filtered = {
        k: v for k, v in state.items()
        if k in model_sd and model_sd[k].shape == v.shape
    }
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


def profile_weights(weights_path, phi, input_shape, num_classes,
                    test_interval=100, cuda=True):
    """基于 checkpoint 分项统计；thop 缺失时仅 gflops 为 N/A，其余仍尽量计算。"""
    result = dict(_PROFILE_NA)
    if not weights_path or not os.path.isfile(weights_path):
        return result

    result['model_size_MB'] = f'{os.path.getsize(weights_path) / (1024 ** 2):.3f}'

    use_cuda = cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    try:
        state = _load_checkpoint(weights_path, device)
        result['params_M'] = _checkpoint_params_m(state)
    except Exception as e:
        print(f'[metrics_summary] load checkpoint failed: {e}')
        return result

    model = None
    try:
        model = _build_loaded_model(state, phi, input_shape, num_classes, device)
    except Exception as e:
        print(f'[metrics_summary] build model failed: {e}')
        return result

    dummy = torch.randn(1, 3, input_shape[0], input_shape[1], device=device)
    try:
        from thop import profile
        with torch.no_grad():
            flops, _ = profile(model, (dummy,), verbose=False)
        result['gflops'] = f'{flops * 2 / 1e9:.3f}'
    except ImportError:
        print('[metrics_summary] thop not installed; gflops=N/A. Run: pip install thop')
    except Exception as e:
        print(f'[metrics_summary] gflops profiling failed: {e}')

    try:
        with torch.no_grad():
            for _ in range(10):
                model(dummy)
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(test_interval):
                model(dummy)
            if use_cuda:
                torch.cuda.synchronize()
            tact_time = (time.time() - t0) / test_interval
        result['infer_latency_ms'] = f'{tact_time * 1000:.3f}'
        result['infer_fps'] = f'{(1.0 / tact_time if tact_time > 0 else 0):.2f}'
    except Exception as e:
        print(f'[metrics_summary] inference benchmark failed: {e}')

    return result


def _row_float(row, key, default=None):
    raw = row.get(key, '')
    if raw in ('', None):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _mean_field(rows, key):
    vals = [_row_float(r, key) for r in rows]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def summarize_epochs(epoch_csv, map_threshold=DEFAULT_MAP_THRESHOLD):
    with open(epoch_csv, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'empty epoch csv: {epoch_csv}')

    best_map, best_map_epoch = -1.0, ''
    best_val, best_val_epoch = float('inf'), ''
    total_train_s = 0.0
    epoch_to_map45 = ''
    train_time_to_map45_s = 0.0
    cumulative_train_s = 0.0

    for row in rows:
        epoch = int(row['epoch'])
        val_loss = float(row['val_loss'])
        train_s = float(row['train_time_s'])
        total_train_s += train_s
        cumulative_train_s += train_s

        map_val = _row_float(row, 'map_50')
        if map_val is not None and map_val > best_map:
            best_map, best_map_epoch = map_val, epoch

        if val_loss < best_val:
            best_val, best_val_epoch = val_loss, epoch

        if not epoch_to_map45 and map_val is not None and map_val >= map_threshold:
            epoch_to_map45 = str(epoch)
            train_time_to_map45_s = cumulative_train_s

    avg_train_images_per_s = _mean_field(rows, 'train_images_per_s')
    avg_epoch_time_s = _mean_field(rows, 'epoch_time_s')
    peak_train_mem_gb = None
    peak_vals = [_row_float(r, 'peak_mem_GB') for r in rows]
    peak_vals = [v for v in peak_vals if v is not None and v > 0]
    if peak_vals:
        peak_train_mem_gb = max(peak_vals)

    n = len(rows)
    stable_n = max(STABLE_EPOCH_MIN, int(n * STABLE_EPOCH_RATIO))
    stable_rows = rows[-stable_n:]

    def _stable(key):
        m = _mean_field(stable_rows, key)
        return '' if m is None else f'{m:.6f}'

    return {
        'completed_epochs': n,
        'best_map_50': '' if best_map < 0 else f'{best_map:.6f}',
        'best_map_epoch': best_map_epoch,
        'best_val_loss': '' if best_val == float('inf') else f'{best_val:.6f}',
        'best_val_epoch': best_val_epoch,
        'total_train_time_h': f'{total_train_s / 3600:.4f}',
        'avg_train_images_per_s': '' if avg_train_images_per_s is None else f'{avg_train_images_per_s:.3f}',
        'avg_epoch_time_s': '' if avg_epoch_time_s is None else f'{avg_epoch_time_s:.3f}',
        'epoch_to_map45': epoch_to_map45,
        'train_time_to_map45_h': '' if not epoch_to_map45 else f'{train_time_to_map45_s / 3600:.4f}',
        'peak_train_mem_GB': '' if peak_train_mem_gb is None else f'{peak_train_mem_gb:.3f}',
        'stable_loss_yolo': _stable('loss_yolo'),
        'stable_loss_triplet': _stable('loss_triplet'),
        'stable_loss_critic': _stable('loss_critic'),
        'stable_loss_generator': _stable('loss_generator'),
    }


def _style_fields(args):
    style_cfg = getattr(args, 'style_cfg', None) or {}
    return {
        'style_experiment': (
            getattr(args, 'style_experiment', None)
            or style_cfg.get('experiment_name')
            or dg_config.STYLE_EXPERIMENT_NAME
        ),
        'routing_mode': (
            getattr(args, 'routing_mode', None)
            or style_cfg.get('mode', '')
        ),
        'soft_temperature': (
            getattr(args, 'soft_temperature', None)
            or style_cfg.get('temperature', dg_config.STYLE_SOFT_TEMPERATURE)
        ),
        'confidence_floor': (
            getattr(args, 'confidence_floor', None)
            or style_cfg.get('confidence_floor', dg_config.STYLE_CONFIDENCE_FLOOR)
        ),
        'adv_weight': (
            getattr(args, 'adv_weight', None)
            or style_cfg.get('adv_weight', dg_config.STYLE_ADV_LOSS_WEIGHT)
        ),
    }


def generate_summary(loss_dir, args):
    loss_dir = os.path.abspath(loss_dir)
    epoch_csv = os.path.join(loss_dir, 'metrics_epoch.csv')
    if not os.path.isfile(epoch_csv):
        raise FileNotFoundError(f'missing metrics_epoch.csv: {epoch_csv}')

    results_abs = os.path.abspath(args.results_path or os.path.dirname(loss_dir))
    if getattr(args, 'results_path_rel', False):
        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_path = os.path.relpath(results_abs, proj_root)
        if not results_path.startswith('.'):
            results_path = './' + results_path
    else:
        results_path = results_abs

    weights_path = os.path.abspath(
        args.weights_path or os.path.join(results_abs, 'best_epoch_weights.pth')
    )
    summary_csv = os.path.join(loss_dir, 'metrics_summary.csv')
    map_threshold = getattr(args, 'map_threshold', DEFAULT_MAP_THRESHOLD)

    epoch_stats = summarize_epochs(epoch_csv, map_threshold=map_threshold)
    weight_metrics = profile_weights(
        weights_path,
        args.phi,
        args.input_shape,
        args.num_classes,
        test_interval=getattr(args, 'fps_interval', 100),
        cuda=not getattr(args, 'cpu', False),
    )

    best_epoch = epoch_stats['best_map_epoch'] or epoch_stats['best_val_epoch']
    total_epochs = args.planned_epochs if args.planned_epochs is not None else epoch_stats['completed_epochs']
    num_gpus = getattr(args, 'num_gpus', 1) or 1
    gpu_hours = float(epoch_stats['total_train_time_h']) * num_gpus
    style_fields = _style_fields(args)

    row = {
        'results_path': results_path,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        **style_fields,
        'phi': args.phi,
        'input_shape': f'{args.input_shape[0]},{args.input_shape[1]}',
        'num_classes': args.num_classes,
        'batch_size': args.batch_size,
        'num_train': args.num_train,
        'num_val': args.num_val,
        'total_epochs': total_epochs,
        'best_epoch': best_epoch,
        'best_map_50': epoch_stats['best_map_50'],
        'best_val_loss': epoch_stats['best_val_loss'],
        'total_train_time_h': epoch_stats['total_train_time_h'],
        'avg_train_images_per_s': epoch_stats['avg_train_images_per_s'],
        'avg_epoch_time_s': epoch_stats['avg_epoch_time_s'],
        'epoch_to_map45': epoch_stats['epoch_to_map45'],
        'train_time_to_map45_h': epoch_stats['train_time_to_map45_h'],
        'peak_train_mem_GB': epoch_stats['peak_train_mem_GB'],
        'gpu_hours': f'{gpu_hours:.4f}',
        'stable_loss_yolo': epoch_stats['stable_loss_yolo'],
        'stable_loss_triplet': epoch_stats['stable_loss_triplet'],
        'stable_loss_critic': epoch_stats['stable_loss_critic'],
        'stable_loss_generator': epoch_stats['stable_loss_generator'],
        'params_M': weight_metrics['params_M'],
        'gflops': weight_metrics['gflops'],
        'model_size_MB': weight_metrics['model_size_MB'],
        'infer_latency_ms': weight_metrics['infer_latency_ms'],
        'infer_fps': weight_metrics['infer_fps'],
    }

    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(row)

    print(f'Generated: {summary_csv}')
    print(f'  completed_epochs = {epoch_stats["completed_epochs"]}')
    print(f'  best_epoch       = {best_epoch}')
    print(f'  best_map_50      = {epoch_stats["best_map_50"]}')
    print(f'  gpu_hours        = {row["gpu_hours"]}')
    print(f'  params_M         = {weight_metrics["params_M"]}')
    print(f'  gflops           = {weight_metrics["gflops"]}')
    print(f'  infer_fps        = {weight_metrics["infer_fps"]}')
    return summary_csv
