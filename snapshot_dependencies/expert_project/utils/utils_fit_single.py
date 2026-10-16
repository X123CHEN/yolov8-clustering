import os
import time
import json

import torch
from tqdm import tqdm

from utils.callbacks import append_metrics_epoch
from utils.utils import get_lr


def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir,
                  Unfreeze_batch_size, local_rank=0):
    loss = 0
    val_loss = 0
    val_images = 0
    epoch_start = time.time()
    train_start = time.time()
    sum_loss_yolo = 0.0
    metric_steps = 0

    if cuda and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(local_rank)

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, bboxes = batch
        if cuda:
            images = images.cuda(local_rank)
            bboxes = bboxes.cuda(local_rank)

        optimizer.zero_grad()

        if not fp16:
            outputs = model_train(images)
            loss_value = yolo_loss(outputs[:5], bboxes)
            if not torch.isfinite(loss_value):
                raise RuntimeError(f'Non-finite training loss at epoch {epoch + 1}, batch {iteration}')
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=2.0)
            optimizer.step()
        else:
            from torch.cuda.amp import autocast

            with autocast():
                outputs = model_train(images)
                loss_value = yolo_loss(outputs[:5], bboxes)
            if not torch.isfinite(loss_value):
                raise RuntimeError(f'Non-finite training loss at epoch {epoch + 1}, batch {iteration}')
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

        if ema:
            ema.update(model_train)

        loss += loss_value.item()
        sum_loss_yolo += loss_value.item()
        metric_steps += 1

        if local_rank == 0:
            pbar.set_postfix(**{
                'yolo_loss': loss / (iteration + 1),
                'yolo_lr': get_lr(optimizer),
            })
            pbar.update(1)

    train_time_s = time.time() - train_start
    peak_mem_GB = 0.0
    if cuda and torch.cuda.is_available():
        peak_mem_GB = torch.cuda.max_memory_allocated(local_rank) / (1024 ** 3)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train_eval = ema.ema if ema else model_train.eval()

    val_start = time.time()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        images, bboxes = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)

            optimizer.zero_grad()
            outputs = model_train_eval(images)
            loss_value = yolo_loss(outputs[:5], bboxes)

        if not torch.isfinite(loss_value):
            raise RuntimeError(f'Non-finite validation loss at epoch {epoch + 1}, batch {iteration}')
        val_loss += loss_value.item() * images.shape[0]
        val_images += images.shape[0]
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / val_images})
            pbar.update(1)

    val_time_s = time.time() - val_start

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')

        train_loss = loss / epoch_step
        if val_images != len(gen_val.dataset):
            raise RuntimeError(f'Incomplete validation: {val_images}/{len(gen_val.dataset)} images')
        val_loss_avg = val_loss / val_images
        loss_history.append_loss(epoch + 1, train_loss, val_loss_avg)
        map_50 = eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        epoch_time_s = time.time() - epoch_start
        images_per_step = Unfreeze_batch_size
        train_images_per_s = (epoch_step * images_per_step / train_time_s) if train_time_s > 0 else 0
        avg_loss_yolo = sum_loss_yolo / metric_steps if metric_steps else 0.0

        append_metrics_epoch(loss_history.log_dir, {
            'epoch': epoch + 1,
            'train_loss': f'{train_loss:.6f}',
            'val_loss': f'{val_loss_avg:.6f}',
            'map_50': map_50 if map_50 is not None else '',
            'train_time_s': f'{train_time_s:.3f}',
            'val_time_s': f'{val_time_s:.3f}',
            'epoch_time_s': f'{epoch_time_s:.3f}',
            'steps': epoch_step,
            'images_per_step': images_per_step,
            'train_images_per_s': f'{train_images_per_s:.3f}',
            'loss_yolo': f'{avg_loss_yolo:.6f}',
            'loss_triplet': '0.000000',
            'loss_critic': '0.000000',
            'loss_generator': '0.000000',
            'peak_mem_GB': f'{peak_mem_GB:.3f}',
        })

        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (train_loss, val_loss_avg))

        save_state_dict = ema.ema.state_dict() if ema else model.state_dict()

        if (epoch + 1) <= 50 or (epoch + 1) == Epoch:
            torch.save(save_state_dict, os.path.join(
                save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, train_loss, val_loss_avg)))
        if (epoch + 1) % 50 == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(
                save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, train_loss, val_loss_avg)))

        if len(loss_history.val_loss) <= 1 or val_loss_avg <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            with open(os.path.join(save_dir, 'best_checkpoint.json'), 'w', encoding='utf-8') as f:
                json.dump(dict(epoch=epoch + 1, val_loss=val_loss_avg,
                               criterion='minimum val_loss', validation_images=val_images,
                               weights='best_epoch_weights.pth'), f, indent=2)

        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
