import os
import time

import torch
from tqdm import tqdm
import numpy as np
from utils.utils import get_lr
from core import loss
from core.loss import SupConLoss_feats
from collections import OrderedDict
from nets.evaluate_maddg import evaluate
from dg_config import CFG
from utils.callbacks import append_metrics_epoch

def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
# def fit_one_epoch(model_train, model, ema, ema_backbone, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir,Unfreeze_batch_size,

                  FeatEmbder,Discriminator1,Discriminator2,Discriminator3,PreFeatExtorS1,PreFeatExtorS2,PreFeatExtorS3,
                  # backbone,FeatEmbder,Discriminator1,Discriminator2,Discriminator3,PreFeatExtorS1,PreFeatExtorS2,PreFeatExtorS3,
                  current_epoch,criterionAdv,args,TripletLossCal,optimizer_critic1,optimizer_critic2,optimizer_critic3,summary_writer,saver,
                  local_rank=0):

    loss        = 0
    val_loss    = 0
    epoch_start = time.time()
    train_start = time.time()
    sum_loss_yolo = 0.0
    sum_loss_triplet = 0.0
    sum_loss_critic = 0.0
    sum_loss_generator = 0.0
    metric_steps = 0

    if cuda and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(local_rank)

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    # backbone.train()
    model_train.train()

    ####wrq修改代码，
    # FeatExtor.train()
    # FeatEmbder.train()
    Discriminator1.train()
    Discriminator2.train()
    Discriminator3.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        loss_yolo_weighted = torch.tensor(0.0)
        Loss_G = torch.tensor(0.0)
        Loss_triplet = 0
        loss_critic1 = torch.zeros(1)
        loss_critic2 = torch.zeros(1)
        loss_critic3 = torch.zeros(1)
        loss_generator1 = torch.zeros(1)
        loss_generator2 = torch.zeros(1)
        loss_generator3 = torch.zeros(1)
        # images, bboxes = batch
        images0,bboxes0, images1,bboxes1, images2,bboxes2 = batch #日本，印度，美国
        # images0_trip,bboxes0_trip, images1_trip,bboxes1_trip, images2_trip,bboxes2_trip= batch #日本，印度，美国
        feat_20_20_x_y_japan, feat_20_20_x_y_india, feat_20_20_x_y_usa, feat_40_40_x_y_japan, feat_40_40_x_y_india, feat_40_40_x_y_usa = box2feat_xywh(bboxes0, bboxes1, bboxes2)
        
        with torch.no_grad():
            if cuda:
                images0     = images0.cuda(local_rank)
                images1     = images1.cuda(local_rank)
                images2     = images2.cuda(local_rank)
                # pre_img1,pre_img2,pre_img3 = np.split(images, 3, axis=0)
                pre_img1, pre_img2, pre_img3 = images0, images1, images2#日本，印度，美国
                #A域：Japan；B域：India; C域：USA
                bboxes0     = bboxes0.cuda(local_rank)
                bboxes1     = bboxes1.cuda(local_rank)
                bboxes2     = bboxes2.cuda(local_rank)
        # with torch.no_grad():
            if CFG.use_adv:
                pre_feat_ext1 = PreFeatExtorS1(pre_img1)[2]  #日本  # 一定要改2  Backbone那边2才是对应20*20的特征aaaa！！
                pre_feat_ext2 = PreFeatExtorS2(pre_img2)[2]  #印度
                pre_feat_ext3 = PreFeatExtorS3(pre_img3)[2]  #美国

        #----------------------#
        #   清零梯度
        #----------------------#
        # optimizer.zero_grad()  #需要的时候在做梯度清零，不要一堆的优化器梯度清零堆在一起，需要的时候再拿出来。
        if not fp16:
            # optimizer_DG_conf.zero_grad()
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            images0_list = torch.chunk(images0, Unfreeze_batch_size, dim=0)
            images1_list = torch.chunk(images1, Unfreeze_batch_size, dim=0)
            images2_list = torch.chunk(images2, Unfreeze_batch_size, dim=0)
            images = []
            # 使用zip将三个列表中的张量进行轮流遍历
            for img0, img1, img2 in zip(images0_list, images1_list, images2_list):
                concatenated_tensor = torch.cat([img0, img1, img2], dim=0)
                images.append(concatenated_tensor)
            # images = torch.cat((images0, images1, images2), dim=0)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            images = torch.cat(images, dim=0).to(device)

            # dbox, cls, origin_cls, anchors, strides
            # feat_ext1,feat_ext2,feat_ext3 = backbone(images)
            # outputs = model_train(feat_ext1,feat_ext2,feat_ext3)
            outputs = model_train(images)
            feat_ext3 = outputs[5]       #20 feature map
            feat_ext2 = outputs[6]       #40 feature map
            # feat_ext1 = outputs[7]     #80 feature map

            # #----------------------#
            # #   前向传播
            # ----------------------#
            # optimizer.zero_grad()
            all_bboxes = []
            all_bboxes.extend(bboxes0)
            all_bboxes.extend(bboxes1)
            all_bboxes.extend(bboxes2)

            # 使用 sorted 函数对 all_bboxes 进行排序
            bboxes = torch.stack(sorted(all_bboxes, key=sort_key), dim=0)
            # outputs = model_train(feat_ext1.detach(), feat_ext2.detach(), feat_ext3.detach())
            # outputs = model_train(feat_ext1.detach(), feat_ext2.detach(), feat_ext3.detach())
            outputs = outputs[:5]
            loss_yolo_raw = yolo_loss(outputs, bboxes)
            # ============ domain generalization and yolo_model supervision ============#
            # ************************* confusion all **********************************#
            # predict on generator                       # yolov8 骨干提出3个不同尺度的特征，然后对三个尺度的特征都进行生成损失的计算
            loss_generator1 = torch.zeros((), device=feat_ext3.device)
            loss_generator2 = torch.zeros((), device=feat_ext3.device)
            loss_generator3 = torch.zeros((), device=feat_ext3.device)
            loss_critic1 = torch.zeros((), device=feat_ext3.device)
            loss_critic2 = torch.zeros((), device=feat_ext3.device)
            loss_critic3 = torch.zeros((), device=feat_ext3.device)
            if CFG.use_adv:
                loss_generator1 = criterionAdv(Discriminator1(feat_ext3), True)
                loss_generator2 = criterionAdv(Discriminator2(feat_ext3), True)
                loss_generator3 = criterionAdv(Discriminator3(feat_ext3), True)

            ########## cross-domain triplet loss #########

            if CFG.align_type != 'none' and epoch >= args.dongjie_epochs:
            # if (val_loss / epoch_step_val) <= min(loss_history.val_loss):
                feat_embd_j_20 = feat_xywh2feat_tensor_upgrade1(feat_20_20_x_y_japan, feat_ext3)  
                feat_embd_i_20 = feat_xywh2feat_tensor_upgrade1(feat_20_20_x_y_india, feat_ext3)
                feat_embd_u_20 = feat_xywh2feat_tensor_upgrade1(feat_20_20_x_y_usa, feat_ext3)
                feat_embd_j_40 = feat_xywh2feat_tensor_upgrade1(feat_40_40_x_y_japan, feat_ext2)
                feat_embd_i_40 = feat_xywh2feat_tensor_upgrade1(feat_40_40_x_y_india, feat_ext2)
                feat_embd_u_40 = feat_xywh2feat_tensor_upgrade1(feat_40_40_x_y_usa, feat_ext2)
                # feat_embd_j_80 = feat_xywh2feat_tensor(feat_80_80_x_y_japan, feat_ext1)
                # feat_embd_i_80 = feat_xywh2feat_tensor(feat_80_80_x_y_india, feat_ext1)
                # feat_embd_u_80 = feat_xywh2feat_tensor(feat_80_80_x_y_usa, feat_ext1)


                lab_list_j = feat_20_20_x_y_japan[:, 1]
                lab_list_i = feat_20_20_x_y_india[:, 1]
                lab_list_u = feat_20_20_x_y_usa[:, 1]

                Loss_triplet_list = []
                # Loss_triplet1_list = []
                index = 0
                # index1 = 0
                max_num = max(len(lab_list_j), len(lab_list_i), len(lab_list_u))
                while index < max_num:
                    lab_j, lab_i, lab_u = \
                            lab_list_j[index % len(lab_list_j):(index % len(lab_list_j)) + max_num], \
                            lab_list_i[index % len(lab_list_i):(index % len(lab_list_i)) + max_num], \
                            lab_list_u[index % len(lab_list_u):(index % len(lab_list_u)) + max_num]
                    embd_j, embd_i, embd_u = \
                            feat_embd_j_20[index % len(feat_embd_j_20):(index % len(feat_embd_j_20)) + max_num], \
                            feat_embd_i_20[index % len(feat_embd_i_20):(index % len(feat_embd_i_20)) + max_num], \
                            feat_embd_u_20[index % len(feat_embd_u_20):(index % len(feat_embd_u_20)) + max_num]

                    embd_j_40, embd_i_40, embd_u_40 = \
                            feat_embd_j_40[index % len(feat_embd_j_40):(index % len(feat_embd_j_40)) + max_num], \
                            feat_embd_i_40[index % len(feat_embd_i_40):(index % len(feat_embd_i_40)) + max_num], \
                            feat_embd_u_40[index % len(feat_embd_u_40):(index % len(feat_embd_u_40)) + max_num]

                    # embd_j_80, embd_i_80, embd_u_80 = \
                    #         feat_embd_j_80[index % len(feat_embd_j_80):(index % len(feat_embd_j_80)) + max_num], \
                    #         feat_embd_i_80[index % len(feat_embd_i_80):(index % len(feat_embd_i_80)) + max_num], \
                    #         feat_embd_u_80[index % len(feat_embd_u_80):(index % len(feat_embd_u_80)) + max_num]

                    if CFG.align_type == 'supcon':
                        loss_tri_20 = SupConLoss_feats(torch.cat([embd_j, embd_i, embd_u], 0),
                                                       torch.cat([lab_j, lab_i, lab_u], 0), CFG.supcon_temp)
                        loss_tri_40 = SupConLoss_feats(torch.cat([embd_j_40, embd_i_40, embd_u_40], 0),
                                                       torch.cat([lab_j, lab_i, lab_u], 0), CFG.supcon_temp)
                    else:
                        loss_tri_20 = TripletLossCal(args, embd_j, embd_i, embd_u, lab_j, lab_i, lab_u)
                        loss_tri_40 = TripletLossCal(args, embd_j_40, embd_i_40, embd_u_40, lab_j, lab_i, lab_u)
                    # loss_tri2 = TripletLossCal(args, embd_j_80, embd_i_80, embd_u_80, lab_j, lab_i, lab_u)
                    Loss_triplet_list.append(loss_tri_20)
                    Loss_triplet_list.append(loss_tri_40)
                    # Loss_triplet_list.append(loss_tri2)
                    index += max_num

                Loss_triplet = sum(Loss_triplet_list) / len(Loss_triplet_list) if len(Loss_triplet_list) > 0 else 0
            else:
                Loss_triplet = 0
                
            Loss_gen_feat_tgt = args.W_genave * (loss_generator1 + loss_generator2 + loss_generator3)
            # Loss_G = args.W_trip * Loss_triplet
            Loss_G = args.W_gen * Loss_gen_feat_tgt          # 对抗关闭时为 0 张量
            if torch.is_tensor(Loss_triplet):                # 内容对齐(triplet/supcon)开启时才加
                Loss_G = Loss_G + args.W_trip * Loss_triplet
            loss_yolo_weighted = args.W_yolo * loss_yolo_raw
            loss_value = loss_yolo_weighted + Loss_G
            triplet_val = Loss_triplet.item() if torch.is_tensor(Loss_triplet) else float(Loss_triplet)
            critic_val = (loss_critic1 + loss_critic2 + loss_critic3).item() / 3.0
            generator_val = (loss_generator1 + loss_generator2 + loss_generator3).item() / 3.0
            sum_loss_yolo += loss_yolo_weighted.item()
            sum_loss_triplet += triplet_val
            sum_loss_critic += critic_val
            sum_loss_generator += generator_val
            metric_steps += 1
            # Loss_G.backward()
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=2.0)  # clip gradients
            # torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
            optimizer.step()

            if CFG.use_adv:
                feat_src1 = torch.cat([pre_feat_ext1, pre_feat_ext1, pre_feat_ext1], 0)
                feat_src2 = torch.cat([pre_feat_ext2, pre_feat_ext2, pre_feat_ext2], 0)
                feat_src3 = torch.cat([pre_feat_ext3, pre_feat_ext3, pre_feat_ext3], 0)
                feat_fake = feat_ext3.detach()
                critic_scale = 100  # czh修改，将1000改为100

                with torch.no_grad():
                    real_loss1 = criterionAdv(Discriminator1(feat_src1), True)
                    fake_loss1 = criterionAdv(Discriminator1(feat_fake), False)
                    loss_critic1 = 0.5 * (real_loss1 + fake_loss1) * critic_scale
                    real_loss2 = criterionAdv(Discriminator2(feat_src2), True)
                    fake_loss2 = criterionAdv(Discriminator2(feat_fake), False)
                    loss_critic2 = 0.5 * (real_loss2 + fake_loss2) * critic_scale
                    real_loss3 = criterionAdv(Discriminator3(feat_src3), True)
                    fake_loss3 = criterionAdv(Discriminator3(feat_fake), False)
                    loss_critic3 = 0.5 * (real_loss3 + fake_loss3) * critic_scale

                if (iteration + 1) % args.critic_update_interval == 0:
                    optimizer_critic1.zero_grad()
                    real_loss1 = criterionAdv(Discriminator1(feat_src1), True)
                    fake_loss1 = criterionAdv(Discriminator1(feat_fake), False)
                    loss_critic1 = 0.5 * (real_loss1 + fake_loss1) * critic_scale
                    loss_critic1.backward()
                    optimizer_critic1.step()

                    optimizer_critic2.zero_grad()
                    real_loss2 = criterionAdv(Discriminator2(feat_src2), True)
                    fake_loss2 = criterionAdv(Discriminator2(feat_fake), False)
                    loss_critic2 = 0.5 * (real_loss2 + fake_loss2) * critic_scale
                    loss_critic2.backward()
                    optimizer_critic2.step()

                    optimizer_critic3.zero_grad()
                    real_loss3 = criterionAdv(Discriminator3(feat_src3), True)
                    fake_loss3 = criterionAdv(Discriminator3(feat_fake), False)
                    loss_critic3 = 0.5 * (real_loss3 + fake_loss3) * critic_scale
                    loss_critic3.backward()
                    optimizer_critic3.step()
            #----------------------#
            #   反向传播
            #----------------------#
        if ema:
            ema.update(model_train)
        loss += loss_value.item()

        # ============ tensorboard the log info ============#
        info = {
            'Loss_yolo': loss_yolo_weighted.item(),
            'Loss_G': Loss_G.item(),
            # 'Loss_triplet': Loss_triplet.item(),
            'Loss_triplet': Loss_triplet.item() if torch.is_tensor(Loss_triplet) else 0,
            'loss_critic1': loss_critic1.item(),
            'loss_generator1': loss_generator1.item(),
            'loss_critic2': loss_critic2.item(),
            'loss_generator2': loss_generator2.item(),
            'loss_critic3': loss_critic3.item(),
            'loss_generator3': loss_generator3.item(),
        }

        global_step = 0
        for tag, value in info.items():
            summary_writer.add_scalar(tag, value, global_step)
        # ============ print the log info ============#

        if (iteration + 1) % args.log_step == 0:
            errors = OrderedDict([
                ('Loss_triplet', Loss_triplet.item() if torch.is_tensor(Loss_triplet) else 0),
                # ('Loss_triplet', Loss_triplet.item()),
                ('Loss_yolo', loss_value.item()),
                ('Loss_G', Loss_G.item()),
                ('loss_critic1', loss_critic1.item()),
                ('loss_generator1', loss_generator1.item()),
                ('loss_critic2', loss_critic2.item()),
                ('loss_generator2', loss_generator2.item()),
                ('loss_critic3', loss_critic3.item()),
                ('loss_generator3', loss_generator3.item())])

            saver.print_current_errors(epoch=(epoch + 1), i=(iteration + 1), errors=errors)
        global_step += 1
        if local_rank == 0:
            pbar.set_postfix(**{'yolo_loss'  : loss / (iteration + 1),
                                'yolo_lr'    : get_lr(optimizer),
                                # 'Loss_triplet': Loss_triplet,
                                # 'Loss_gen_feat_tgt': Loss_gen_feat_tgt,
                                # 'Loss_G': Loss_G,
                                # 'loss_critic1': loss_critic1,
                                # 'loss_critic2': loss_critic2,
                                # 'loss_critic3': loss_critic3,
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
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
        # backbone_eval = ema_backbone.ema
    else:
        # backbone_eval = backbone.eval()
        model_train_eval = model_train.eval()

    val_start = time.time()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, bboxes = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            # feat        = backbone(images)
            # outputs     = model_train_eval(feat[0],feat[1],feat[2])
            outputs = model_train_eval(images)
            loss_value  = yolo_loss(outputs[:5], bboxes)
            # loss_value  = yolo_loss(outputs, bboxes)

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    val_time_s = time.time() - val_start

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')

        train_loss = loss / epoch_step
        val_loss_avg = val_loss / epoch_step_val
        loss_history.append_loss(epoch + 1, train_loss, val_loss_avg)
        map_50 = eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        epoch_time_s = time.time() - epoch_start
        images_per_step = Unfreeze_batch_size * 3
        train_images_per_s = (epoch_step * images_per_step / train_time_s) if train_time_s > 0 else 0

        avg_loss_yolo = sum_loss_yolo / metric_steps if metric_steps else 0.0
        avg_loss_triplet = sum_loss_triplet / metric_steps if metric_steps else 0.0
        avg_loss_critic = sum_loss_critic / metric_steps if metric_steps else 0.0
        avg_loss_generator = sum_loss_generator / metric_steps if metric_steps else 0.0

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
            'loss_triplet': f'{avg_loss_triplet:.6f}',
            'loss_critic': f'{avg_loss_critic:.6f}',
            'loss_generator': f'{avg_loss_generator:.6f}',
            'peak_mem_GB': f'{peak_mem_GB:.3f}',
        })

        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (train_loss, val_loss_avg))

        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            # save_state_dict_backbone = backbone.state_dict()
            save_state_dict_head = ema.ema.state_dict()
        else:
            # save_state_dict_backbone = backbone.state_dict()
            save_state_dict_head = model.state_dict()

        # if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        #     # torch.save(save_state_dict_backbone, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f_backbone.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
        #     # torch.save(save_state_dict_head, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f_head.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
        #     torch.save(save_state_dict_head, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if (epoch + 1) <= 50 or (epoch + 1) == Epoch:
            torch.save(save_state_dict_head, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
                epoch + 1, train_loss, val_loss_avg)))
        if (epoch + 1) % 50 == 0 or epoch + 1 == Epoch: #每50个epoch保存一次权重。
            torch.save(save_state_dict_head, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
                epoch + 1, train_loss, val_loss_avg)))

        if len(loss_history.val_loss) <= 1 or val_loss_avg <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            # print('Save best model to best_epoch_weights_all.pth')
            # torch.save(save_state_dict_backbone, os.path.join(save_dir, "best_epoch_weights_backbone.pth"))
            # torch.save(save_state_dict_head, os.path.join(save_dir, "best_epoch_weights_head.pth"))
            torch.save(save_state_dict_head, os.path.join(save_dir, "best_epoch_weights.pth"))

        # torch.save(save_state_dict_backbone, os.path.join(save_dir, "last_epoch_weights_backbone.pth"))
        # torch.save(save_state_dict_head, os.path.join(save_dir, "last_epoch_weights_head.pth"))
        torch.save(save_state_dict_head, os.path.join(save_dir, "last_epoch_weights.pth"))

#---------------------------------------------------------------------------------------------------------------------------#



















# 定义一个排序关键字函数，按张量的第一个数字进行排序
def sort_key(tensor):
    return tensor[0]# 如果张量是一维的



def box2feat_xywh(bboxes0,bboxes1,bboxes2):
    feat_20_20_x_y_japan = torch.cat([bboxes0[:, :2],
                                          (torch.round((bboxes0[:, 2]) * 20)).unsqueeze(1),      #x
                                          (torch.round((bboxes0[:, 3]) * 20)).unsqueeze(1),      #y
                                          (torch.round((bboxes0[:, 4]) * 20)).unsqueeze(1),      #w
                                          (torch.round((bboxes0[:, 5]) * 20)).unsqueeze(1),      #h
                                          ], dim=1)
    feat_20_20_x_y_india = torch.cat([bboxes1[:, :2],
                                          (torch.round((bboxes1[:, 2]) * 20)).unsqueeze(1),       
                                          (torch.round((bboxes1[:, 3]) * 20)).unsqueeze(1),
                                          (torch.round((bboxes1[:, 4]) * 20)).unsqueeze(1),
                                          (torch.round((bboxes1[:, 5]) * 20)).unsqueeze(1),
                                          ], dim=1)
    feat_20_20_x_y_usa = torch.cat([bboxes2[:, :2],
                                        (torch.round((bboxes2[:, 2]) * 20)).unsqueeze(1),
                                        (torch.round((bboxes2[:, 3]) * 20)).unsqueeze(1),
                                        (torch.round((bboxes2[:, 4]) * 20)).unsqueeze(1),
                                        (torch.round((bboxes2[:, 5]) * 20)).unsqueeze(1),
                                        ], dim=1)

    feat_40_40_x_y_japan = torch.cat([bboxes0[:, :2],
                                          (torch.round((bboxes0[:, 2]) * 40)).unsqueeze(1),
                                          (torch.round((bboxes0[:, 3]) * 40)).unsqueeze(1),
                                          (torch.round((bboxes0[:, 4]) * 40)).unsqueeze(1),
                                          (torch.round((bboxes0[:, 5]) * 40)).unsqueeze(1),
                                          ], dim=1)
    feat_40_40_x_y_india = torch.cat([bboxes1[:, :2],
                                          (torch.round((bboxes1[:, 2]) * 40)).unsqueeze(1),
                                          (torch.round((bboxes1[:, 3]) * 40)).unsqueeze(1),
                                          (torch.round((bboxes1[:, 4]) * 40)).unsqueeze(1),
                                          (torch.round((bboxes1[:, 5]) * 40)).unsqueeze(1),
                                          ], dim=1)
    feat_40_40_x_y_usa = torch.cat([bboxes2[:, :2],
                                        (torch.round((bboxes2[:, 2]) * 40)).unsqueeze(1),
                                        (torch.round((bboxes2[:, 3]) * 40)).unsqueeze(1),
                                        (torch.round((bboxes2[:, 4]) * 40)).unsqueeze(1),
                                        (torch.round((bboxes2[:, 5]) * 40)).unsqueeze(1),
                                        ], dim=1)
    # feat_80_80_x_y_japan = torch.cat([bboxes0[:, :2],
    #                                   (torch.round((bboxes0[:, 2]) * 80)).unsqueeze(1),
    #                                   (torch.round((bboxes0[:, 3]) * 80)).unsqueeze(1)], dim=1)
    # feat_80_80_x_y_india = torch.cat([bboxes1[:, :2],
    #                                   (torch.round((bboxes1[:, 2]) * 80)).unsqueeze(1),
    #                                   (torch.round((bboxes1[:, 3]) * 80)).unsqueeze(1)], dim=1)
    # feat_80_80_x_y_usa = torch.cat([bboxes2[:, :2],
    #                                 (torch.round((bboxes2[:, 2]) * 80)).unsqueeze(1),
    #                                 (torch.round((bboxes2[:, 3]) * 80)).unsqueeze(1)], dim=1)
    return feat_20_20_x_y_japan, feat_20_20_x_y_india, feat_20_20_x_y_usa, feat_40_40_x_y_japan, feat_40_40_x_y_india, feat_40_40_x_y_usa

def feat_xywh2feat_tensor(feat_20_20_x_y_country, feat_extx):
    # 提取20*20的特征图对应的bbox的特征，用于triplet_loss   
    new_feats_X0_country = []
    for row in feat_20_20_x_y_country:
        img_index = int(row[0])
        x_position = int(row[2]) - 1
        y_position = int(row[3]) - 1
        feature_vector = feat_extx[img_index, :, x_position, y_position]
        new_feats_X0_country.append(feature_vector)
    new_feats_X0_country = torch.stack(new_feats_X0_country)
    return new_feats_X0_country

import math
def feat_xywh2feat_tensor_upgrade1(feat_20_20_x_y_country, feat_extx):
    # 提取20*20的特征图对应的bbox的特征，用于triplet_loss   
    new_feats_X0_country = []
    for row in feat_20_20_x_y_country:
        img_index = int(row[0])
        x_position = row[2]-1
        y_position = row[3]-1
        w = row[4]
        h = row[5]
        x1 = max(math.floor(x_position - w/2),0)
        x2 = max(math.floor(x_position + w/2),0)
        y1 = max(math.floor(y_position - h/2),0)
        y2 = max(math.floor(y_position + h/2),0)

        # if x1 != x2 and y1 != y2:
        #     feature_vector = feat_extx[img_index, :, x1:x2, y1:y2]
        # elif x1 == x2:
        #     # print(int(x_mid))
        #     # print(int(y1),int(y2))
        #     feature_vector = feat_extx[img_index, :, int(x_position), int(y_position)]
        #     # feature_vector = feat_extx[img_index, :, int(x_position), y1:y2]
        # elif y1 == y2:
        #     # print(int(x1),int(x2))
        #     # print(int(y_mid))
        #     feature_vector = feat_extx[img_index, :, int(x_position), int(y_position)]
        #     # feature_vector = feat_extx[img_index, :, x1:x2, int(y_position)]
        # elif y1 ==y2 and x1 ==x2:
        #     # print(int(x_mid))
        #     # print(int(y_mid))
        #     feature_vector = feat_extx[img_index, :, int(x_position), int(y_position)]
        if x1 != x2 and y1 != y2:
            feature_vector = feat_extx[img_index, :, x1:x2, y1:y2]
        elif x1 == x2 or y1 == y2:
            # print(int(x_mid))
            # print(int(y1),int(y2))
            feature_vector = feat_extx[img_index, :, int(x_position), int(y_position)]
        feature_vector = avgpool_feature(feature_vector)
        new_feats_X0_country.append(feature_vector)
    new_feats_X0_country = torch.stack(new_feats_X0_country)
    return new_feats_X0_country


import torch.nn as nn
def avgpool_feature(feature):
    if len(feature.shape) == 1:
        # 如果特征向量是一维的，保持原向量
        return feature
    elif len(feature.shape) == 2:
        # 如果特征向量是二维的，进行平均池化变成512，1，然后展平为512
        return feature.mean(dim=1)
        # return feature.mean(dim=1, keepdim=True).unsqueeze(0)
    elif len(feature.shape) == 3:
        # 如果特征向量是三维的，进行平均池化变成512，1，1，然后展平为512
        pooling_layer = nn.AdaptiveAvgPool2d((1, 1))
        return pooling_layer(feature).view(-1)
    else:
        raise ValueError("不支持的特征形状")
    