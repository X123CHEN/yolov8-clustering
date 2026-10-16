import os

import torch
from tqdm import tqdm
import numpy as np
from utils.utils import get_lr
from core import loss
from collections import OrderedDict
from nets.evaluate_maddg import evaluate

def fit_one_epoch(model_train, head,backbone, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir,Unfreeze_batch_size,

                  Discriminator1,Discriminator2,Discriminator3,PreFeatExtorS1,PreFeatExtorS2,PreFeatExtorS3,
                  current_epoch,criterionAdv,args,TripletLossCal,optimizer_DG_conf,optimizer_critic1,optimizer_critic2,optimizer_critic3,summary_writer,saver,
                  local_rank=0):
    loss        = 0
    val_loss    = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    backbone.train()
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
        # images, bboxes = batch
        images0,bboxes0, images1,bboxes1, images2,bboxes2 = batch #日本，印度，美国
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
            pre_feat_ext1 = PreFeatExtorS1(pre_img1)[2]  #日本  # 一定要改2  Backbone那边2才是对应20*20的特征aaaa！！
            pre_feat_ext2 = PreFeatExtorS2(pre_img2)[2]  #印度
            pre_feat_ext3 = PreFeatExtorS3(pre_img3)[2]  #美国

        #----------------------#
        #   清零梯度
        #----------------------#
        # optimizer.zero_grad()  #需要的时候在做梯度清零，不要一堆的优化器梯度清零堆在一起，需要的时候再拿出来。
        if not fp16:
            optimizer_DG_conf.zero_grad()
            # optimizer.zero_grad()
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
            feat_ext1,feat_ext2,feat_ext3 = backbone(images)

            # ============ domain generalization and yolo_model supervision ============#
            # ************************* confusion all **********************************#
            # predict on generator                       # yolov8 骨干提出3个不同尺度的特征，然后对三个尺度的特征都进行生成损失的计算
            loss_generator1 = criterionAdv(Discriminator1(feat_ext3), True)
            loss_generator2 = criterionAdv(Discriminator2(feat_ext3), True)
            loss_generator3 = criterionAdv(Discriminator3(feat_ext3), True)

            ########## cross-domain triplet loss #########
            # index = 0
            # Loss_triplet_list = []
            # while index + 2 < len(lab_list):
            #     lab1, lab2, lab3 = lab_list[index], lab_list[index + 1], lab_list[index + 2]
            #     loss_tri = TripletLossCal(args, feat_embd, lab1, lab2, lab3)
            #     Loss_triplet_list.append(loss_tri)
            #     index += 3
            # Loss_triplet = sum(Loss_triplet_list) / len(Loss_triplet_list)
            Loss_gen_feat_tgt = args.W_genave * (loss_generator1 + loss_generator2 + loss_generator3)
            Loss_G = args.W_gen * Loss_gen_feat_tgt
            Loss_G.backward()
            optimizer_DG_conf.step()
            # ************************* confusion domain 1 with 2,3 **********************************#
            feat_src1 = torch.cat([pre_feat_ext1, pre_feat_ext1, pre_feat_ext1], 0)  # feat_src: torch.Size([6, 512, 20, 20])
            # predict on discriminator
            optimizer_critic1.zero_grad()
            real_loss1 = criterionAdv(Discriminator1(feat_src1), True)
            fake_loss1 = criterionAdv(Discriminator1(feat_ext3.detach()), False)
            loss_critic1 = 0.5 * (real_loss1 + fake_loss1) * 10
            loss_critic1.backward()
            optimizer_critic1.step()
            # ************************* confusion domain 2 with 1,3 **********************************#
            feat_src2 = torch.cat([pre_feat_ext2, pre_feat_ext2, pre_feat_ext2], 0)
            optimizer_critic2.zero_grad()
            real_loss2 = criterionAdv(Discriminator2(feat_src2), True)
            fake_loss2 = criterionAdv(Discriminator2(feat_ext3.detach()), False)
            loss_critic2 = 0.5 * (real_loss2 + fake_loss2) * 10
            loss_critic2.backward()
            optimizer_critic2.step()
            # ************************* confusion domain 3 with 1,2 **********************************
            feat_src3 = torch.cat([pre_feat_ext3, pre_feat_ext3, pre_feat_ext3], 0)
            optimizer_critic3.zero_grad()
            real_loss3 = criterionAdv(Discriminator3(feat_src3), True)
            fake_loss3 = criterionAdv(Discriminator3(feat_ext3.detach()), False)
            loss_critic3 = 0.5 * (real_loss3 + fake_loss3) * 10
            loss_critic3.backward()
            optimizer_critic3.step()
            #----------------------#
            #   反向传播
            #----------------------#
            # #----------------------#
            # #   前向传播
            # ----------------------#
            optimizer.zero_grad()
            all_bboxes = []
            all_bboxes.extend(bboxes0)
            all_bboxes.extend(bboxes1)
            all_bboxes.extend(bboxes2)
            bboxes = torch.stack(sorted(all_bboxes, key=sort_key), dim=0)
            outputs = model_train(feat_ext1.detach(), feat_ext2.detach(), feat_ext3.detach())
            loss_value = yolo_loss(outputs, bboxes)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
            optimizer.step()
        if ema:
            # ema.update(backbone)
            ema.update(model_train)
        loss += loss_value.item()

        # ============ tensorboard the log info ============#
        info = {
            # 'Loss_triplet': Loss_triplet.item(),
            'Loss_yolo': loss_value.item(),
            'Loss_G': Loss_G.item(),
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
                                })
            pbar.update(1)

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
            optimizer_DG_conf.zero_grad()
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            feat        = backbone(images)
            outputs     = model_train_eval(feat[0],feat[1],feat[2])
            loss_value  = yolo_loss(outputs, bboxes)

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))

        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict_backbone = backbone.state_dict()
            save_state_dict_head = ema.ema.state_dict()
        else:
            save_state_dict_backbone = backbone.state_dict()
            save_state_dict_head = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict_backbone, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f_backbone.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            torch.save(save_state_dict_head, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f_head.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            # torch.save(save_state_dict_head, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            # print('Save best model to best_epoch_weights.pth')
            print('Save best model to best_epoch_weights_all.pth')
            torch.save(save_state_dict_backbone, os.path.join(save_dir, "best_epoch_weights_backbone.pth"))
            torch.save(save_state_dict_head, os.path.join(save_dir, "best_epoch_weights_head.pth"))
            # torch.save(save_state_dict_head, os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(save_state_dict_backbone, os.path.join(save_dir, "last_epoch_weights_backbone.pth"))
        torch.save(save_state_dict_head, os.path.join(save_dir, "last_epoch_weights_head.pth"))
        # torch.save(save_state_dict_head, os.path.join(save_dir, "last_epoch_weights.pth"))

# 定义一个排序关键字函数，按张量的第一个数字进行排序
def sort_key(tensor):
    return tensor[0]# 如果张量是一维的