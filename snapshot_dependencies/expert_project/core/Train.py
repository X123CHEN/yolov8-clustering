from core.utils import get_inf_iterator, mkdir
from core import evaluate
from torch.nn import DataParallel
from core import loss
import itertools
import os
from collections import OrderedDict
import torchvision.utils as vutils
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F

from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pdb import set_trace as st


def Train123(args, FeatExtor, FeatEmbdmodel, Discriminator1, Discriminator2, Discriminator3,
          PreFeatExtorS1, PreFeatExtorS2, PreFeatExtorS3,
          data_loader1_real, data_loader1_fake,
          data_loader2_real, data_loader2_fake,
          data_loader3_real, data_loader3_fake,
          data_loader_target,
          summary_writer, Saver, savefilename):

    ####################
    # 1. setup network #
    ####################
    # set train state for Dropout and BN layers
    FeatExtor.train()
    FeatEmbdmodel.train()

    Discriminator1.train()
    Discriminator2.train()
    Discriminator3.train()

    PreFeatExtorS1.eval()
    PreFeatExtorS2.eval()
    PreFeatExtorS3.eval()

    FeatExtor = DataParallel(FeatExtor)  # DataParallel：数据并行处理

    FeatEmbder = DataParallel(FeatEmbdmodel)#wrq该FeatEmbder  为FeatEmbdmodel

    Discriminator1 = DataParallel(Discriminator1)
    Discriminator2 = DataParallel(Discriminator2)
    Discriminator3 = DataParallel(Discriminator3)

    PreFeatExtorS1 = DataParallel(PreFeatExtorS1)
    PreFeatExtorS2 = DataParallel(PreFeatExtorS2)
    PreFeatExtorS3 = DataParallel(PreFeatExtorS3)

    ####################
    # setup criterion and optimizer
    criterionAdv = loss.GANLoss()  # 对抗性损失（Adversarial Loss）
    # criterionCls = torch.nn.BCEWithLogitsLoss()  # 分类任务的损失函数，常见的分类损失函数包括交叉熵损失（Cross-Entropy Loss）。

    optimizer_DG_conf = optim.Adam(itertools.chain(FeatExtor.parameters(), FeatEmbder.parameters()),
                                   lr=args.lr_DG_conf,
                                   betas=(args.beta1, args.beta2))

    optimizer_critic1 = optim.Adam(Discriminator1.parameters(),
                                   lr=args.lr_critic,
                                   betas=(args.beta1, args.beta2))

    optimizer_critic2 = optim.Adam(Discriminator2.parameters(),
                                   lr=args.lr_critic,
                                   betas=(args.beta1, args.beta2))

    optimizer_critic3 = optim.Adam(Discriminator3.parameters(),
                                   lr=args.lr_critic,
                                   betas=(args.beta1, args.beta2))

    iternum = max(len(data_loader1_real), len(data_loader1_fake),
                  len(data_loader2_real), len(data_loader2_fake),
                  len(data_loader3_real), len(data_loader3_fake))

    print('iternum={}'.format(iternum))

    ####################
    # 2. train network #
    ####################
    global_step = 0

    for epoch in range(args.epochs):

        data1_real = get_inf_iterator(data_loader1_real)
        data1_fake = get_inf_iterator(data_loader1_fake)

        data2_real = get_inf_iterator(data_loader2_real)
        data2_fake = get_inf_iterator(data_loader2_fake)

        data3_real = get_inf_iterator(data_loader3_real)
        data3_fake = get_inf_iterator(data_loader3_fake)

        for step in range(iternum):

            FeatExtor.train()
            FeatEmbder.train()


            Discriminator1.train()
            Discriminator2.train()
            Discriminator3.train()

            # ============ one batch extraction ============#
            # ----------------------------------------------------------wrq修改
            cat_img1_real, lab1_real, real1_x_y_w_h = next(data1_real)
            cat_img1_fake, lab1_fake, fake1_x_y_w_h = next(data1_fake)

            cat_img2_real, lab2_real, real2_x_y_w_h = next(data2_real)
            cat_img2_fake, lab2_fake, fake2_x_y_w_h = next(data2_fake)

            cat_img3_real, lab3_real, real3_x_y_w_h = next(data3_real)
            cat_img3_fake, lab3_fake, fake3_x_y_w_h = next(data3_fake)

            # ============ one batch collection ============#
            ori_img1 = torch.cat([cat_img1_real, cat_img1_fake], 0).cuda()
            # depth_img1 = torch.cat([depth_img1_real,depth_img1_fake],0)
            lab1 = torch.cat([lab1_real, lab1_fake], 0)

            ori_img2 = torch.cat([cat_img2_real, cat_img2_fake], 0).cuda()
            # depth_img2 = torch.cat([depth_img2_real,depth_img2_fake],0)
            lab2 = torch.cat([lab2_real, lab2_fake], 0)

            ori_img3 = torch.cat([cat_img3_real, cat_img3_fake], 0).cuda()
            # depth_img3 = torch.cat([depth_img3_real,depth_img3_fake],0)
            lab3 = torch.cat([lab3_real, lab3_fake], 0)

            # wrq修，修改尺寸， 让 img1、2、3的尺寸相同可cat，，使用resize
            # 假设 ori_img1 的大小是 (height, width)
            height, width = ori_img1.size(2), ori_img1.size(3)
            # 使用 interpolate 函数将 ori_img2 调整为相同的大小
            ori_img2 = F.interpolate(ori_img2, size=(height, width), mode='bilinear', align_corners=False)
            # 使用 interpolate 函数将 ori_img3 调整为相同的大小
            ori_img3 = F.interpolate(ori_img3, size=(height, width), mode='bilinear', align_corners=False)

            ori_img = torch.cat([ori_img1, ori_img2, ori_img3], 0)
            ori_img = ori_img.cuda()

            label = torch.cat([lab1, lab2, lab3], 0)
            # label = label.long().squeeze().cuda()

#wrq 分界线###############################################################################

            with torch.no_grad():
                pre_feat_ext1 = PreFeatExtorS1(ori_img1)[1]
                pre_feat_ext2 = PreFeatExtorS2(ori_img2)[1]
                pre_feat_ext3 = PreFeatExtorS3(ori_img3)[1]

            # ============ Depth supervision ============#

            ######### 1. depth loss #########
            # optimizer_DG_depth.zero_grad()
            #
            # feat_ext_all, feat_ext = FeatExtor(ori_img)
            # depth_Pre = DepthEsmator(feat_ext_all)
            #
            # Loss_depth = args.W_depth*criterionDepth(depth_Pre, depth_GT)
            #
            # Loss_depth.backward()
            # optimizer_DG_depth.step()

            # ============ domain generalization supervision ============#

            optimizer_DG_conf.zero_grad()

            #下列特征分别大小分别是：1024 * 0.5=512 , 20, 20      512, 40, 40。     256, 80, 80。
            feat_ext, feat_ext2, feat_ext3,  = FeatExtor(ori_img)
            # _, feat_ext = FeatExtor(ori_img)
            #wrq修改代码， 周涵 建议, _, d1, d2, d3 = FeatExtor(ori_img)

            feat_tgt = feat_ext

            # ************************* confusion all **********************************#

            # predict on generator                       # yolov8 骨干提出3个不同尺度的特征，然后对三个尺度的特征都进行生成损失的计算
            loss_generator1 = criterionAdv(Discriminator1(feat_tgt), True)
            loss_generator2 = criterionAdv(Discriminator2(feat_tgt), True)
            loss_generator3 = criterionAdv(Discriminator3(feat_tgt), True)

            loss_generator4 = criterionAdv(Discriminator1(feat_ext2), True)
            loss_generator5 = criterionAdv(Discriminator2(feat_ext2), True)
            loss_generator6 = criterionAdv(Discriminator3(feat_ext2), True)

            loss_generator7 = criterionAdv(Discriminator1(feat_ext3), True)
            loss_generator8 = criterionAdv(Discriminator2(feat_ext3), True)
            loss_generator9 = criterionAdv(Discriminator3(feat_ext3), True)


            # feat_embd, label_pred = FeatEmbder(feat_ext)
            feat_embd = FeatEmbder(feat_ext)  ##wrq 需要修改


            ########## cross-domain triplet loss #########
            Loss_triplet = TripletLossCal(args, feat_embd, lab1, lab2, lab3)

            # Loss_cls = criterionCls(label_pred.squeeze(), label.float())

            Loss_gen_feat_tgt = args.W_genave * (loss_generator1 + loss_generator2 + loss_generator3)
            Loss_gen_feat_ext2 = args.W_genave * (loss_generator4 + loss_generator5 + loss_generator6)
            Loss_gen_feat_ext3 = args.W_genave * (loss_generator7 + loss_generator8 + loss_generator9)


            # Loss_G = args.W_trip * Loss_triplet + args.W_cls * Loss_cls + args.W_gen * Loss_gen
            Loss_G = args.W_trip * Loss_triplet + args.W_gen * (Loss_gen_feat_tgt + Loss_gen_feat_ext2 + Loss_gen_feat_ext3)

            Loss_G.backward()
            optimizer_DG_conf.step()

            # ************************* confusion domain 1 with 2,3 **********************************#

            feat_src = torch.cat([pre_feat_ext1, pre_feat_ext1, pre_feat_ext1], 0)

            # predict on discriminator
            optimizer_critic1.zero_grad()

            real_loss = criterionAdv(Discriminator1(feat_src), True)
            fake_loss = criterionAdv(Discriminator1(feat_tgt.detach()), False)

            loss_critic1 = 0.5 * (real_loss + fake_loss)

            loss_critic1.backward()
            optimizer_critic1.step()

            # ************************* confusion domain 2 with 1,3 **********************************#

            feat_src = torch.cat([pre_feat_ext2, pre_feat_ext2, pre_feat_ext2], 0)

            # predict on discriminator
            optimizer_critic2.zero_grad()

            real_loss = criterionAdv(Discriminator2(feat_src), True)
            fake_loss = criterionAdv(Discriminator2(feat_tgt.detach()), False)

            loss_critic2 = 0.5 * (real_loss + fake_loss)

            loss_critic2.backward()
            optimizer_critic2.step()

            # ************************* confusion domain 3 with 1,2 **********************************#

            feat_src = torch.cat([pre_feat_ext3, pre_feat_ext3, pre_feat_ext3], 0)

            # predict on discriminator
            optimizer_critic3.zero_grad()

            real_loss = criterionAdv(Discriminator3(feat_src), True)
            fake_loss = criterionAdv(Discriminator3(feat_tgt.detach()), False)

            loss_critic3 = 0.5 * (real_loss + fake_loss)

            loss_critic3.backward()
            optimizer_critic3.step()

            # ============ tensorboard the log info ============#
            info = {
                # 'Loss_depth': Loss_depth.item(),
                'Loss_triplet': Loss_triplet.item(),
                # 'Loss_cls': Loss_cls.item(),
                'Loss_G': Loss_G.item(),
                'loss_critic1': loss_critic1.item(),
                'loss_generator1': loss_generator1.item(),
                'loss_critic2': loss_critic2.item(),
                'loss_generator2': loss_generator2.item(),
                'loss_critic3': loss_critic3.item(),
                'loss_generator3': loss_generator3.item(),
            }
            for tag, value in info.items():
                summary_writer.add_scalar(tag, value, global_step)

                # if (step+1) % args.tst_step == 0:
                # depth_Pre_real = torch.cat([depth_Pre[0:args.batchsize],depth_Pre[2*args.batchsize:3*args.batchsize], depth_Pre[4*args.batchsize:5*args.batchsize]],0)
                # depth_Pre_fake = torch.cat([depth_Pre[args.batchsize:2*args.batchsize],depth_Pre[3*args.batchsize:4*args.batchsize], depth_Pre[5*args.batchsize:6*args.batchsize]],0)
                #
                # depth_Pre_all = vutils.make_grid(depth_Pre, normalize=True, scale_each=True)
                # depth_Pre_real = vutils.make_grid(depth_Pre_real, normalize=True, scale_each=True)
                # depth_Pre_fake = vutils.make_grid(depth_Pre_fake, normalize=True, scale_each=True)

                # summary_writer.add_image('Depth_Image_all', depth_Pre_all, global_step)
                # summary_writer.add_image('Depth_Image_real', depth_Pre_real, global_step)
                # summary_writer.add_image('Depth_Image_fake', depth_Pre_fake, global_step)

            # ============ print the log info ============#
            if (step + 1) % args.log_step == 0:
                errors = OrderedDict([
                    # ('Loss_depth', Loss_depth.item()),
                    ('Loss_triplet', Loss_triplet.item()),
                    # ('Loss_cls', Loss_cls.item()),
                    ('Loss_G', Loss_G.item()),

                    ('loss_critic1', loss_critic1.item()),
                    ('loss_generator1', loss_generator1.item()),

                    ('loss_critic2', loss_critic2.item()),
                    ('loss_generator2', loss_generator2.item()),

                    ('loss_critic3', loss_critic3.item()),
                    ('loss_generator3', loss_generator3.item())])

                Saver.print_current_errors((epoch + 1), (step + 1), errors)

            if (step + 1) % args.tst_step == 0:
                evaluate.evaluate_img(args, FeatExtor, data_loader_target, (epoch + 1), (step + 1), Saver)

            global_step += 1

            #############################
            # 2.4 save model parameters #
            #############################
            if ((step + 1) % args.model_save_step == 0):
                model_save_path = os.path.join(args.results_path, 'snapshots', savefilename)
                mkdir(model_save_path)

                torch.save(FeatExtor.state_dict(), os.path.join(model_save_path,
                                                                "DGFA-Ext-{}-{}.pt".format(epoch + 1, step + 1)))

                torch.save(FeatEmbder.state_dict(), os.path.join(model_save_path,
                                                                 "DGFA-Embd-{}-{}.pt".format(epoch + 1, step + 1)))

                # torch.save(DepthEsmator.state_dict(), os.path.join(model_save_path,
                #     "DGFA-Depth-{}-{}.pt".format(epoch+1, step+1)))

                torch.save(Discriminator1.state_dict(), os.path.join(model_save_path,
                                                                     "DGFA-D1-{}-{}.pt".format(epoch + 1, step + 1)))

                torch.save(Discriminator2.state_dict(), os.path.join(model_save_path,
                                                                     "DGFA-D2-{}-{}.pt".format(epoch + 1, step + 1)))

                torch.save(Discriminator3.state_dict(), os.path.join(model_save_path,
                                                                     "DGFA-D3-{}-{}.pt".format(epoch + 1, step + 1)))

        if ((epoch + 1) % args.model_save_epoch == 0):
            model_save_path = os.path.join(args.results_path, 'snapshots', savefilename)
            mkdir(model_save_path)

            torch.save(FeatExtor.state_dict(), os.path.join(model_save_path,
                                                            "DGFA-Ext-{}.pt".format(epoch + 1)))

            torch.save(FeatEmbder.state_dict(), os.path.join(model_save_path,
                                                             "DGFA-Embd-{}.pt".format(epoch + 1)))

            torch.save(Discriminator1.state_dict(), os.path.join(model_save_path,
                                                                 "DGFA-D1-{}.pt".format(epoch + 1)))

            torch.save(Discriminator2.state_dict(), os.path.join(model_save_path,
                                                                 "DGFA-D2-{}.pt".format(epoch + 1)))

            torch.save(Discriminator3.state_dict(), os.path.join(model_save_path,
                                                                 "DGFA-D3-{}.pt".format(epoch + 1)))

    torch.save(FeatExtor.state_dict(), os.path.join(model_save_path,
                                                    "DGFA-Ext-final.pt"))

    torch.save(FeatEmbder.state_dict(), os.path.join(model_save_path,
                                                     "DGFA-Embd-final.pt"))

    # torch.save(DepthEsmator.state_dict(), os.path.join(model_save_path,
    #     "DGFA-Depth-final.pt"))

    torch.save(Discriminator1.state_dict(), os.path.join(model_save_path,
                                                         "DGFA-D1-final.pt"))

    torch.save(Discriminator2.state_dict(), os.path.join(model_save_path,
                                                         "DGFA-D2-final.pt"))

    torch.save(Discriminator3.state_dict(), os.path.join(model_save_path,
                                                         "DGFA-D3-final.pt"))
    ###return Loss_G    wrq修改代码，增加了Loss_G的返回值，然后加到yolo_loss里面让模型一起评估总损失




