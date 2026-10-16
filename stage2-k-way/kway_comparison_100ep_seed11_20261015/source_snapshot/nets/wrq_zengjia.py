
backbone = Backbone(base_channels, base_depth, deep_mul, phi, pretrained)
_, _, feat_ext = backbone(input_tensor)
feat_embd = FeatEmbder(feat_ext)
Loss_triplet = TripletLossCal(args, feat_embd, lab1, lab2, lab3)

Loss_triplet.backward()
optimizer_DG_conf.step()  ###加载器要叠加

###输出打印 Loss_triplet  数值也要更新代码


###

class FeatEmbedder(nn.Module):
    # wrq修改in_channels=128 为 24
    # def __init__(self, embed_size, in_channels=128, nettype=None):
    def __init__(self, embed_size, in_channels=24, nettype=None):
        super(FeatEmbedder, self).__init__()

        self.nettype = nettype

        self.conv = nn.Sequential(
            conv3x3(in_channels, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            conv3x3(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            conv3x3(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        x = self.conv(x)
        x = self.avgpooling(x)
        x = x.view(x.size(0), -1)
        feat = x

        return F.normalize(feat, p=2, dim=1)


def TripletLossCal(args, feat_ext, lab1, lab2, lab3):
    criterionTri_inter = TripletLoss(margin=0.5)
    criterionTri_intra = TripletLoss(margin=0.1)

    # avgpool = nn.AvgPool2d(kernel_size=32, stride=1)
    # feat_ext_pl = avgpool(feat_ext).squeeze()

    feat_embd1 = feat_ext[0:args.batchsize * 2]
    feat_embd2 = feat_ext[args.batchsize * 2:args.batchsize * 4]
    feat_embd3 = feat_ext[args.batchsize * 4:args.batchsize * 6]

    #跨域损失，  1 2 3  三个源域
    ########## 1.1 cross-domain triplet loss #########
    loss_Tri_12 = criterionTri_inter(torch.cat([feat_embd1, feat_embd2], 0), torch.cat([lab1, lab2], 0))[0]
    loss_Tri_23 = criterionTri_inter(torch.cat([feat_embd2, feat_embd3], 0), torch.cat([lab2, lab3], 0))[0]
    loss_Tri_13 = criterionTri_inter(torch.cat([feat_embd1, feat_embd3], 0), torch.cat([lab1, lab3], 0))[0]
    loss_tri_inter = loss_Tri_12 + loss_Tri_23 + loss_Tri_13

    #域内损失 1 2 3 自己跟自己的损失
    ########### 1.2 intra-domain triplet loss #########
    loss_Tri_1 = criterionTri_intra(feat_embd1, lab1)[0]
    loss_Tri_2 = criterionTri_intra(feat_embd2, lab2)[0]
    loss_Tri_3 = criterionTri_intra(feat_embd3, lab3)[0]
    loss_tri_intra = loss_Tri_1 + loss_Tri_2 + loss_Tri_3

    #parser.add_argument('--W_intra', type=int, default=0.1)
    Loss_triplet = loss_tri_inter + args.W_intra * loss_tri_intra  #args.W_intra  这是什么？


    return Loss_triplet
