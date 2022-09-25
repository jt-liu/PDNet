# turn eff-b3 to eff-b7
import torch
import torch.nn as nn
import torch.nn.functional as F
from .miniViT_for_vA2 import mViT


class ReSampleModuel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.project = nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, From):
        from_to = F.interpolate(From, scale_factor=2.0, mode='bilinear', align_corners=True)
        return self.project(from_to)


class SElayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SElayer, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(features // 2)
        self.conv2 = nn.Conv2d(features // 2, features // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(features // 2)
        self.conv3 = nn.Conv2d(features // 2, features, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(features)
        self.relu = nn.ReLU()
        self.se = SElayer(features)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        out = out + x
        out = self.relu(out)
        return out


class ConvAfterResidualUnit(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            SElayer(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class DecoderBN(nn.Module):
    def __init__(self, num_features=2048):
        super(DecoderBN, self).__init__()
        features = int(num_features)
        self.min_val = 0.1
        self.max_val = 10.0

        self.residual_3_1 = ResidualConvUnit(64)

        self.residual_3_2 = ResidualConvUnit(64)
        self.residual_5_1 = ResidualConvUnit(48)

        self.residual_3_3 = ResidualConvUnit(64)
        self.residual_5_2 = ResidualConvUnit(48)
        self.residual_6_1 = ResidualConvUnit(80)

        self.residual_3_4 = ResidualConvUnit(64)
        self.conv_residual_3_4 = ConvAfterResidualUnit(in_channel=64, out_channel=64)
        self.residual_5_3 = ResidualConvUnit(48)
        self.conv_residual_5_3 = ConvAfterResidualUnit(in_channel=48, out_channel=48)
        self.residual_6_2 = ResidualConvUnit(80)
        self.conv_residual_6_2 = ConvAfterResidualUnit(in_channel=80, out_channel=80)
        self.residual_8_1 = ResidualConvUnit(224)
        self.conv_residual_8_1 = ConvAfterResidualUnit(in_channel=224, out_channel=224)

        self.conv_fea_12 = nn.Sequential(
            SElayer(2560),
            nn.Conv2d(2560, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.act = nn.ReLU()

        self.conv_predict_3 = nn.Sequential(
            nn.Conv2d(64, 30, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.resample_5 = ReSampleModuel(48)
        self.conv_predict_5 = nn.Sequential(
            nn.Conv2d(48, 30, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.resample_6_1 = ReSampleModuel(80)
        self.conv_predict_6_1 = nn.Sequential(
            nn.Conv2d(80, 35, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(35),
            nn.ReLU()
        )
        self.resample_6_2 = ReSampleModuel(35)
        self.conv_predict_6_2 = nn.Sequential(
            nn.Conv2d(35, 30, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.resample_8_1 = ReSampleModuel(224)
        self.conv_predict_8_1 = nn.Sequential(
            nn.Conv2d(224, 68, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(68),
            nn.ReLU()
        )
        self.resample_8_2 = ReSampleModuel(68)
        self.conv_predict_8_2 = nn.Sequential(
            nn.Conv2d(68, 34, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(34),
            nn.ReLU()
        )
        self.resample_8_3 = ReSampleModuel(34)
        self.conv_predict_8_3 = nn.Sequential(
            nn.Conv2d(34, 30, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.resample_12_1 = ReSampleModuel(1024)
        self.conv_predict_12_1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.resample_12_2 = ReSampleModuel(512)
        self.conv_predict_12_2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.resample_12_3 = ReSampleModuel(256)
        self.conv_predict_12_3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.resample_12_4 = ReSampleModuel(128)
        self.conv_predict_12_4 = nn.Sequential(
            nn.Conv2d(128, 30, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.conv_xb1 = nn.Sequential(
            nn.Conv2d(30, 15, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(15, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv_xb2 = nn.Sequential(
            nn.Conv2d(30, 15, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(15, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv_xb3 = nn.Sequential(
            nn.Conv2d(30, 15, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(15, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv_xb4 = nn.Sequential(
            nn.Conv2d(30, 15, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(15, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.conv_all_pre = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.vit = mViT(4, patch_size=16, n_query_channels=128, embedding_dim=4, num_heads=4)
        self.conv_out = nn.Sequential(
            # nn.Conv2d(300, 150, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Identity(),
        )

    def forward(self, features):
        x_3_1 = self.residual_3_1(features[3])

        x_3_2 = self.residual_3_2(x_3_1)
        x_5_1 = self.residual_5_1(features[5])

        x_3_3 = self.residual_3_3(x_3_2)
        x_5_2 = self.residual_5_2(x_5_1)
        x_6_1 = self.residual_6_1(features[6])

        x_3_4 = self.residual_3_4(x_3_3)
        x_3_4 = self.conv_residual_3_4(x_3_4)
        x_3_4 = self.act(x_3_4 + features[3])
        x_5_3 = self.residual_5_3(x_5_2)
        x_5_3 = self.conv_residual_5_3(x_5_3)
        x_5_3 = self.act(x_5_3 + features[5])
        x_6_2 = self.residual_6_2(x_6_1)
        x_6_2 = self.conv_residual_6_2(x_6_2)
        x_6_2 = self.act(x_6_2 + features[6])
        x_8_1 = self.residual_8_1(features[8])
        x_8_1 = self.conv_residual_8_1(x_8_1)
        x_8_1 = self.act(x_8_1 + features[8])
        x_12 = self.act(self.conv_fea_12(features[12]))

        x_d1 = self.conv_predict_3(x_3_4)

        x_d2 = self.resample_5(x_5_3)
        x_d2 = F.interpolate(x_d2, [x_d1.size(2), x_d1.size(3)], mode='bilinear')
        x_d2 = self.conv_predict_5(x_d2)

        x_d3 = self.resample_6_1(x_6_2)
        x_d3 = self.conv_predict_6_1(x_d3)
        x_d3 = self.resample_6_2(x_d3)
        x_d3 = F.interpolate(x_d3, [x_d1.size(2), x_d1.size(3)], mode='bilinear')
        x_d3 = self.conv_predict_6_2(x_d3)

        x_d4 = self.resample_8_1(x_8_1)
        x_d4_1 = self.conv_predict_8_1(x_d4)
        x_d4 = self.resample_8_2(x_d4_1)
        x_d4_2 = self.conv_predict_8_2(x_d4)
        x_d4 = self.resample_8_3(x_d4_2)
        x_d4 = F.interpolate(x_d4, [x_d1.size(2), x_d1.size(3)], mode='bilinear')
        x_d4_3 = self.conv_predict_8_3(x_d4)

        x_d5 = self.resample_12_1(x_12)
        x_d5_1 = self.conv_predict_12_1(x_d5)
        x_d5 = self.resample_12_2(x_d5_1)
        x_d5_2 = self.conv_predict_12_2(x_d5)
        x_d5 = self.resample_12_3(x_d5_2)
        x_d5_3 = self.conv_predict_12_3(x_d5)
        x_d5 = self.resample_12_4(x_d5_3)
        x_d5 = F.interpolate(x_d5, [x_d1.size(2), x_d1.size(3)], mode='bilinear')
        x_d5_4 = self.conv_predict_12_4(x_d5)

        # print(x_d5.shape)
        # print(x_d1.shape, x_d2.shape, x_d3_2.shape, x_d4_3.shape)

        x_block1 = self.conv_xb1(x_d1 + x_d2 + x_d5_4)
        x_block2 = self.conv_xb2(x_d2 + x_d3 + x_d5_4)
        x_block3 = self.conv_xb3(x_d3 + x_d4_3 + x_d1)
        x_block4 = self.conv_xb4(x_d4_3 + x_d5_4 + x_d1)

        del x_d1, x_d2, x_d3, x_d4, x_d5, x_d4_1, x_d4_2, x_d4_3, x_d5_1, x_d5_2, x_d5_3, x_d5_4

        raw_pred = torch.cat([x_block1, x_block2, x_block3, x_block4], dim=1)
        raw_pred = self.conv_all_pre(raw_pred)
        pred = self.vit(raw_pred)
        pred = self.conv_out(pred)

        return pred, x_block1, x_block2, x_block3, x_block4


class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        # y = 0
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    # y = y+1
                    # print(y)
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


class Eff_estimation(nn.Module):
    def __init__(self, num_features=2048):
        super(Eff_estimation, self).__init__()

        basemodel_name = 'tf_efficientnet_b7_ap'

        print('Loading base model ()...'.format(basemodel_name), end='')
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        print('Done.')

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print('Building Encoder-Decoder model..', end='')
        self.encoder = Encoder(basemodel)
        self.decoder = DecoderBN(num_features=num_features)
        print('Done.')

    def forward(self, x):
        out, x_block1, x_block2, x_block3, x_block4 = self.decoder(self.encoder(x))
        out[out < 1e-8] = 1e-8
        x_block1[x_block1 < 1e-8] = 1e-8
        x_block2[x_block2 < 1e-8] = 1e-8
        x_block3[x_block3 < 1e-8] = 1e-8
        x_block4[x_block4 < 1e-8] = 1e-8
        return out, x_block1, x_block2, x_block3, x_block4


if __name__ == '__main__':
    model = Eff_estimation()  # .cuda()
    x = torch.rand(2, 3, 561, 427)  # .cuda()
    '''torch.onnx.export(model, x, "model.onnx", opset_version=11)
    import onnx
    from onnx import shape_inference
    model_onnx = "model.onnx"
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(model_onnx)), model_onnx)'''
    # print(model)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))  # 82.23M add_se_attention:82.40M
    import time
    start1 = time.time()
    out, x_block1, x_block2, x_block3, x_block4 = model(x)
    end1 = time.time()
    print(end1-start1)
    print(out.shape)
