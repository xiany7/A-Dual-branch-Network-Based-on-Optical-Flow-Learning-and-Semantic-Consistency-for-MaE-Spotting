import torch
import torch.nn as nn
from torch.nn import init
# from resample2d_package.resample2d import Resample2d
# from channelnorm_package.channelnorm import ChannelNorm


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )

def i_conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, bias = True):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=bias),
            nn.BatchNorm2d(out_planes),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=bias),
        )

def predict_flow(in_planes):
    # return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)


class SparseFlow(nn.Module):
    def __init__(self, batchNorm=True):
        super(SparseFlow,self).__init__()

        self.batchNorm = batchNorm
        self.conv0   = conv(self.batchNorm,  2,   32)
        self.pool0   = nn.MaxPool2d(2, stride=2)
        self.conv1 = conv(self.batchNorm,  32,   64)
        self.conv1_1   = conv(self.batchNorm,  64,  64)
        self.pool1   = nn.MaxPool2d(2, stride=2)
        self.conv2 = conv(self.batchNorm,  64,  128)
        self.conv2_1 = conv(self.batchNorm,  128,  128)
        self.pool2   = nn.MaxPool2d(2, stride=2)
        self.conv3 = conv(self.batchNorm,  128, 256, 3, 2)
        self.conv3_1 = conv(self.batchNorm,  256,  256)
        self.pool3   = nn.MaxPool2d(2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)

    def forward(self, x):
        
        # print('x:',x.shape)  
        out_conv0 = self.conv0(x)
        out_conv0 = self.pool0(out_conv0)
        # print('out_conv0:',out_conv0.shape)  
        out_conv1 = self.conv1_1(self.conv1(out_conv0)) 
        out_conv1 = self.pool1(out_conv1)
        # print('out_conv1:',out_conv1.shape)   
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv2 = self.pool2(out_conv2)
        # print('out_conv2:',out_conv2.shape) 
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv3 = self.pool3(out_conv3)
       
        return out_conv3

class FlowNetSD(nn.Module):
    def __init__(self,  batchNorm=True):
        super(FlowNetSD,self).__init__()

        self.batchNorm = batchNorm
        self.conv0   = conv(self.batchNorm,  6,   64)
        self.conv1   = conv(self.batchNorm,  64,   64, stride=2)
        self.conv1_1 = conv(self.batchNorm,  64,   128)
        self.conv2   = conv(self.batchNorm,  128,  128, stride=2)
        self.conv2_1 = conv(self.batchNorm,  128,  128)
        self.conv3   = conv(self.batchNorm, 128,  256, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)
        # self.deconv1 = deconv(194,32)
        # self.deconv0 = deconv(162,16)

        self.inter_conv5 = i_conv(self.batchNorm,  1026,   512)
        self.inter_conv4 = i_conv(self.batchNorm,  770,   256)
        self.inter_conv3 = i_conv(self.batchNorm,  386,   128)
        self.inter_conv2 = i_conv(self.batchNorm,  194,   64)
        # self.inter_conv1 = i_conv(self.batchNorm,  162,   32)
        # self.inter_conv0 = i_conv(self.batchNorm,  82,   16)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(512)
        self.predict_flow4 = predict_flow(256)
        self.predict_flow3 = predict_flow(128)
        self.predict_flow2 = predict_flow(64)
        # self.predict_flow1 = predict_flow(32)
        # self.predict_flow0 = predict_flow(16)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        # self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        # self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)
        '''
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.channelnorm = ChannelNorm()
        self.resample1 = Resample2d()
        self.flownetfusion = FlowNetFusion(batchNorm=self.batchNorm)
        '''
        

    

    
    def extract_features(self, x):
        # feat_dict = []
        out_conv0 = self.conv0(x)
        
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        # feat_dict.append(out_conv4)
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        # feat_dict.append(out_conv5)  # torch.Size([1, 512, 4, 4])
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        # feat_dict.append(out_conv6)  # torch.Size([1, 1024, 2, 2])
        if self.training:
    
            flow6       = self.predict_flow6(out_conv6)
            # print(flow6.shape)
            flow6_up    = self.upsampled_flow6_to_5(flow6)
            # print(flow6_up.shape)
            out_deconv5 = self.deconv5(out_conv6)
            # print(out_deconv5.shape)
            concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
            
            out_interconv5 = self.inter_conv5(concat5)
            flow5       = self.predict_flow5(out_interconv5)

            flow5_up    = self.upsampled_flow5_to_4(flow5)
            out_deconv4 = self.deconv4(concat5)
   
            concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
            out_interconv4 = self.inter_conv4(concat4)
            flow4       = self.predict_flow4(out_interconv4)
            flow4_up    = self.upsampled_flow4_to_3(flow4)
            
            out_deconv3 = self.deconv3(concat4)
            
            concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
            out_interconv3 = self.inter_conv3(concat3)
            flow3       = self.predict_flow3(out_interconv3)
            flow3_up    = self.upsampled_flow3_to_2(flow3)
            out_deconv2 = self.deconv2(concat3)

            concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
            out_interconv2 = self.inter_conv2(concat2)
            flow2 = self.predict_flow2(out_interconv2)
            # print(flow2.shape)
            '''
            flow2_up    = self.upsampled_flow2_to_1(flow2)
            # print(flow2_up.shape)
            # print(concat2.shape)
            out_deconv1 = self.deconv1(concat2)
            # print(concat2.shape)
            
            concat1 = torch.cat((out_conv1,out_deconv1,flow2_up),1)
            # print(concat1.shape)
            out_interconv1 = self.inter_conv1(concat1)
            # print(out_interconv1.shape)
            flow1 = self.predict_flow1(out_interconv1)
            flow1_up    = self.upsampled_flow1_to_0(flow1)
            # print(flow2_up.shape)
            # print(concat2.shape)
            out_deconv0 = self.deconv0(concat1)
            # print(concat2.shape)
            
            concat0 = torch.cat((out_conv0,out_deconv0,flow1_up),1)
            # print(concat0.shape)
            out_interconv0 = self.inter_conv0(concat0)
            # print(out_interconv0.shape)
            flow0 = self.predict_flow0(out_interconv0)
            '''
            
            
            
            # return [flow2,flow3,flow4,flow5,flow6], out_conv6
            # return [flow4,flow5,flow6], out_conv6
            return flow2, out_conv6
        else:
            return out_conv6

class FlowNetSD_BackBone(nn.Module):
    def __init__(self, batchNorm=False, freeze_bn=True, freeze_bn_affine=True):
        super(FlowNetSD_BackBone, self).__init__()
        self.batchNorm = batchNorm
        self._model = FlowNetSD(batchNorm=self.batchNorm)
        self._freeze_bn = freeze_bn
        self._freeze_bn_affine = freeze_bn_affine
        self.batchNorm = batchNorm

   
    
    def load_pretrained_weight(self, model_path='/home/developers/xianyun/spotting/core/models/pretrained/FlowNet2-SD_encoder_checkpoint.pth.tar'):
        # pretrained_dict = torch.load(model_path, map_location='cuda:{}'.format(self.local_rank))
        
        pretrained_dict = torch.load(model_path)
        model_dict = self._model.state_dict()
        # 筛除不加载的层结构
        pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if k in model_dict}
        '''
        for k, v in pretrained_dict.items():
            print(k)
        '''
        
        # 更新当前网络的结构字典
        model_dict.update(pretrained_dict)
        self._model.load_state_dict(model_dict)
        
    def forward(self, x):     
        return self._model.extract_features(x)
    
if __name__ == '__main__':
    model = FlowNetSD_BackBone()
    model.load_pretrained_weight()
    model.train()
    x = torch.rand([2,6,128,128])
    flow, feat = model(x)
    print(feat.shape)
    print(flow.shape)
        
