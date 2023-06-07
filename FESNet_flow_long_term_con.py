import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
sys.path.append('/home/developers/xianyun/spotting/core/models/model_zoo')
from i3d import I3D_BackBone
from resnet3d import Resnet3D18
from FlowNetSD import FlowNetSD_BackBone



class Unit1D(nn.Module):
    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=1,
                 stride=1,
                 padding='same',
                 activation_fn=F.relu,
                 use_bias=True,):
        super(Unit1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels,
                                output_channels,
                                kernel_shape,
                                stride,
                                padding=0,
                                bias=use_bias)
        self._activation_fn = activation_fn
        self._padding = padding
        self._stride = stride
        self._kernel_shape = kernel_shape

    def compute_pad(self, t):
        if t % self._stride == 0:
            return max(self._kernel_shape - self._stride, 0)
        else:
            return max(self._kernel_shape - (t % self._stride), 0)

    def forward(self, x):
        if self._padding == 'same':
            batch, channel, t = x.size()
            pad_t = self.compute_pad(t)
            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f
            x = F.pad(x, [pad_t_f, pad_t_b])
        x = self.conv1d(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x
        


   

   

class Unit3D(nn.Module):
    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding='spatial_valid',
                 activation_fn=F.relu,
                 use_batch_norm=False,
                 use_bias=False):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.padding = padding

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                bias=self._use_bias)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        if self.padding == 'same':
            (batch, channel, t, h, w) = x.size()
            pad_t = self.compute_pad(0, t)
            pad_h = self.compute_pad(1, h)
            pad_w = self.compute_pad(2, w)

            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f
            pad_h_f = pad_h // 2
            pad_h_b = pad_h - pad_h_f
            pad_w_f = pad_w // 2
            pad_w_b = pad_w - pad_w_f

            pad = [pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b]
            x = F.pad(x, pad)

        if self.padding == 'spatial_valid':
            (batch, channel, t, h, w) = x.size()
            pad_t = self.compute_pad(0, t)
            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f

            pad = [0, 0, 0, 0, pad_t_f, pad_t_b]
            x = F.pad(x, pad)

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x



class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return torch.exp(input * self.scale)

class CoarsePyramid(nn.Module):
    def __init__(self, feat_channels, kernel_shapes, out_channels, num_classes, feat_t, layer_num):
        super(CoarsePyramid, self).__init__()

        self.pyramids = nn.ModuleList()
        self.loc_heads = nn.ModuleList()
        self.num_classes = num_classes
        self.layer_num = layer_num
        self.feat_t = feat_t

        self.pyramids.append(nn.Sequential(
            Unit3D(
                in_channels=feat_channels[0],
                output_channels=out_channels,
                kernel_shape=[1, kernel_shapes[0], kernel_shapes[0]],
                padding='spatial_valid',
                use_batch_norm=False,
                use_bias=True,
                activation_fn=None
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        ))
        
        for i in range(self.layer_num):
            self.pyramids.append(nn.Sequential(
                Unit1D(
                    in_channels=out_channels,
                    output_channels=out_channels,
                    kernel_shape=3,
                    stride=2,
                    use_bias=True,
                    activation_fn=None
                ),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True)
            ))
        
        loc_towers = []
        for i in range(2):
            loc_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.loc_tower = nn.Sequential(*loc_towers)
        conf_towers = []
        # sample_conf_towers = []
        for i in range(2):
            conf_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True)
                )
            )
      

            
            
        self.conf_tower = nn.Sequential(*conf_towers)

        self.loc_head = Unit1D(
            in_channels=out_channels,
            output_channels=2,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )
        self.conf_head = Unit1D(
            in_channels=out_channels,
            output_channels=self.num_classes,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )

        self.priors = []
        t = self.feat_t
        for i in range(self.layer_num):
            self.loc_heads.append(ScaleExp())
            self.priors.append(
                torch.Tensor([[(c + 0.5) / t] for c in range(t)]).view(-1, 1)
            )
            t = t // 2

    def forward(self, feat_dict):
        pyramid_feats = []
        locs = []
        confs = []

        # x1 = self.action(x1)
        batch_num, _, _, _, _ = feat_dict.shape
       
        for i, conv in enumerate(self.pyramids):
            feat_dict = conv(feat_dict)
            if i == 0:
                feat_dict = feat_dict.squeeze(-1).squeeze(-1)  # flatten
                # feat_dict = self.MultiscaleTemproalFusion(feat_dict)
                sample_feat = feat_dict
            elif i >= 1:
                pyramid_feats.append(feat_dict)
        
        # pyramid_feats = self.AdaptiveTemproalFusion(pyramid_feats)
        sample_pyramid_feats = []
        t = self.feat_t
        for i in range(self.layer_num):
            # sample_pyramid_feats.append(torch.nn.functional.interpolate(input=pyramid_feats[0], size=(t), mode='linear', align_corners=False))
            sample_pyramid_feats.append(torch.nn.functional.interpolate(input=sample_feat, size=(t), mode='linear', align_corners=False))
            t = t // 2
      
        for i, feat in enumerate(pyramid_feats):

            loc_feat = self.loc_tower(feat) 
            conf_feat = self.conf_tower(feat)

            locs.append(
                self.loc_heads[i](self.loc_head(loc_feat))
                    .view(batch_num, 2, -1)
                    .permute(0, 2, 1).contiguous()   # torch.Size([1, 64, 2])
            )
           
            confs.append(
                self.conf_head(conf_feat).view(batch_num, self.num_classes, -1)
                    .permute(0, 2, 1).contiguous()  # torch.Size([1, 64, 21])
            )
            
            
            
        loc = torch.cat([o.view(batch_num, -1, 2) for o in locs], 1)
        conf = torch.cat([o.view(batch_num, -1, self.num_classes) for o in confs], 1)
        priors = torch.cat(self.priors, 0).to(loc.device).unsqueeze(0)
   
        
        # CLS CON
        if self.training:
   
            sample_confs = []
            for i, feat in enumerate(sample_pyramid_feats):
                conf_feat = self.conf_tower(feat)
                sample_confs.append(
                    self.conf_head(conf_feat).view(batch_num, self.num_classes, -1)
                        .permute(0, 2, 1).contiguous()  # torch.Size([1, 64, 21])
                )
            sample_conf = torch.cat([o.view(batch_num, -1, self.num_classes) for o in sample_confs], 1)
            return loc, conf, priors, sample_conf
        else:
            return loc, conf, priors



class FESNet(nn.Module):
    def __init__(self, backbone_model, num_classes, feat_t, layer_num, in_channels=3,  training=True):
        super(FESNet, self).__init__()
        if backbone_model == 'flownetsd':
            self.backbone = FlowNetSD_BackBone()
            self.coarse_pyramid_detection = CoarsePyramid([1024, 512, 512], [2, 4, 8], 512, num_classes, feat_t, layer_num)
        elif backbone_model == 'i3d':
            self.backbone = I3D_BackBone(in_channels=in_channels)
            self.coarse_pyramid_detection = CoarsePyramid([1024, 832, 480], [3, 6, 12], 512, num_classes, feat_t, layer_num)
        elif backbone_model == '3dresnet18':
            self.backbone = Resnet3D18(in_channels=in_channels)
            self.coarse_pyramid_detection = CoarsePyramid([512, 256, 128], 512, num_classes, feat_t, layer_num)

        
        self.reset_params()
        # self.boundary_max_pooling = BoundaryMaxPooling()
        self._training = training
        
        if self._training:
            print('Load pretrained weight!')
            self.backbone.load_pretrained_weight()
      
    @staticmethod
    def weight_init(m):
        def glorot_uniform_(tensor):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
            scale = 1.0
            scale /= max(1., (fan_in + fan_out) / 2.)
            limit = np.sqrt(3.0 * scale)
            return nn.init._no_grad_uniform_(tensor, -limit, limit)

        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) \
                or isinstance(m, nn.ConvTranspose3d):
            glorot_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, step=1):
        # x should be [B, C, T, 128, 128] for CASME_sq
        if self.training:
            if step == 1:
                feat_dict = []
                flow_dict = []
                # flow_dict = [[],[],[]]
                img1 = x[:, :, 0, :, :]
                img1.squeeze(2)                     # torch.Size([16, 3, 128, 128])
                for i in range(x.shape[2]-1):
                    img2 = x[:, :, i+1, :, :]  
                    img2.squeeze(2)                 
                    pair = torch.cat((img1,img2), axis=1)  # torch.Size([16, 6, 128, 128])
                    flow, feat = self.backbone(pair) 
                    flow_dict.append(flow)
                    feat_dict.append(feat)
                    
    
                # print(feat_dict.shape)
                # print(flow_dict.shape)
                feat_dict.append(feat)
                feat_dict = torch.stack(feat_dict, 2)
                # print(feat_dict.shape)
                flow_dict = torch.stack(flow_dict, 1)
                return {
                    'feat': feat_dict,
                    'flow': flow_dict
                    }
                
            elif step == 2:
                # loc, conf, priors= self.coarse_pyramid_detection(x)
                loc, conf, priors, sample_conf= self.coarse_pyramid_detection(x)
                #loc, conf, priors, sample_loc= self.coarse_pyramid_detection(x)
                # loc, conf, priors, sample_conf, sample_loc= self.coarse_pyramid_detection(x)
                '''
                print(loc.shape)    
                print(conf.shape)    
                print(priors.shape)    
                print(sample_conf.shape) 
                '''
                 
            return {
                'loc': loc,
                'conf': conf,
                'priors': priors,
                'sample_conf': sample_conf,
                #'sample_loc': sample_loc,
                }
              
                
        else:
            feat_dict = []
            img1 = x[:, :, 0, :, :]
            img1.squeeze(2)                     # torch.Size([16, 3, 128, 128])
            for i in range(x.shape[2]-1):
                img2 = x[:, :, i+1, :, :]  
                img2.squeeze(2)                 
                pair = torch.cat((img1,img2), axis=1)  # torch.Size([16, 6, 128, 128])
                feat = self.backbone(pair)
                feat_dict.append(feat)
          
                
            # feat_dict = torch.cat((feat_dict,feat),axis=2)  # torch.Size([16, 1024, 8, 2, 2])
            feat_dict.append(feat)
            
            feat_dict = torch.stack(feat_dict, 2)
            loc, conf, priors = self.coarse_pyramid_detection(feat_dict)
            return {
                'loc': loc,
                'conf': conf,
                'priors': priors,
                }



def test_inference(repeats=3, clip_frames=256):
    model = FESNet(backbone_model='flownetsd',num_classes = 2, feat_t=48//2, layer_num=3,in_channels=3)
    model.train()
    model.cuda()
    import time
    run_times = []
    x = torch.randn([1, 3, clip_frames, 96, 96]).cuda()
    warmup_times = 2
    for i in range(repeats + warmup_times):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            y = model(x)
            print(y['loc'].shape)
            d 
        torch.cuda.synchronize()
        run_times.append(time.time() - start)

    infer_time = np.mean(run_times[warmup_times:])
    infer_fps = clip_frames * (1. / infer_time)
    print('inference time (ms):', infer_time * 1000)
    print('infer_fps:', int(infer_fps))
    # print(y['loc'].size(), y['conf'].size(), y['priors'].size())


if __name__ == '__main__':
    test_inference(20, 48)
