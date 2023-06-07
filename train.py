import time
import datetime
import os
import sys
root_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
from core.data.dataloader import get_spotting_dataset
# from core.models.model_zoo.FESNet_flow_short_term import FESNet
# from core.models.model_zoo.FESNet_woflow_long_term import FESNet
from core.models.model_zoo.FESNet_flow_long_term_con import FESNet
from core.loss.multisegment_loss_con import MultiSegmentLoss
from core.loss.multiscaleloss import multiscaleEPE
from core.utils.segment_utils import softnms_v2
from core.eval.result_helper import pred_clips_analysis
# from core.utils.logger import setup_logger
sys.path.append('/home/developers/xianyun/spotting/core/data/dataset_new')
import data_provider as dt
import random
import argparse
from ruamel import yaml
import os
os.environ ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
               
def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)  # seed for module random
    np.random.seed(seed)  # seed for numpy
    torch.manual_seed(seed)  # seed for PyTorch CPU
    torch.cuda.manual_seed(seed)  # seed for current PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # seed for all PyTorch GPUs
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    #确保精度和能复现
    if cuda_deterministic:# slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def detection_collate_train(batch):
    flows = []
    clips = []
    annos = []
    for sample in batch:
        clips.append(sample[0])
        flows.append(sample[1])
        annos.append(sample[-1])

    return torch.stack(clips, 0), torch.stack(flows, 0), annos


def detection_collate_val(batch):
    names = []
    clips = []
    offsets = []
   
    for sample in batch:
        clips.append(sample[0])
        names.append(sample[1])
        offsets.append(sample[2])
        

    return torch.stack(clips, 0), names, offsets, sample[3]
    

def resume_training(resume, subject, model, optimizer, checkpoint_path, train_state_path, cls):
    start_epoch = 1
    if resume > 0:
        start_epoch = resume + 1
        model_path = os.path.join(checkpoint_path, 'checkpoint-{}-epoch{}-subject{}.ckpt'.format(cls, resume, subject))
        model.load_state_dict(torch.load(model_path))
        train_path = os.path.join(train_state_path, 'checkpoint-{}-epoch{}-subject{}.ckpt'.format(cls, resume, subject))
        state_dict = torch.load(train_path)
        optimizer.load_state_dict(state_dict['optimizer'])
    return start_epoch

def save_model(epoch, model, optimizer, checkpoint_path, train_state_path, subject, cls):
    checkpoint_name = 'checkpoint-{}-epoch{}-subject{}.ckpt'.format(cls, epoch, subject)
    '''
    torch.save(model.module.state_dict(),
               os.path.join(checkpoint_path, checkpoint_name))
    '''
    torch.save(model.state_dict(),
               os.path.join(checkpoint_path, checkpoint_name))
    if epoch == 50 or epoch == 40:
        torch.save({'optimizer': optimizer.state_dict()},
                os.path.join(train_state_path, checkpoint_name))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Trainer(object):
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.clip_length = cfg[args.dataset][args.cls]['clip_length']
        self.neg_pos_ratio = cfg[args.dataset][args.cls]['neg_pos_ratio']
        self.stride = cfg[args.dataset][args.cls]['stride']
        self.fps = cfg[args.dataset]['fps']
        self.sample_fps = cfg[args.dataset][args.cls]['sample_fps']
        self.batch_size = cfg[args.dataset][args.cls]['batch_size']
        self.num_class = cfg['num_class']
        self.layer_num = cfg[args.dataset][args.cls]['layer_num']
        self.loss_weight = cfg[args.dataset][args.cls]['loss_weight']
        # dataset and dataloader
        train_dataset = get_spotting_dataset(args.dataset, args.cls, args.subject_out, self.clip_length, cfg['crop_size'], self.stride,\
                                              self.sample_fps, self.fps, args.vname2id, self.neg_pos_ratio, roi=True, long_term=True, with_flow=True, test_mode=False)
        val_dataset = get_spotting_dataset(args.dataset, args.cls, args.subject_out, self.clip_length, cfg['crop_size'], self.stride,\
                                              self.sample_fps, self.fps, args.vname2id, self.neg_pos_ratio, roi=True, long_term=True, with_flow=True, test_mode=True)
        if args.distributed:
            train_sampler = data.distributed.DistributedSampler(train_dataset)
            validation_sampler = data.distributed.DistributedSampler(val_dataset)

            self.train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=self.batch_size,\
                                                        num_workers=4, collate_fn=detection_collate_train, pin_memory=False, drop_last=True)
     
            self.val_loader = torch.utils.data.DataLoader(val_dataset, sampler=validation_sampler, batch_size=self.batch_size*2,\
                                                        num_workers=4, collate_fn=detection_collate_val, pin_memory=False, drop_last=False)
        else:
            self.train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size,\
                                                        num_workers=4, collate_fn=detection_collate_train, pin_memory=False, drop_last=True)
     
            self.val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=self.batch_size,\
                                                        num_workers=4, collate_fn=detection_collate_val, pin_memory=False, drop_last=False)
        
        
        
        # create network
        self.model = FESNet(backbone_model=cfg['backbone'], num_classes = self.num_class, feat_t=self.clip_length//2, layer_num=self.layer_num, in_channels=3)
        
       
        self.model = self.model.to(device=args.device)
        if len(cfg[args.dataset][args.cls]['gpu_ind']) > 1:
            if args.distributed:
                self.model  = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],\
                                                                output_device=args.local_rank, find_unused_parameters=True)
            else:
                self.model = torch.nn.DataParallel(self.model, device_ids=cfg[args.dataset][args.cls]['gpu_ind'])
  
        self.model = self.model.to(args.device)
        # Setup loss
        self.criterion = MultiSegmentLoss(self.num_class, self.clip_length, cfg['piou'], 1.0, use_focal_loss=True)

        # Setup optimizer
        trained_vars = list(self.model.parameters())
      
        self.optimizer = torch.optim.Adam(trained_vars, lr=cfg[args.dataset]['lr'], weight_decay=cfg['weight_decay'])

        # Start training
        self.start_epoch = resume_training(cfg[args.dataset][args.cls]['resume'], args.subject_out, self.model, self.optimizer, args.save_dir, args.train_state_path, args.cls)

      

    def train(self):
        epochs = self.cfg[args.dataset][args.cls]['epochs']
        if self.args.local_rank==0:
            print('Start training, Total Epochs: {:d}'.format(epochs))

        # initialize loss
        loss_loc_val = AverageMeter()
        loss_conf_val = AverageMeter()
        loss_flow_val = AverageMeter()
        loss_con_val = AverageMeter()
        cost_val = AverageMeter()

        self.model.train()
        start_time = time.time()
        for i in range(self.start_epoch, epochs + 1):
            self.epoch_now = i
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(self.epoch_now)
                self.val_loader.sampler.set_epoch(self.epoch_now)
           
            for n_iter, (clips, target_flow, targets) in enumerate(self.train_loader):
                
                # print('clips:',clips.shape)  # torch.Size([24, 3, 48, 128, 128])
                # print('target_flow:',target_flow.shape)   # torch.Size([24, 47, 2, 128, 128])
               
                clips = clips.to(device=self.args.device)
                targets = [t.to(device=self.args.device) for t in targets]
                target_flow = target_flow.to(device=self.args.device)
                c, h, w = target_flow.shape[-3:]
                
                output_dict = self.model(clips, step=1)
                #output_dict = self.model(clips)
                if self.args.sparse:
                    # Since Target pooling is not very precise when sparse,
                    # take the highest resolution prediction and upsample it instead of downsampling target
                    '''
                    for j in range(len(output_dict['flow'])):
                        output_dict['flow'][j] = F.interpolate(output_dict['flow'][j], (c,h,w))
                    '''
                    output_dict['flow'] = F.interpolate(output_dict['flow'], (c,h,w))
                    
                loss_f = multiscaleEPE(output_dict['flow'], target_flow.view(-1, c, h, w), weights=self.cfg['multiscale_weights'], sparse=self.args.sparse)
          
                
               
                output_dict = self.model(output_dict['feat'], step=2)
                
                # loss_l, loss_c = self.criterion([output_dict['loc'], output_dict['conf'], output_dict['priors'][0]], targets)
                loss_l, loss_c, loss_con = self.criterion([output_dict['loc'], output_dict['conf'], output_dict['priors'][0], output_dict['sample_conf']], targets)
                # loss_l, loss_c, loss_con = self.criterion([output_dict['loc'], output_dict['conf'], output_dict['priors'][0], output_dict['sample_loc']], targets)
                # loss_l, loss_c, loss_con = self.criterion([output_dict['loc'], output_dict['conf'], output_dict['priors'][0], output_dict['sample_conf'], output_dict['sample_loc']], targets)
                
                loss_l = loss_l * self.loss_weight[0]
                loss_c = loss_c * self.loss_weight[1]
                loss_f = loss_f * self.loss_weight[2]
                loss_con = loss_con * self.loss_weight[3]

                cost = loss_l + loss_c + loss_f + loss_con

                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()
                
                loss_loc_val.update(loss_l.cpu().detach().numpy())
                loss_conf_val.update(loss_c.cpu().detach().numpy())
                loss_flow_val.update(loss_f.cpu().detach().numpy())
                loss_con_val.update(loss_con.cpu().detach().numpy())
                cost_val.update(cost.cpu().detach().numpy())


            eta_seconds = ((time.time() - start_time) / i) * (epochs- i)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))


            if self.args.local_rank==0:
             
                print(
                    "[Train] subject: {:d}/{:d} || Epoch: {:d}/{:d} || Loss: {:.4f} || loc: {:.4f} || conf: {:.4f} || flow: {:.4f} || con: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                    self.args.subject_out+1 , self.args.all_subject, self.epoch_now, epochs, cost_val.avg, loss_loc_val.avg, loss_conf_val.avg, loss_flow_val.avg, loss_con_val.avg, 
                    str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))
         
            
            if self.epoch_now % self.cfg['val_epoch'] == 0:
                self.validation()
                self.model.train()
                if self.args.local_rank==0:
                    save_model(i, self.model, self.optimizer, self.args.save_dir, self.args.train_state_path, subject=self.args.subject_out, cls=self.args.cls)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        if self.args.local_rank==0:
            print("Total training time: {}".format(total_training_str))
       

    def validation(self):
        # model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        self.model.eval()
        pred_clips = []
        true_labels = []
      
  
        for _ in range(len(self.args.vname2id)):
            pred_clips.append([])
            true_labels.append([])
  
     
        for _, (clips, video_names, offsets, true_labels) in enumerate(self.val_loader):
            
            clips = clips.to(device=args.device)
            with torch.no_grad():
                output_dict = self.model(clips)

            locs, confs, priors = output_dict['loc'], output_dict['conf'], output_dict['priors'][0]
            # print("locs:", locs.shape)  # torch.Size([bs, len(priors), 2])
            # print("confs:", confs.shape)  # torch.Size([bs, len(priors), num_class])
            for loc, conf, video_name, offset in zip (locs, confs, video_names, offsets):
                
                # 生成时序边界框
                decoded_segments = torch.cat(
                    [priors[:, :1] * self.clip_length  - loc[:, :1],
                    priors[:, :1] * self.clip_length  + loc[:, 1:]], dim=-1)
                decoded_segments.clamp_(min=0, max=self.clip_length)
                score_func = nn.Softmax(dim=-1)
                conf = score_func(conf)
                conf = conf.view(-1, self.num_class).transpose(1, 0)
                # print("conf:", conf.shape)  # torch.Size([num_class, len(priors)])
                conf_scores = conf.clone()                  
                c_mask = conf_scores[1] > 0.5 # torch.Size([len(priors)])
                # print("c_mask:", c_mask)
                scores = conf_scores[1][c_mask] # torch.Size([len(priors)])
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_segments)
                segments = decoded_segments[l_mask].view(-1, 2)
                segments = (segments + offset) / self.sample_fps * self.fps
                # segments = torch.Tensor(segments.cpu().detach().numpy())
                # scores = torch.Tensor(scores.unsqueeze(1).cpu().detach().numpy())
                segments = torch.cat([segments, scores.unsqueeze(1)], -1)
                pred_clips[self.args.vname2id[video_name]].append(segments)

        if self.args.distributed:
            pred_clips_all = []
            for i in range(len(self.args.vname2id)):
                if len(pred_clips[i]) == 0:
                    pred_clips[i] = torch.zeros([1,3]).to(device=self.args.device)
                else:
                    pred_clips[i] = torch.cat(pred_clips[i], 0)
                preds_num_list = [torch.zeros([1]).to(device=self.args.device) for _ in range(4)]
                torch.distributed.barrier()
                torch.distributed.all_gather(preds_num_list, torch.Tensor([len(pred_clips[i])]).to(device=self.args.device))
                max_preds_num = int(max(preds_num_list).item())
                preds_gather_list = [torch.zeros([max_preds_num, 3]).to(device=self.args.device) for _ in range(4)]
                if len(pred_clips[i]) < max_preds_num:
                    padding = torch.zeros([max_preds_num - len(pred_clips[i]) , 3]).to(device=self.args.device)
                    pred_clips[i] = torch.cat((pred_clips[i], padding), 0)
                torch.distributed.barrier()
                torch.distributed.all_gather(preds_gather_list, pred_clips[i])
                if self.args.local_rank==0:
                    preds_gather_list = torch.cat(preds_gather_list, 0)
                    # print(preds_gather_list)
                    mask = preds_gather_list[:,2] > 0
                    mask = mask.unsqueeze(1).expand_as(preds_gather_list)
                    preds_gather_list = preds_gather_list[mask].view(-1, 3)
                    # print(preds_gather_list)
                    if len(preds_gather_list) == 0:
                        pred_clips_all.append([])
                        continue
                    # print(preds_gather_list.shape)
                    if preds_gather_list.shape[0] > 1:
                        preds_gather_list, _ = softnms_v2(preds_gather_list, sigma=self.cfg['nms_sigma'], top_k=self.cfg[self.args.dataset][self.args.cls]['max_props_num'])
                        s_mask = preds_gather_list[:, 2] > torch.mean(preds_gather_list[:, 2]) + cfg[args.dataset][args.cls]['p'] * (torch.max(preds_gather_list[:, 2])-torch.mean(preds_gather_list[:, 2]))
                        s_mask = s_mask.unsqueeze(1).expand_as(preds_gather_list)
                        preds_gather_list = preds_gather_list[s_mask].view(-1, 3)
                    pred_clips_all.append(preds_gather_list.detach().cpu().numpy())
            pred_clips = pred_clips_all     
        else:
            for i in range(len(self.args.vname2id)):
                if len(pred_clips[i]) == 0:
                    continue
                pred_clips[i] = torch.cat(pred_clips[i], 0)
                if len(pred_clips[i])> 1:
                    pred_clips[i], _ = softnms_v2(pred_clips[i], sigma=self.cfg['nms_sigma'], top_k=self.cfg[self.args.dataset][self.args.cls]['max_props_num'])
       
                    s_mask = pred_clips[i][:, 2] > torch.mean(pred_clips[i][:, 2]) + cfg[args.dataset][args.cls]['p'] * (torch.max(pred_clips[i][:, 2])-torch.mean(pred_clips[i][:, 2]))
                    s_mask = s_mask.unsqueeze(1).expand_as(pred_clips[i])
                    pred_clips[i] = pred_clips[i][s_mask].view(-1, 3)
        if self.args.local_rank==0:    
   
            N_clips, _, N_labels, N_find_labels, _, _, recall, _, precision_small, _, F1_small = pred_clips_analysis(pred_clips, true_labels, 0.5)
            
            print("[ Val ] Total : {:3d} || TP : {:3d} || FP : {:4d} || FN : {:3d} || Prec: {:.4f} || Recall: {:.4f} || F1-score: {:.4f}".format(N_labels,\
                                                                N_find_labels, N_clips-N_find_labels, N_labels-N_find_labels, precision_small, recall, F1_small))
      
        
            args.tps[self.args.subject_out, self.epoch_now-1] = N_find_labels
            args.props[self.args.subject_out, self.epoch_now-1] = N_clips
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Macro-expression spotting with Pytorch')
    parser.add_argument('--config_file', type=str,
                        default='/home/developers/xianyun/spotting/scripts/cfg_flownetsd.yaml',
                        help='config file')
    parser.add_argument('--dataset', type=str, default='SAMM_LV',
                        choices=['ME2', 'SAMM_LV'],
                        help='dataset name (default: ME2)')
    parser.add_argument('--cls', type=str, default='macro',
                        choices=['macro', 'micro'],
                        help='expression class (default: macro)')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='node rank for distributed training')
    parser.add_argument('--sparse', type=bool, default= True,
                        help='coumpute sparse flow')
    parser.add_argument('--save_dir', type=str, default='../resume/test',
                        help='the path to save checkpoint')
    parser.add_argument('--train_state_path', type=str, default='../resume/test',
                        help='the path to save train state')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config_file))
    args.distributed = cfg[args.dataset][args.cls]['distributed']
    args.deterministic = cfg[args.dataset][args.cls]['cuda_deterministic']
    if args.distributed:
        torch.distributed.init_process_group(backend="nccl")
        args.local_rank = torch.distributed.get_rank()
        init_seeds(1 + args.local_rank, args.deterministic)
        torch.cuda.set_device(args.local_rank)
    else:
        args.local_rank = 0

    if not cfg['no_cuda'] and torch.cuda.is_available():
        if args.distributed:
            args.device = torch.device("cuda", args.local_rank)
        else:
            args.device = torch.device("cuda")
    else:
        args.device = torch.device('cpu')

  
    tp, fp, fn, prec, rec, score = {}, {}, {}, {}, {}, {}

    data_handler = dt.DataProvider(args.dataset, 'LOSO')
    args.videos = data_handler.produce_videos(args.cls,'LOSO')


    args.all_subject = len(args.videos)
    if args.local_rank==0: 
        args.tps = np.zeros((args.all_subject, cfg[args.dataset][args.cls]['epochs']))
        args.props = np.zeros((args.all_subject, cfg[args.dataset][args.cls]['epochs']))

    
    for subject_out in range(args.all_subject):
        args.subject_out = subject_out
        key = 'subject{:d}:'.format(args.subject_out+1)
        args.vname2id = {}
        for j in range(len(args.videos[key])):
            name = args.videos[key][j][0] + '/' + args.videos[key][j][1] 
            args.vname2id[name] = j
        # print(args.vname2id)
        init_seeds(1000, args.deterministic)
       
        trainer = Trainer(args, cfg)
        trainer.train()
        tmp_tps = np.array(np.sum(args.tps, axis=0))
        tmp_props = np.array(np.sum(args.props, axis=0))
        print('TPs:{}'.format(tmp_tps))
        print('Props:{}'.format(tmp_props))
        print('Precs:{}'.format(tmp_tps/tmp_props))
        torch.cuda.empty_cache()
        

    if args.local_rank==0:     
        tps_epochs = np.array(np.sum(args.tps, axis=0))
        props_epochs = np.array(np.sum(args.props, axis=0))
        prec_epochs = tps_epochs/props_epochs
        rec_epochs = tps_epochs/cfg[args.dataset][args.cls]['total']
        score_epochs = np.zeros(cfg[args.dataset][args.cls]['epochs'])
        for i in range(cfg[args.dataset][args.cls]['epochs']):
            fs = 2*prec_epochs[i]*rec_epochs[i]/(prec_epochs[i]+rec_epochs[i]) if (prec_epochs[i]+rec_epochs[i])!=0 else 0
            score_epochs[i] = fs
        print('TPs of every epoch:{}'.format(tps_epochs))
        print('Props of every epoch:{}'.format(props_epochs))
        print('Prcision of every epoch:{}'.format(prec_epochs))
        print('Recall of every epoch:{}'.format(rec_epochs))
        print('F1-score of every epoch:{}'.format(score_epochs))

        tp[args.cls] = int(tps_epochs[np.argmax(score_epochs)])
        fp[args.cls] = int(props_epochs[np.argmax(score_epochs)] - tps_epochs[np.argmax(score_epochs)])
        fn[args.cls] = int(cfg[args.dataset][args.cls]['total']- tps_epochs[np.argmax(score_epochs)])
        prec[args.cls] = prec_epochs[np.argmax(score_epochs)]
        rec[args.cls] = rec_epochs[np.argmax(score_epochs)]
        score[args.cls] = np.max(score_epochs)
        print('[{}] Total : {:3d} || TP : {:3d} || FP : {:4d} || FN : {:3d} || Prec: {:.4f} || Recall: {:.4f} || F1-score: {:.4f} '.format(args.cls, cfg[args.dataset][args.cls]['total'], tp[args.cls], fp[args.cls], fn[args.cls], prec[args.cls], rec[args.cls], score[args.cls]))
    
