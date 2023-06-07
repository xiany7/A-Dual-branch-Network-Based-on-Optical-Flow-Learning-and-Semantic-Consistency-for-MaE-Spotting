from torch.utils.data import Dataset

from PIL import Image
from .videotransforms import *
import os
import os.path
import numpy as np
from numpy.random import randint
import torch
import math
import json
import pickle

def split_videos(list_file,
                 npy_data_path,
                 sample_fps,
                 fps,
                 neg_pos_ratio,
                 clip_length=16,
                 stride=2,
                 test_mode=False,
                 vname2id=None
                 ):   
    # video_infos = get_video_info(config['dataset']['training']['video_info_path'])
    # video_annos = get_video_anno(video_infos,
    #                              config['dataset']['training']['video_anno_path'])
    video_list = []
    data_dict = {}
    true_labels = []
    if test_mode:
        true_labels = []
        for _ in range(len(vname2id)):
            true_labels.append([])

    with open(list_file, 'r') as fobj:
        infos = json.load(fobj)
    infos = infos['database']

    for video_name in infos.keys():
        # loading video frame data
        data = np.load(os.path.join(npy_data_path, video_name + '.npy'))
        data = np.transpose(data, [3, 0, 1, 2])
        data_dict[video_name] = data
        count = infos[video_name]['numFrame']
        
        ratio = sample_fps / fps
        sample_count = math.floor(count * ratio)

        # split videos
        if sample_count <= clip_length:
            offsetlist = [0]
        else:
            offsetlist = list(range(0, sample_count - clip_length + 1, stride))
            if (sample_count - clip_length) % stride:
                offsetlist += [sample_count - clip_length]

        if test_mode:
 
            for i in range(len(infos[video_name]['annotations'])):
                segment = infos[video_name]['annotations'][i]['segment']
                true_labels[vname2id[video_name]].append([segment[0], segment[1]])
            for offset in offsetlist:
                video_list.append({
                    'video_name': video_name,
                    'offset': offset,
                })
         
       
        else:
            # modifying label
            annos = []
            true_labels = []
            for i in range(len(infos[video_name]['annotations'])):
                segment = infos[video_name]['annotations'][i]['segment']
                label = infos[video_name]['annotations'][i]['label']
                # print('segment',segment)
                annos.append([segment[0] * ratio, segment[1] * ratio, label])
                true_labels.append([segment[0] * ratio, segment[1] * ratio])
          
            # print("annos:",annos)
            for offset in offsetlist:
                # print('offset:', offset)
                left, right = offset + 1, offset + clip_length
                cur_annos = []
                save_offset = False
                for anno in annos:
                    max_l = max(left, anno[0])
                    min_r = min(right, anno[1])
                    ioa = (min_r - max_l) * 1.0 / (anno[1] - anno[0])
                    # print('ioa:', ioa)
                    if ioa == 1.0:   # sample_count <= clip_length
                        save_offset = True
                    if ioa >= 0.5:
                        cur_annos.append([max(anno[0] - offset, 1),
                            min(anno[1] - offset, clip_length),
                            anno[2]])
                if save_offset:
                    video_list.append({
                        'video_name': video_name,
                        'offset': offset,
                        'annos': cur_annos,
                    })
          
            true_labels.sort()
          
            neg_regions = []
            i = 0    # 考虑重叠的表情区间
            start = true_labels[i][0]
            end = true_labels[i][1]
            while i < len(true_labels)-1:
                if end < true_labels[i+1][0]:
                    neg_regions.extend([start, end])
                    start = true_labels[i+1][0]
                    end = true_labels[i+1][1]
                else:
                    if true_labels[i][1] > true_labels[i+1][1]:
                        end = true_labels[i][1]
                    else:
                        end = true_labels[i+1][1]
                i = i+1
                if i == len(true_labels)-1:
                    neg_regions.extend([start, end])
                    break
                        
             
            neg_regions.extend([1, sample_count])
            neg_regions.sort()
            
            
            neg_regions = [[neg_regions[i], neg_regions[i + 1]] for i in range(len(neg_regions) - 1)]
            neg_regions = list(filter(lambda x: x not in true_labels and math.floor(x[1]) - math.ceil(x[0]) > clip_length, neg_regions))
            
            if len(neg_regions)==0:
                continue
            neg_num = neg_pos_ratio * len(annos)
            # print(neg_num)
            for i in range(int(neg_num)):
                region = random.choice(neg_regions)
                max_offset = math.floor(region[1] - clip_length -1)
                if max_offset > 1:
                    offset = random.choice(range(1, math.floor(region[1] - clip_length -1)))
                if max_offset == 1:
                    offset = 1
                # print(offset)
                video_list.append({
                    'video_name': video_name,
                    'offset': offset,
                    'annos': [[0, 0, 0]],
                    })
            # print(video_list)
          
            
    
    return video_list, data_dict, true_labels
   
        

def annos_transform(annos, clip_length):
    res = []
    for anno in annos:
        res.append([
            anno[0] * 1.0 / clip_length,
            anno[1] * 1.0 / clip_length,
            anno[2]
        ])
    return res



class VideoDataset(Dataset):
    def __init__(self, 
                 root_path, 
                 list_file,
                 flow_path,
                 sample_fps,
                 fps,
                 neg_pos_ratio,
                 subject,
                 clip_length=16,
                 crop_size=96,
                 stride=2,
                 rgb_norm=True,
                 origin_ratio=0.5,
                 test_mode=False,
                 vname2id=None):
        self.flow_path = flow_path
        self.test_mode = test_mode
        self.clip_length = clip_length
        self.video_list, self.data_dict, self.true_labels = split_videos(list_file, root_path, sample_fps, fps,\
                                                         neg_pos_ratio, self.clip_length, stride, self.test_mode, vname2id)
        self.crop_size = crop_size
        self.random_crop = RandomCrop(200)
        self.random_flip = RandomHorizontalFlip(p=0.5)
        self.center_crop = CenterCrop(200)
        self.rgb_norm = rgb_norm
        self.origin_ratio = origin_ratio
        '''
        # print("/home/developers/xianyun/spotting/core/data/samm_lv/micro/video_list%d.pkl"%subject)
        with open("/home/developers/xianyun/spotting/core/data/me2/micro/video_list%d.pkl"%subject, "wb") as tf:
            pickle.dump(self.video_list,tf)
        '''
        
        
        
       
        
        
        
       


    def __len__(self):
        return len(self.video_list)


    def __getitem__(self, index):
        sample_info = self.video_list[index] # [video_name offset annos]
        video_data = self.data_dict[sample_info['video_name']]
        offset = sample_info['offset']
        input_data = video_data[:, offset: offset + self.clip_length]  #找到对应视频片段的图片数据
        c, t, h, w = input_data.shape
        if t < self.clip_length:
            # padding t to clip_length
            pad_t = self.clip_length - t
            zero_clip = np.zeros([c, pad_t, h, w], input_data.dtype)
            input_data = np.concatenate([input_data, zero_clip], 1)

        if self.test_mode:
            input_data = self.center_crop(input_data)
            input_data = torch.from_numpy(input_data).float()
            input_data = torch.nn.functional.interpolate(input=input_data, size=(self.crop_size, self.crop_size), mode='bilinear', align_corners=False)
            if self.rgb_norm:
                input_data = (input_data / 255.0) * 2.0 - 1.0  # 归一化[-1, 1]
 
            return input_data, sample_info['video_name'], offset, self.true_labels
        else:
            target_flow = np.load(os.path.join(self.flow_path, sample_info['video_name'] +'/'+str(offset) + '.npy'))
            # print(target_flow.shape)
            # target_flow = flow_data[offset: offset + self.clip_length-1,:]  #找到对应视频片段的图片数据
            target_flow = np.transpose(target_flow, [0, 3, 1, 2])
            t, c, h, w = target_flow.shape
            # assert c == 3,os.path.join(self.flow_path, sample_info['video_name'] +'/'+str(offset) + '.npy')
            if t < self.clip_length-1:
                # padding t to clip_length
                pad_t = self.clip_length - t - 1
                zero_clip = np.zeros([pad_t, c, h, w], input_data.dtype)
                target_flow = np.concatenate([target_flow, zero_clip], 0)
            
            # input_data = self.random_flip(self.random_crop(input_data))  # (3, 48, 128, 128)
            # target_flow = self.random_flip(self.random_crop(target_flow))  # (47, 2, 128, 128)
            input_data = self.random_crop(input_data)  # (3, 48, 128, 128)
            target_flow = self.random_crop(target_flow)  # (47, 2, 128, 128)
            input_data, target_flow = self.random_flip([input_data,target_flow])  # (3, 48, 128, 128)
            
            # target_flow = self.random_flip(target_flow)  # (47, 2, 128, 128)
            input_data = torch.from_numpy(input_data).float()
            target_flow = torch.from_numpy(target_flow).float()
            input_data = torch.nn.functional.interpolate(input=input_data, size=(self.crop_size, self.crop_size), mode='bilinear', align_corners=False)
            target_flow = torch.nn.functional.interpolate(input=target_flow, size=(self.crop_size, self.crop_size), mode='bilinear', align_corners=False)
            
            if self.rgb_norm:
                input_data = (input_data / 255.0) * 2.0 - 1.0  # 归一化[-1, 1]
            annos = sample_info['annos']
            annos = annos_transform(annos, self.clip_length)
            annos = np.stack(annos, 0)
            annos = torch.FloatTensor(annos)
    
            return input_data, target_flow, annos
            
            
  
            

        
     
        
            


