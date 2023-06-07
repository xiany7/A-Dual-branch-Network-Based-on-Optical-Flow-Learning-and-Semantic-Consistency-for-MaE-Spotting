"""
This module provides data loaders and transformers for popular vision datasets.
"""




def get_spotting_dataset(dataset, cls, subject_out, clip_length, crop_size, stride, sample_fps, fps, vname2id, neg_pos_ratio, roi, long_term, with_flow, test_mode):
    if with_flow:
        if long_term:
            from .videodataset import VideoDataset
        else:
            from .videodataset_flow_short_term import VideoDataset
    else:
        from .videodataset_woflow import VideoDataset
    if dataset == 'ME2':
        if cls == 'macro':
            root_path = '/home/developers/xianyun/Data_zoo/CASME_sq/npy_rawpic'
            if roi==True and long_term==True:
                flow_path = '/home/developers/xianyun/Data_zoo/CASME_sq/npy_data_fbflow_Long_term_v4/' # wo_strain
            elif roi==False and long_term==True:
                flow_path = '/home/developers/xianyun/Data_zoo/CASME_sq/npy_data_fbflow_Long_term_dp/' 
            elif roi==True and long_term==False:
                flow_path = '/home/developers/xianyun/Data_zoo/CASME_sq/npy_data_fbflow_short_term_v4/' + cls
        else:
            root_path = '/home/developers/xianyun/Data_zoo/CASME_sq/npy_rawpic_micro'
            flow_path = '/home/developers/xianyun/Data_zoo/CASME_sq/npy_data_fbflow_Long_term_v4_micro_len32/'
            # flow_path = '/home/developers/xianyun/Data_zoo/CASME_sq/npy_data_fbflow_Long_term_v4_micro/'
        
        if test_mode == True:
            list_file = '/home/developers/xianyun/Data_zoo/CASME_sq/LOSO_info/v2/'+cls+'/val_info%d.json'%subject_out
            
        else:
            list_file = '/home/developers/xianyun/Data_zoo/CASME_sq/LOSO_info/v2/'+cls+'/train_info%d.json'%subject_out
           
    elif dataset == 'SAMM_LV':
        if cls == 'macro':
            root_path = '/home/developers/xianyun/Data_zoo/SAMM_LV/npy_rawpic'
            if roi==True and long_term==True:
                flow_path = '/home/developers/xianyun/Data_zoo/SAMM_LV/npy_data_fbflow_Long_term_v4_fps12/'
                # flow_path = '/home/developers/xianyun/Data_zoo/SAMM_LV/npy_data_fbflow_long_term_v4_seed0/' 
            elif roi==False and long_term==True:
                flow_path = '/home/developers/xianyun/Data_zoo/SAMM_LV/npy_data_fbflow_long_term_dp_fps12/' 
            elif roi==True and long_term==False:
                flow_path = '/home/developers/xianyun/Data_zoo/SAMM_LV/npy_data_fbflow_short_term_v4_fps12/' 
             
        else:    
            root_path = '/home/developers/xianyun/Data_zoo/SAMM_LV/npy_rawpic_micro_40'
            flow_path = '/home/developers/xianyun/Data_zoo/SAMM_LV/npy_data_fbflow_Long_term_v4_fps40_micro/' 
        
  
            
        
        
        if test_mode == True:
            list_file = '/home/developers/xianyun/Data_zoo/SAMM_LV/LOSO_info/v2/'+cls+'/val_info%d.json'%subject_out
        else:
            list_file = '/home/developers/xianyun/Data_zoo/SAMM_LV/LOSO_info/v2/'+cls+'/train_info%d.json'%subject_out
    
    return VideoDataset(root_path=root_path, list_file=list_file,  flow_path=flow_path, sample_fps=sample_fps, fps=fps, neg_pos_ratio=neg_pos_ratio, clip_length=clip_length,\
                         crop_size=crop_size, stride=stride,  subject=subject_out ,test_mode=test_mode, vname2id=vname2id)
    