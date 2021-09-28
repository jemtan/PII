#import abc
#import itertools
import numpy as np

from tensorflow.keras.utils import to_categorical
import copy
import poissonBlend
from multiprocessing import Process, Queue, Pool


def create_interp_mask(ima,patch_center,patch_width,patch_interp):
    dims=np.shape(ima)
    mask_i = np.zeros_like(ima)
    for frame_ind in range(dims[0]):
        coor_min = patch_center[frame_ind]-patch_width[frame_ind]
        coor_max = patch_center[frame_ind]+patch_width[frame_ind]
        
        #clip coordinates to within image dims
        coor_min = np.clip(coor_min,0,dims[1:3])     
        coor_max = np.clip(coor_max,0,dims[1:3])

        mask_i[frame_ind,
               coor_min[0]:coor_max[0],
               coor_min[1]:coor_max[1]] = patch_interp[frame_ind]
    return mask_i


def get_non_zero_range(ima1):
    #get non-zero columns in images
    nonzero = np.max(ima1,axis=(0,1,3))>0
    first_ind = np.argmax(nonzero)
    last_ind = len(nonzero)-np.argmax(nonzero[::-1])
    return first_ind, last_ind


def poisson_patch_ex(ima1,ima2,num_classes=None,core_percent=0.8,tolerance=None,in_q=None,out_q=None,batch_dim=[32,256,256,1],batch_buffer=16,sli_inds=None):
    #exchange patches between two image arrays based on a random interpolation factor

    #check for valid input of same size
    valid_input = len(ima1) > 0 and len(ima1) == len(ima2)
    if (in_q is None or in_q.qsize() < batch_buffer*batch_dim[0]) and valid_input:
        #if less than batch_buffer batches in the queue, add more

        #create random anomaly
        dims = np.array(np.shape(ima1))
        core = core_percent*dims#width of core region
        offset = (1-core_percent)*dims/2#offset to center core

        min_width = np.round(0.05*dims[1])
        max_width = np.round(0.2*dims[1])#make sure it is less than offset

        center_dim1 = np.random.randint(offset[1],offset[1]+core[1],size=dims[0])
        center_dim2 = np.random.randint(offset[2],offset[2]+core[2],size=dims[0])
        patch_center = np.stack((center_dim1,center_dim2),1)
        patch_width = np.random.randint(min_width,max_width,size=dims[0])
        if num_classes == None:
            patch_interp = np.random.uniform(0.05,0.95,size=dims[0])
        else:
            #interpolation between 0 and 1, num class options
            patch_interp = np.random.choice(num_classes-1,size=dims[0])/(num_classes-1)#subtract 1 to exclude default class
            patch_interp = 1-patch_interp    

        label = create_interp_mask(ima1,patch_center,patch_width,patch_interp)
        #make border zero for poisson blending
        border_mask = np.zeros_like(label)
        border_mask[:,1:-1,1:-1,:] = 1
        label = border_mask*label

        if in_q is not None:
            #put in queue
            for i in range(len(ima1)):
                index_in = sli_inds if sli_inds is None else sli_inds[i]
                in_q.put((ima1[i:i+1],ima2[i:i+1],label[i:i+1],True,index_in))

    if out_q is None and valid_input: 
        #no queues, process on its own
        all_ret = poissonBlend.poisson_blend(ima1,ima2,label)
        patchex1,patchex2 = all_ret['patch_ex1'],all_ret['patch_ex2']
        label = all_ret['mask']
        index_out = sli_inds

    elif out_q.qsize()>=batch_dim[0]:
        #at least one batch ready in finished queue, extract it
        patchex1,patchex2 = np.zeros(batch_dim),np.zeros(batch_dim)
        label = np.zeros(batch_dim)#label may not be defined
        ima1,ima2 = np.zeros(batch_dim),np.zeros(batch_dim)
        index_out = []#np.zeros(batch_dim[0])
        for i in range(batch_dim[0]):
            #get slices from queue (not necessarily in order)
            all_ret = out_q.get()
            patchex1[i],patchex2[i] = all_ret['patch_ex1'],all_ret['patch_ex2']
            label[i] = all_ret['mask']
            ima1[i],ima2[i] = all_ret['ima1'],all_ret['ima2']
            if 'index' in all_ret.keys():
                index_out.append(all_ret['index'])
    else:
        #no batch ready
        return None

    #process patches and labels before returning them 
    #anywhere the two values are not equal is valid
    spatial_mask = np.ceil(label)
    if tolerance:
        valid_label1 = np.any(
            np.floor(spatial_mask*ima1*tolerance**-1)*tolerance != \
            np.floor(spatial_mask*patchex1*tolerance**-1)*tolerance,
            axis=3)
        valid_label2 = np.any(
            np.floor(spatial_mask*ima2*tolerance**-1)*tolerance != \
            np.floor(spatial_mask*patchex2*tolerance**-1)*tolerance,
            axis=3)
            
    else:
        valid_label1 = np.any(spatial_mask*ima1 != spatial_mask*patchex1, axis=3)
        valid_label2 = np.any(spatial_mask*ima2 != spatial_mask*patchex2, axis=3)
    label1 = valid_label1[...,None]*label
    label2 = valid_label2[...,None]*label

    if num_classes is not None:
        label1 = label1*(num_classes-1)
        label1 = to_categorical(label1,num_classes)
        label2 = label2*(num_classes-1)
        label2 = to_categorical(label2,num_classes)
        
    ret_vals = [(patchex1,label1), (patchex2, label2)]
    if index_out is not None and len(index_out) > 0:
        ret_vals += [index_out]

    return tuple(ret_vals)


def patch_ex(ima1,ima2,num_classes=None,core_percent=0.8,tolerance=None,**kwargs):
    #exchange patches between two image arrays based on a random interpolation factor

    #create random anomaly
    dims = np.array(np.shape(ima1))
    core = core_percent*dims#width of core region
    offset = (1-core_percent)*dims/2#offset to center core

    min_width = np.round(0.05*dims[1])
    max_width = np.round(0.2*dims[1])#make sure it is less than offset

    center_dim1 = np.random.randint(offset[1],offset[1]+core[1],size=dims[0])
    center_dim2 = np.random.randint(offset[2],offset[2]+core[2],size=dims[0])
    patch_center = np.stack((center_dim1,center_dim2),1)
    patch_width = np.random.randint(min_width,max_width,size=dims[0])
    if num_classes == None:
        patch_interp = np.random.uniform(0.05,0.95,size=dims[0])
    else:
        #interpolation between 0 and 1, num class options
        patch_interp = np.random.choice(num_classes-1,size=dims[0])/(num_classes-1)#subtract 1 to exclude default class
        
    offset = 1E-5#offset to separate 0 patches from background
    mask_i = create_interp_mask(ima1,patch_center,patch_width,patch_interp+offset)
    patch_mask = np.clip(np.ceil(mask_i),0,1)#all patches set to 1
    mask_i = mask_i-patch_mask*offset#get rid of offset
    mask_inv = patch_mask-mask_i
    zero_mask = 1-patch_mask#zero in the region of the patch

    patch_set1 = mask_i*ima1 + mask_inv*ima2 #interpolate between patches
    patch_set2 = mask_inv*ima1 + mask_i*ima2

    patchex1 = ima1*zero_mask + patch_set1
    patchex2 = ima2*zero_mask + patch_set2

    if tolerance:
        valid_label = np.any(
            np.floor(patch_mask*ima1*tolerance**-1)*tolerance != \
            np.floor(patch_mask*ima2*tolerance**-1)*tolerance,
            axis=3)
            
    else:
        valid_label = np.any(patch_mask*ima1 != patch_mask*ima2, axis=3)
    label = valid_label[...,None]*mask_inv

    if num_classes is not None:
        label = label*(num_classes-1)
        label = to_categorical(label,num_classes)

    return (patchex1,label), (patchex2, label)



