from os import path
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import spsolve
import copy
import os
from multiprocessing import Process, Queue, Pool
from scipy.ndimage.interpolation import map_coordinates



def discretized_laplacian_operator(n, m):
    #set up matrix to perform discretized laplacian with derivative via finite difference with 4 neighboring pixels   
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
        
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    
    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)
    
    #could do the opposite way and start with identity
    #mat_A = scipy.sparse.identity(m*n)#.tolil|()

    return mat_A

def insert_identity(laplacian_op,mask):
    dims = np.shape(mask)
    for y in range(1, dims[0] - 1):
        for x in range(1, dims[1] - 1):
            if mask[y, x] == 0:
                k = x + y * dims[1]
                laplacian_op[k, k] = 1
                laplacian_op[k, k + 1] = 0
                laplacian_op[k, k - 1] = 0
                laplacian_op[k, k + dims[1]] = 0
                laplacian_op[k, k - dims[1]] = 0
    laplacian_op = laplacian_op.tocsc()
    return laplacian_op


def insert_laplacian(laplacian_op,mask):
    dims = np.shape(mask)
    for y in range(1, dims[0] - 1):
        for x in range(1, dims[1] - 1):
            if mask[y, x] > 0:
                k = x + y * dims[1]
                laplacian_op[k, k] = 4
                laplacian_op[k, k + 1] = -1
                laplacian_op[k, k - 1] = -1
                laplacian_op[k, k + dims[1]] = -1
                laplacian_op[k, k - dims[1]] = -1
    laplacian_op = laplacian_op.tocsc()
    return laplacian_op

def insert_laplacian_indexed(laplacian_op,mask,central_val=4):
    dims = np.shape(mask)
    mask_flat = mask.flatten()
    inds = np.array((mask_flat>0).nonzero())
    laplacian_op[inds,inds] = central_val
    laplacian_op[inds,inds+1] = -1
    laplacian_op[inds,inds-1] = -1
    laplacian_op[inds,inds+dims[1]] = -1
    laplacian_op[inds,inds-dims[1]] = -1
    laplacian_op = laplacian_op.tocsc() 
    return laplacian_op

def poisson_blend(ima1,ima2,mask,ret_orig=False,index=None):

    dims = np.shape(ima1)
    clip_vals1 = (np.min(ima1),np.max(ima1))
    clip_vals2 = (np.min(ima2),np.max(ima2))
    patch_ex1 = np.zeros_like(ima1)
    patch_ex2 = np.zeros_like(ima2)
    for sli in range(dims[0]):
        identity_matrix = scipy.sparse.identity(dims[1]*dims[2]).tolil()
        partial_laplacian = insert_laplacian_indexed(identity_matrix,mask[sli,:,:,0],central_val=4)        

        mask_flat = mask[sli,:,:,0].flatten()
        ima1_flat = ima1[sli,:,:,0].flatten()
        ima2_flat = ima2[sli,:,:,0].flatten()
       
        #discrete approximation of gradient 
        grad_matrix = insert_laplacian_indexed(identity_matrix,mask[sli,:,:,0],central_val=0)
        grad_matrix.eliminate_zeros()#get rid of central, only identity or neighbours
        grad_mask = grad_matrix!=0
        ima1_grad = grad_matrix.multiply(ima1_flat)#negative neighbour values
        ima1_grad = ima1_grad + scipy.sparse.diags(ima1_flat).dot(grad_mask)#add center value to sparse elements to get difference
        ima2_grad = grad_matrix.multiply(ima2_flat)
        ima2_grad = ima2_grad + scipy.sparse.diags(ima2_flat).dot(grad_mask)

        #mixing, favor the stronger gradient to improve blending
        alpha = np.max(mask_flat)
        ima1_greater_mask = (1-alpha)*np.abs(ima1_grad)>alpha*np.abs(ima2_grad)
        ima2_greater_mask = (1-alpha)*np.abs(ima2_grad)>alpha*np.abs(ima1_grad)
        ima1_guide = alpha*ima2_grad - ima1_greater_mask.multiply(alpha*ima2_grad) + ima1_greater_mask.multiply((1-alpha)*ima1_grad)
        ima2_guide = alpha*ima1_grad - ima2_greater_mask.multiply(alpha*ima1_grad) + ima2_greater_mask.multiply((1-alpha)*ima2_grad)

        ima1_guide = np.squeeze(np.array(np.sum(ima1_guide,1)))
        ima2_guide = np.squeeze(np.array(np.sum(ima2_guide,1)))

        ima1_guide[mask_flat == 0] = ima1_flat[mask_flat == 0]
        ima2_guide[mask_flat == 0] = ima2_flat[mask_flat == 0]

        x1 = spsolve(partial_laplacian,ima1_guide)
        x2 = spsolve(partial_laplacian,ima2_guide)    

        x1 = np.clip(x1,clip_vals1[0],clip_vals1[1])
        x2 = np.clip(x2,clip_vals2[0],clip_vals2[1])

        patch_ex1[sli,:,:,0] = np.reshape(x1,(dims[1],dims[2]))
        patch_ex2[sli,:,:,0] = np.reshape(x2,(dims[1],dims[2]))

    ret_vals = {'patch_ex1':patch_ex1,
                'patch_ex2':patch_ex2,
                'mask':mask}

    if ret_orig:
        ret_vals.update({'ima1':ima1,
                         'ima2':ima2})

    if index:
        ret_vals.update({'index':index})

    return ret_vals
     

        
def multiproc_worker(in_q,out_q):
    print(os.getpid(),"working")
    while True:
        item = in_q.get(True)
        ret_val = poisson_blend(*item)
        out_q.put(ret_val)



