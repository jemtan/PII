import numpy as np
import itertools
import copy 
from datetime import datetime
import time
import os
import pickle

from sklearn.metrics import average_precision_score
import tensorflow as tf
import readData
import self_sup_task_poisson as self_sup_task
import poissonBlend
from multiprocessing import Process, Queue, Pool
from models.wide_residual_network import create_wide_residual_network_selfsup
from scipy.signal import savgol_filter
from utils import save_roc_pr_curve_data
import gc


def train_folder(input_dir,input_list,output_dir,data):

    gpu = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)

    data_frame = get_data_frame(data,input_dir,input_list,shuffle_order=True)
    mdl = get_mdl(data,data_frame,restore=False)

    submit_train(mdl,data_frame,output_dir,data)

    return

def predict_folder(test_dirs,output_dir,data):
    gpu = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)

    data_frame = get_data_frame(data,test_dirs[0][1],test_dirs[0][0],shuffle_order=False)
    mdl = get_mdl(data,data_frame,restore=True,model_dir=output_dir)

    submit_test(mdl,data_frame,test_dirs,output_dir)

    return

def get_data_frame(data,input_dir,input_list=None,shuffle_order=False,load_labels=False):
    if 'cxr' in data:
        batch_dim = [32,256,256,1]
        primary_axis = 0

    else:
        raise ValueError("data type not correctly defined. Either choose 'cxr' or add a new definition")


    data_frame = readData.data_frame(batch_dim,primary_axis)
    if input_list is None:
        input_list = os.listdir(input_dir)
    
    data_frame.load_data(input_list,input_dir,shuffle_order=shuffle_order,load_labels=load_labels)
    return data_frame

def get_mdl(data,data_frame,restore=False,model_dir=''):

    if 'cxr' in data:
        n, k = (16,4)#network size
        net_f='create_wide_residual_network_dec'
        n_classes = 1

    else:
        raise ValueError("data type not correctly defined. Either choose 'cxr' or add a new definition")


    if restore:
        #grab weights and build model
        model_fnames = os.listdir(model_dir)
        model_fnames = [fn for fn in model_fnames if 'weights' in fn][0]
        model_path = os.path.join(model_dir,model_fnames)
        print(model_path)
        mdl = tf.keras.models.load_model(model_path)

    else:
        #build new model
        mdl = create_wide_residual_network_selfsup(data_frame.batch_dim[1:],
            n_classes, n, k, net_f=net_f)

    return mdl

@tf.function
def train_step(mdl,x, y):
    loss_fn = mdl.compiled_loss
    with tf.GradientTape() as tape:
        logits = mdl(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, mdl.trainable_weights)
    mdl.optimizer.apply_gradients(zip(grads, mdl.trainable_weights))
    mdl.compiled_metrics.update_state(y, logits)
    return loss_value

@tf.function
def test_step(mdl,x, y):
    loss_fn = mdl.compiled_loss    
    logits = mdl(x, training=False)
    loss_value = loss_fn(y, logits)        
    return loss_value

@tf.function
def pred_step(mdl,x):
    pred = mdl(x, training=False)
    return pred

def grouped(iterable, n):
    #get n elements at a time
    return zip(*[iter(iterable)]*n)


def submit_train(mdl,data_frame,output_dir,data,epochs=50,cyclic_epochs=0,save_name='',multiproc=True,training_batch_size=32,batch_buffer=16):
    print('training start: {}'.format(datetime.now().strftime('%Y-%m-%d-%H%M')))

    num_classes = mdl.output_shape[-1]
    num_classes = None if num_classes <= 1 else num_classes
    fpi_args = {'num_classes':num_classes,
                'core_percent': 0.8,
                'tolerance': 1E-3,
                'batch_dim':[training_batch_size]+data_frame.batch_dim[1:],
                'batch_buffer':batch_buffer
    }

    elem_in_epoch = len(data_frame.file_list)
    
    if cyclic_epochs>0:
        half_cycle_len = elem_in_epoch//4
        lr_min = 1E-4
        lr_max = 1E-1
        half1 = np.linspace(lr_min,lr_max,half_cycle_len)
        half2 = np.linspace(lr_max,lr_min,half_cycle_len)
        lr_cycle = np.concatenate((half1,half2),0)

    if multiproc:
        in_queue = Queue()
        out_queue = Queue()
        n_workers = 8
        the_pool = Pool(n_workers, poissonBlend.multiproc_worker,(in_queue,out_queue))
        fpi_args.update(in_q=in_queue,out_q=out_queue)


    for epoch_i in range(epochs+cyclic_epochs):
        if epoch_i>epochs and elem_i < len(lr_cycle):
            #cyclic training portion, adjust learning rate
            tf.keras.backend.set_value(mdl.optimizer.lr, lr_cycle[elem_i])

        elem_i = 0
        #get subjects in pairs for mixing
        for batch_in,batch_in2 in grouped(data_frame.tf_dataset.repeat(),2):
            #draw batches indefinitely and break under condition
            if elem_i >= elem_in_epoch//2:
                #epoch complete
                break

            if not multiproc or in_queue.qsize() < training_batch_size*batch_buffer:
                #feed in new data
                elem1,elem2 = batch_in.numpy(),batch_in2.numpy()
            else:
                #feed in empty
                elem1,elem2 = [],[]

            patch_ex_result = self_sup_task.poisson_patch_ex(elem1,elem2,**fpi_args)#exchange patches
            if patch_ex_result is None:
                #no batches ready
                if in_queue.qsize() < training_batch_size*batch_buffer:
                    #buffer not full
                    continue
                else:
                    #buffer full, wait a bit to prevent data reading
                    time.sleep(1)
            else:
                #get batch and train, increment counter
                elem_set1,elem_set2 = patch_ex_result
                train_step(mdl,elem_set1[0],elem_set1[1])
                train_step(mdl,elem_set2[0],elem_set2[1])
                elem_i+=1


        print('epoch {}: {}'.format(str(epoch_i),datetime.now().strftime('%Y-%m-%d-%H%M')))
    
        #measure loss
        for batch_in,batch_in2 in grouped(data_frame.tf_dataset,2):
            break
        elem_set1,elem_set2 = self_sup_task.patch_ex(batch_in,batch_in2,**fpi_args)
        avg_loss = []
        avg_loss.append(test_step(mdl,elem_set1[0],elem_set1[1]))
        avg_loss.append(test_step(mdl,elem_set2[0],elem_set2[1]))
        avg_loss = np.mean(avg_loss)
        print('Avg loss: {}'.format(avg_loss))
        if epoch_i == 0:
            best_loss = avg_loss
        elif avg_loss < best_loss:
            best_loss = avg_loss
            print('new best loss')
            save_model(mdl,output_dir,save_name+'_bestLoss',time_stamp=False)

        if epoch_i % 10 == 0 or epoch_i>epochs:
            #save every 10 epochs or every epoch in cyclic mode
            save_model(mdl,output_dir,save_name)
        
    #save final model
    save_model(mdl,output_dir,save_name+'_final')

    if multiproc:
        in_queue.close()
        in_queue.join_thread()
        out_queue.close()
        out_queue.join_thread()

    return


def submit_test(mdl,data_frame,test_dirs,output_dir,batch_size=32,save_name='',pred_dir_name='pred_out'):
    print('testing start: {}'.format(datetime.now().strftime('%Y-%m-%d-%H%M')))
    #pred_out_dir = os.path.join(output_dir,pred_dir_name)
    #if not os.path.isdir(pred_out_dir):
    #    os.mkdir(pred_out_dir)

    subject_score = []
    subject_mean_quartile = []
    subject_label = []
    subject_i = 0

    for cur_test_dir in test_dirs:
        #load current test dataset
        cur_list,cur_dir = cur_test_dir
        data_frame.load_data(cur_list,cur_dir,shuffle_order=False)
        test_set_name = cur_list.split('/')[-1].split('.')[0]#just extract set name
        data_frame_label = 0 if 'norm_' in test_set_name else 1
 
        for batch_in in data_frame.tf_dataset:
            label = data_frame_label

            pred = pred_step(mdl,batch_in)
            output_chan = np.shape(pred)[-1]
            if output_chan > 1:
                pred *= np.arange(output_chan)/(output_chan-1)
                pred = np.sum(pred,-1,keepdims=True)

            #derive subject-level score
            im_level_score = np.mean(pred,axis=(1,2,3))
            subject_score.extend(im_level_score)

            top_25 = np.percentile(pred,75,axis=(1,2,3))
            im_level_score_quartile = np.mean(pred[pred>top_25])
            subject_mean_quartile.extend(im_level_score_quartile)#mean of top quartile values

            subject_label.extend(np.max(label)*np.ones_like(im_level_score))

            subject_i += 1
        
        print('{} complete: {}'.format(test_set_name,datetime.now().strftime('%Y-%m-%d-%H%M')))

    res_file_name = '{}_results_{}.npz'.format(save_name,datetime.now().strftime('%Y-%m-%d-%H%M'))
    res_file_path = os.path.join(output_dir, res_file_name)
    save_roc_pr_curve_data(np.array(subject_score), np.array(subject_label), res_file_path)

    res_file_name = '{}_quartile_results_{}.npz'.format(save_name,datetime.now().strftime('%Y-%m-%d-%H%M'))
    res_file_path = os.path.join(output_dir, res_file_name)
    save_roc_pr_curve_data(np.array(subject_mean_quartile), np.array(subject_label), res_file_path)

    return


def save_model(mdl,results_dir,fname,time_stamp=True):
    #save model
    if time_stamp:
        #mdl_weights_name = fname+'_{}_weights.h5'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))
        mdl_weights_name = fname+'_{}_weights'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))
    else:
        #mdl_weights_name = fname+'_weights.h5'
        mdl_weights_name = fname+'_weights'

    mdl_weights_path = os.path.join(results_dir, mdl_weights_name)
    mdl.save(mdl_weights_path)

    return



class index_sampling(object):
    def __init__(self,total_len):
        self.total_len = total_len
        self.ind_generator = rand_ind_fisheryates(self.total_len)

    def get_inds(self,batch_size):
        cur_inds = list(itertools.islice(self.ind_generator,batch_size))
        if len(cur_inds) < batch_size:
            #end of iterator - reset/shuffle
            self.ind_generator = rand_ind_fisheryates(self.total_len)
            cur_inds = list(itertools.islice(self.ind_generator,batch_size))
        return cur_inds

    def reset():
        self.ind_generator = rand_ind_fisheryates(self.total_len)
        return    

def rand_ind_fisheryates(num_inds):
    numbers=np.arange(num_inds,dtype=np.uint32)
    for ind_i in range(num_inds):
        j=np.random.randint(ind_i,num_inds)
        numbers[ind_i],numbers[j]=numbers[j],numbers[ind_i]
        yield numbers[ind_i]


if __name__ == '__main__':

    #example
    data = 'cxr'
    dataset_name = 'MaleAdultPA'
    train_list = 'train_lists/norm_'+dataset_name+'_train_list.txt'
    data_dir = '/path/to/ChestXray-NIHCC/images'
    out_dir = 'outDir'
    test_set_all = [['test_lists/norm_'+dataset_name+'_test_list.txt',
                     data_dir],
                    ['test_lists/anomaly_'+dataset_name+'_test_list.txt',
                     data_dir]]

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    #train
    train_folder(data_dir,train_list,out_dir,data)

    #test
    predict_folder(test_set_all,out_dir,data)


