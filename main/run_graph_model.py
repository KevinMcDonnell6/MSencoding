
import numpy as np
import tensorflow as tf
from sklearn import metrics
from multiprocessing import Pool


import model_utils
import GraphModel as Model

import data_processing as dp

import os
import sys
import time 




logging_dir = "graphs/"+model_utils.args.logging_dir

print(logging_dir)
log_file = os.getcwd()+"/"+logging_dir+"/LogFile.txt"
if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(logging_dir)
log_file_handle = open(log_file, 'a+')


if model_utils.data_type=="msp":
    with open(model_utils.val_file) as filehandle:
        val_spec = dp.read_spectra_only_msp_prosit(filehandle,
                                                   model_utils.data_type, 
                                                   model_utils.val_spec_loc)
 
if model_utils.data_type=="mgf":
    with open(model_utils.val_file) as filehandle:
        val_spec = dp.read_spectra_only(filehandle,
                                        model_utils.data_type, 
                                        model_utils.val_spec_loc)



def sigmoid(x):
    
    return 1/(1+np.exp(-x))

def get_random_stack(data,stack_size):

        indices = np.random.randint(low=0,
                                    high=len(data),
                                    size=stack_size)

        return([data[i] for i in indices])

multiprocessing_pool = Pool(5)



def tf_run_data(model,
                session,
                tf_mode,
                data,
                log_file_handle,
                evaluate=False,
                print_to_file=True,
                print_to_screen=True):   
                
    stack_size = len(data[0])
    batch_size = model_utils.train_batch_size      
    
    logit_batches = []
    label_batches = []
    mask_batches = []
    
    loss_sum=0.0
    
    peak_total = 0
    
    extract_times = []
    step_times = []
    
    tp = 0
    tn = 0
    by_total = 0
    not_by_total = 0
    by_total_pred = 0
    not_by_total_pred = 0
    
    
    for start in range(0, stack_size, batch_size):
        end = min(start+batch_size, stack_size)
        
        indices = np.arange(start,end)
        batch_size = len(indices)
        
        extract_start_time = time.time()
        
        b_or_y = np.array(data[0])[indices]
        by_mask = np.array(data[1])[indices]
        
        AA_windows = np.array(data[2])[indices]
        AA_windows_bw = np.array(data[3])[indices]
        
        features = np.array(data[4])[indices]
        
        ion_windows = np.array(data[5])[indices]
        
        indx, vals = list(zip(*[data[6][idx] for idx in indices]))
        d0,d1,d2 = list(zip(*indx))
        adjacency = np.zeros((batch_size,
                              model_utils.max_num_peaks,
                              model_utils.max_num_peaks,
                              model_utils.vocab_size-3
                              ),
                              dtype=np.float32)
        
        for idx in range(batch_size):
            adjacency[idx,d0[idx],d1[idx],d2[idx]]=1
            
        extract_times.append(time.time()-extract_start_time)
        
        
        step_start_time = time.time()
        
        logits, loss, _ = model.step(session=session,
                                     mode=tf_mode,
                                     targets=b_or_y,
                                     mask=by_mask,
                                     adj_matrix=adjacency,
                                      AA_windows_fw=AA_windows,
                                      AA_windows_bw=AA_windows_bw,
                                      ion_windows = ion_windows,
                                     features=features)
        
        step_times.append(time.time()-step_start_time)
        
        loss_sum += loss*batch_size
                
        if evaluate:
            peak_total += np.sum(by_mask)
            
            by_total += np.sum(b_or_y)
            not_by_total += np.sum(by_mask)-np.sum(b_or_y)
            tp += np.sum(np.greater_equal(sigmoid(np.squeeze(logits)),0.5)*b_or_y)
            tn+=np.sum((1-np.greater_equal(sigmoid(np.squeeze(logits)),0.5))*(1-b_or_y)*by_mask)
            by_total_pred += np.sum(np.greater_equal(sigmoid(np.squeeze(logits)),0.5)*by_mask)
            not_by_total_pred += np.sum(np.less(sigmoid(np.squeeze(logits)),0.5)*by_mask)
            
            logit_batches.append(logits)
            label_batches.append(b_or_y)
            mask_batches.append(by_mask)
        
            
    print("")
    print("Avg {0} Extract time: {1}".format(tf_mode,np.round(np.mean(extract_times),3))) 
    print("Avg {0} Step time: {1}".format(tf_mode,np.round(np.mean(step_times),3)))
    print("Total {0} Step (model) time: {1}".format(tf_mode,np.round(np.sum(step_times),3)))

    
    print("Avg {0} Extract time: {1}".format(tf_mode,np.round(np.mean(extract_times),3)),file=log_file_handle)
    print("Avg {0} Step time: {1}".format(tf_mode,np.round(np.mean(step_times),3)),file=log_file_handle)     
    print("Total {0} Step (model) time: {1}".format(tf_mode,np.round(np.sum(step_times),3)),file=log_file_handle)           
    
    if evaluate:
        AUC = metrics.roc_auc_score(np.concatenate(label_batches).flatten(),
                                                             np.concatenate(logit_batches).flatten(),
                                                             sample_weight=np.concatenate(mask_batches).flatten())
        if print_to_file:
            
            print("total accuracy:{0:.4f}".format((tp+tn)/max(1,peak_total)),file=log_file_handle)
            print("by recall:     {0:.4f}".format(tp/max(1,(by_total))),file=log_file_handle)
            print("by prec:       {0:.4f}".format(tp/max(1,by_total_pred)),file=log_file_handle)
            print("not by recall: {0:.4f}".format(tn/max(1,not_by_total)),file=log_file_handle)
            print("not by prec  : {0:.4f}".format(tn/max(1,not_by_total_pred)),file=log_file_handle,flush=True)
            
            print("AUC:{0:.4f}".format(AUC),
                  file=log_file_handle)
        if print_to_screen:
    
            print("AUC:{0:.4f}".format(AUC))
            
            
            print("total accuracy:{0:.4f}".format((tp+tn)/max(1,peak_total)))
            print("by recall:     {0:.4f}".format(tp/max(1,(by_total))))
            print("by prec:       {0:.4f}".format(tp/max(1,by_total_pred)))
            print("not by recall: {0:.4f}".format(tn/max(1,not_by_total)))
            print("not by prec  : {0:.4f}".format(tn/max(1,not_by_total_pred)))
    
    
    

        
    return(loss_sum)


##################################
    




def train_epoch(model,
                session,
                MP,
                train_data_start,
                train_data_size,
                train_file,
                train_spec_loc,
                val_spec_loc,
                model_path,
                best_val_loss,
                total_data_size
                ):
    
    if model_utils.data_type=="msp":
        with open(train_file) as filehandle:
            train_spec = dp.read_spectra_only_msp_prosit(filehandle,model_utils.data_type, 
                                                         train_spec_loc[train_data_start:train_data_start+train_data_size])


    if model_utils.data_type=="mgf":
        with open(train_file) as filehandle:
            train_spec = dp.read_spectra_only(filehandle,model_utils.data_type, 
                                              train_spec_loc[train_data_start:train_data_start+train_data_size])

    

    num_spectra_seen = 0

    train_loops = len(train_spec)//model_utils.train_stack_size
    print("num loops:",train_loops)
    i = 0
    
    
    while True:
              
        ##########################################################################
    #                   Val PHASE
        val_period = 10
        if i%val_period==0:
            val_data = list(
                              zip(*MP.map(dp.process_spectrum_Graph,
                                          get_random_stack(val_spec,stack_size=model_utils.val_stack_size)))
                              )
        
            print("")
            print("--val--")
            
            epoch = (model.global_step.eval()*model_utils.train_batch_size)/total_data_size
            
            print("epoch:{0}".format(epoch),file=log_file_handle)
            print("Global Step:{0}".format(model.global_step.eval()),file=log_file_handle)
 
            val_loss_sum = tf_run_data(model=M, 
                                        session=session, 
                                        tf_mode="test", 
                                        data=val_data, 
                                        log_file_handle=log_file_handle,
                                        evaluate=True,
                                        print_to_file=True,
                                        print_to_screen=True)
        
    
   
            epoch = round(epoch,3)
            if val_loss_sum < best_val_loss:
               
                print("-- saving model --" )
                print("prev best loss: {0:.6f}".format(best_val_loss))
                print("new best loss: {0:.6f}".format(val_loss_sum))
                
                model.saver.save(session,
                            model_path+str(epoch),
                            global_step=model.global_step,
                            write_meta_graph=False)
                
                best_val_loss = val_loss_sum
                
        ##########################################################################
    #                   TRAIN PHASE
                
        
        print(i)
        sys.stdout.flush()
        
        num_spectra_seen += model_utils.train_stack_size
        if num_spectra_seen > len(train_spec):
            break

        start_time = time.time()
        train_data = list(
                            zip(*MP.map(dp.process_spectrum_Graph,
                                        get_random_stack(train_spec,stack_size=model_utils.train_stack_size)))
                            )
        process_time = time.time() - start_time
        print("Data Process Time: {}".format(process_time))
        
        if(len(train_data)==0):
            break
        
        train_loss_sum = tf_run_data(model=M, 
                                     session=sess, 
                                        tf_mode="train", 
                                        data=train_data, 
                                        log_file_handle=log_file_handle, 
                                        evaluate=False,
                                        print_to_file=False,
                                        print_to_screen=False)
        
    
   
        
        
        
            
        if i%val_period==0:
            print("Train Loss:{0}".format(train_loss_sum/max(1,len(train_data[0]))),file=log_file_handle)
            print("Val Loss:{0}".format(val_loss_sum/max(1,len(val_data[0]))),file=log_file_handle)
            
            print("end")
            
        i+=1
        
        
    return best_val_loss


##################################
    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    
    M = Model.model(session=sess)

    save_model_path = os.path.dirname(log_file)+"/saved_model_"

    ckpt = tf.train.get_checkpoint_state(os.getcwd()+"/"+logging_dir)
    if ckpt:
        print("\n\nRestoring Model\n\n")
        M.saver.restore(sess,ckpt.model_checkpoint_path)
    else: 
        sess.run(tf.global_variables_initializer())
    
   
    best_val_loss = float("inf")
    
    loop_count=0
    
    while True:
        num_train_spec = len(model_utils.train_spec_loc)
        dataset_size = min(3000,num_train_spec)
        dataset_start = (loop_count%(num_train_spec//dataset_size))*dataset_size
        
        best_val_loss = train_epoch(model=M,
                                    session=sess,
                                    MP = multiprocessing_pool,
                                    train_data_start=dataset_start, 
                                    train_data_size=dataset_size, 
                                    train_file=model_utils.train_file, 
                                    train_spec_loc=model_utils.train_spec_loc, 
                                    val_spec_loc=model_utils.val_spec_loc,
                                    model_path=save_model_path,
                                    best_val_loss=best_val_loss,
                                    total_data_size=num_train_spec)
        
        
        loop_count+=1
        if loop_count>30:
            break
        
