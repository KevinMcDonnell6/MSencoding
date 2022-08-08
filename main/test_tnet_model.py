
import numpy as np
import tensorflow as tf
from sklearn import metrics
from multiprocessing import Pool
import data_processing as dp
import model_utils
import TnetModel as Model

import os
import gc

test_spec_loc = dp.inspect_file_location("mgf",model_utils.test_file)
with open(model_utils.test_file) as filehandle:
    test_spec = dp.read_spectra_only(filehandle,"mgf", test_spec_loc)

multiprocessing_pool = Pool(5)
model_utils.test_batch_size = 1
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as session:
    
    model = Model.model(session=session)

    path = os.getcwd()+"/graphs/"+model_utils.args.logging_dir
    save_model_path = os.path.dirname(path)+"/saved_model_"

    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt:
        print("\n\nRestoring Model\n\n")
        model.saver.restore(session,ckpt.model_checkpoint_path)
    else: 
        raise ValueError("No trained model in directory")       
            
      
    test_batch = []
    num_spectra_seen = 0
    
    for i in range(int(np.ceil(len(test_spec)/model_utils.test_stack_size))):
        test_data = list(
                          zip(*multiprocessing_pool.map(dp.process_spectrum_Tnet,
                                                        test_spec[(i*model_utils.test_stack_size):((i+1)*model_utils.test_stack_size)]))
                          )

        print(i,"/",int(np.ceil(len(test_spec)/model_utils.test_stack_size)))

        n_dev = len(test_data[0])
        dev_batch_size = model_utils.test_batch_size      
        
        test_peak_total = 0
        
        for start in range(0, n_dev, dev_batch_size):
            end = min(start+dev_batch_size, n_dev)
            
            indices = np.arange(start,end)
            batch_size = len(indices)
            
            b_or_y = np.array(test_data[0])[indices]
            by_mask = np.array(test_data[1])[indices]
            
            
            ion_loc = np.array(test_data[2])[indices]
            ion_loc_bw = np.array(test_data[3])[indices]
            
            peaks = np.array(test_data[4])[indices]
            
            features = np.array(test_data[5])[indices]
            
            ion_loc_self = np.array(test_data[6])[indices]
            
            scans = [test_data[7][idx] for idx in indices]
            
            logits, loss, _ = model.step(session=session,
                                          mode="test",
                                          targets=b_or_y,
                                          mask=by_mask,
                                          ion_locations=ion_loc,
                                          ion_locations_bw=ion_loc_bw,
                                          ion_locations_self=ion_loc_self,
                                          peaks=peaks,
                                          features=features)

            test_batch.append([b_or_y,by_mask,logits,scans])

            
            test_peak_total += np.sum(by_mask)
            
            
        del test_data
        gc.collect()


all_by = np.concatenate([i[0] for i in test_batch])
all_by_mask = np.concatenate([i[1] for i in test_batch])
all_logits = np.concatenate([i[2] for i in test_batch])
all_probs = dp.sigmoid(all_logits)
all_scans = np.concatenate([i[3] for i in test_batch])

new_dir="graphs/"+model_utils.args.logging_dir+"/"+model_utils.test_file.split("/")[-1]
os.mkdir(new_dir)
np.save(new_dir+"/all_by",all_by)
np.save(new_dir+"/all_by_mask",all_by_mask)
np.save(new_dir+"/all_logits",all_logits)
np.save(new_dir+"/all_probs",all_probs)
np.save(new_dir+"/all_scans",all_scans)



fpr, tpr, _ = metrics.roc_curve(all_by.flatten(), 
                                all_logits.flatten(), 
                                sample_weight=all_by_mask.flatten())


print("AUC:"+str(metrics.auc(fpr,tpr)))

print("Avg Precision:"+str(metrics.average_precision_score(all_by.flatten(), 
                                                            all_logits.flatten(), 
                                                            sample_weight=all_by_mask.flatten())))
