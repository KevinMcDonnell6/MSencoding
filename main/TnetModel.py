
import tensorflow as tf
import model_utils

def FL(logits,labels,gamma=0,alpha=0.9,pos_weight=None):
   
    p =tf.sigmoid(logits)
    
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    
    modulation_pos = alpha*(tf.math.pow((1 - p + 1e-8), gamma,name="powerpos"))
    modulation_neg = (1-alpha)* (tf.math.pow(p+1e-8, gamma,name="powerneg"))
    mask = tf.dtypes.cast(labels, dtype=tf.bool)
    modulation = tf.where(mask, modulation_pos, modulation_neg)
    return modulation*loss

  

    
class T_net(object):
    
    def __init__(self):
        
        self.conv1 = tf.layers.Conv1D(filters=64, 
                                    kernel_size=1,
                                    activation=tf.nn.relu,
                                    use_bias=True)
        
        self.conv2 = tf.layers.Conv1D(filters=2*64, 
                                    kernel_size=1,
                                    activation=tf.nn.relu,
                                    use_bias=True)
        
        self.conv3 = tf.layers.Conv1D(filters=4*64, 
                                    kernel_size=1,
                                    activation=tf.nn.relu,
                                    use_bias=True)
        
        self.fc1 = tf.layers.Dense(model_utils.hidden_layer_size,activation=tf.nn.relu)
        self.fc2 = tf.layers.Dense(model_utils.hidden_layer_size,activation=tf.nn.relu)
        
        self.bn_input = tf.layers.BatchNormalization()
        
        self.bn1 = tf.layers.BatchNormalization()
        self.bn2 = tf.layers.BatchNormalization()
        self.bn3 = tf.layers.BatchNormalization()
        self.bn4 = tf.layers.BatchNormalization()
        self.bn5 = tf.layers.BatchNormalization()
        
    def __call__(self,x):
        
        x = self.bn_input(x)
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        
        x = tf.reduce_max(x,2)
        x = self.bn4(self.fc1(x))
        x = self.bn5(self.fc2(x))

        return x
        
        
class model(object):
    
    def __init__(self,
                 session,
                 mode="train",
                 inference=False):
        self.inference = inference
        self.mode= mode
        self.global_step = tf.Variable(0,trainable=False)
        
        
        ##################### Data Placeholders

        self.by_input = tf.placeholder(dtype=tf.float32,
                                       shape=[None,
                                              model_utils.max_num_peaks])
        
        self.input_mask = tf.placeholder(dtype=tf.float32,
                                       shape=[None,
                                              model_utils.max_num_peaks])
        
        self.ion_locations = tf.placeholder(dtype=tf.float32,
                                       shape=[None,
                                              model_utils.max_num_peaks,
                                              model_utils.vocab_size,
                                              model_utils.num_ion_tnet],
                                       name="ion_locations")
        
        self.ion_locations_bw = tf.placeholder(dtype=tf.float32,
                                       shape=[None,
                                              model_utils.max_num_peaks,
                                              model_utils.vocab_size,
                                              model_utils.num_ion_tnet],
                                       name="ion_locations_bw")
        
        self.ion_locations_self = tf.placeholder(dtype=tf.float32,
                                       shape=[None,
                                              model_utils.max_num_peaks,
                                              model_utils.num_ion-1],
                                       name="ion_locations_self")
            
        self.peaks = tf.placeholder(dtype=tf.float32,
                                    shape=[None,
                                           model_utils.max_num_peaks,
                                           2],
                                    name="peaks")
        
        self.features = tf.placeholder(dtype=tf.float32,
                                       shape=[None,model_utils.max_num_peaks,model_utils.num_features],
                                       name="added_features")
        
       
        
        
        
        
        ########  Tensors
        
        self.zero = tf.constant(0)
            
        
        self.JoinDense = tf.layers.Dense(model_utils.hidden_layer_size,
                                         use_bias=True,
                                         name="JoinDense")
        
        self.OutputLayer = tf.layers.Dense(1,
                                           activation=None,
                                           use_bias=False,
                                           name="OutputLayer")

        
        
        
        self.train()
      
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1,save_relative_paths=True)
           
    
        
    
    def Feature_Dense(self):
        
        
        
        all_features = tf.reshape(self.features,[-1,model_utils.num_features])
        feature_encoding = tf.layers.dense(all_features,
                                           units=model_utils.hidden_layer_size,
                                           activation=tf.nn.relu)
        
        feature_encoding = tf.layers.dense(feature_encoding,
                                           units=model_utils.hidden_layer_size,
                                           activation=tf.nn.relu)
        
        
        return feature_encoding
    
       
    def PointNet(self,Tnet,ion_loc):
        
        N=model_utils.max_num_peaks
        vocab_size = model_utils.vocab_size
        num_ion = model_utils.num_ion_tnet
        
        peak_mz = self.peaks[...,0]
        peak_int = self.peaks[...,1]
        
        peak_mz = peak_mz[:,tf.newaxis,:,tf.newaxis]
        peak_int = peak_int[:,tf.newaxis,:,tf.newaxis]
        peak_mz = tf.tile(peak_mz,(1,N,1,1)) 
        peak_mz_mask = tf.cast(tf.greater(peak_mz,1e-5),tf.float32)
        peak_int = tf.tile(peak_int,(1,N,1,1)) 
        ion_loc = tf.reshape(ion_loc,
                             shape=(-1,N,1,vocab_size*num_ion))
        ion_loc_mask = tf.cast(tf.greater(ion_loc,1e-5),tf.float32)
        
        sigma_D = tf.exp(
                        -tf.abs(
                            (peak_mz-ion_loc)*100 
                            )
                        )
        sigma_D = sigma_D*peak_mz_mask*ion_loc_mask
        feature_vec = tf.concat(values=(sigma_D,peak_int),axis=3)
        feature_vec = tf.reshape(feature_vec, (-1,N,(vocab_size*num_ion) +1))
        feature_vec = tf.transpose(feature_vec,(0,2,1))
        feature_vec = Tnet(feature_vec)
        return feature_vec
    
    def PointNet_self(self,Tnet,ion_loc):
        
        N=model_utils.max_num_peaks
        vocab_size = model_utils.vocab_size
        num_ion = model_utils.num_ion-1
        
        peak_mz = self.peaks[...,0]
        peak_int = self.peaks[...,1]
        
        peak_mz = peak_mz[:,tf.newaxis,:,tf.newaxis]
        peak_int = peak_int[:,tf.newaxis,:,tf.newaxis]
        peak_mz = tf.tile(peak_mz,(1,N,1,1))
        peak_mz_mask = tf.cast(tf.greater(peak_mz,1e-5),tf.float32)
        peak_int = tf.tile(peak_int,(1,N,1,1))
        ion_loc = tf.reshape(ion_loc,
                             shape=(-1,N,1,num_ion))
        ion_loc_mask = tf.cast(tf.greater(ion_loc,1e-5),tf.float32)
        
        sigma_D = tf.exp(
                        -tf.abs(
                            (peak_mz-ion_loc)*100
                            )
                        )
        sigma_D = sigma_D*peak_mz_mask*ion_loc_mask
        feature_vec = tf.concat(values=(sigma_D,peak_int),axis=3)
        feature_vec = tf.reshape(feature_vec, (-1,N,num_ion +1))
        feature_vec = tf.transpose(feature_vec,(0,2,1))
        feature_vec = Tnet(feature_vec)
        return feature_vec
        
    def train(self):
        
        batch_size = tf.shape(self.input_mask)[0]
        
       
        
        self.Tnet1 = T_net()
        self.Tnet2 = T_net()
        self.Tnet3 = T_net()
        
        window_encodings = self.PointNet(self.Tnet1,
                                         self.ion_locations)
        window_encodings_bw = self.PointNet(self.Tnet2,
                                            self.ion_locations_bw)
        window_encodings_self = self.PointNet_self(self.Tnet3,
                                                   self.ion_locations_self)
        feature_encodings = self.Feature_Dense()
        
        window_encodings = tf.concat((window_encodings,
                                      window_encodings_bw,
                                      window_encodings_self,
                                        feature_encodings
                                      ),-1)
        window_encodings = self.JoinDense(window_encodings)
        window_encodings = tf.reshape(window_encodings,
                                      [-1,model_utils.max_num_peaks,
                                          model_utils.hidden_layer_size],
                                      name="window_emb_reshape_join")
        
        logits = self.OutputLayer(window_encodings)
        logits = tf.minimum(logits, 80)
        self.logits = logits
        
        
        if self.mode=="train":
            
            labels = tf.expand_dims(self.by_input, -1) 
            losses = FL(logits=logits,
                                labels=labels,
                                gamma=2,
                                alpha=0.9)
            
            batch_loss = tf.reduce_sum(tf.squeeze(losses)*self.input_mask)/tf.cast(batch_size,tf.float32)
             
            
            train_loss = batch_loss
            
            self.loss = train_loss
            
            params = tf.trainable_variables()
            gradients = tf.gradients(train_loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, model_utils.max_global_norm)
            self.losses = clipped_gradients
            optimizer = tf.train.AdamOptimizer(learning_rate=model_utils.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(clipped_gradients, params),global_step=self.global_step)
        
        
    def step(self,
             session, 
             mode,
             targets,
             mask,
             ion_locations,
             ion_locations_bw,
             ion_locations_self,
             peaks,
             features):
        
        input_feed = {}
        
        input_feed[self.by_input.name] = targets
        input_feed[self.input_mask.name] = mask
        
        input_feed[self.ion_locations.name] = ion_locations
        input_feed[self.ion_locations_bw.name] = ion_locations_bw
        input_feed[self.ion_locations_self.name] = ion_locations_self
        input_feed[self.peaks.name] = peaks
        input_feed[self.features.name] = features
        
        
        ops = []
        
        if mode =="train":
            ops.append(self.logits)
            ops.append(self.loss)
            ops.append(self.train_op)
            
        if mode =="test":
            ops.append(self.logits)
            ops.append(self.loss)
            ops.append(self.zero)
            
      
        outputs = session.run(ops,input_feed)
        
        return outputs