
import numpy as np
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


class neigh_agg():
    
    def __init__(self,input_dim,output_dim,v,h_dim,name,aggtype = "mean"):
        
        if aggtype not in ["sum","mean"]:
            raise ValueError("Aggregation type must be 'mean' or 'sum'")
        
        self.v = v
        self.h_dim = h_dim
        self.input_dim = input_dim
        self.aggtype=aggtype
        
        with tf.variable_scope(name):
            self.neigh_weights = tf.get_variable(name=name+"/neigh_weights",
                                                shape=[input_dim,output_dim]
                                                )
            
            self.self_weights = tf.get_variable(name=name+"/self_weights",
                                                shape=[input_dim,output_dim]
                                                )
            
            self.bias = tf.get_variable(name=name+"bias",
                                        shape=2*output_dim,
                                        initializer=tf.zeros_initializer)
        
    def __call__(self,node_encodings,adj_matrices,reuse=False,include_Adj_weights=True):
        
        h = node_encodings
        batch_size = tf.shape(h)[0]
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        Adj = tf.reduce_sum(adj_matrices,axis=0)
        
        if not include_Adj_weights:
            Adj = tf.where(tf.greater(Adj,0), tf.ones_like(Adj), tf.zeros_like(Adj))
        
        neigh_sum = tf.matmul(Adj,h)
        if self.aggtype =="sum":
            neigh_mean = neigh_sum
        elif self.aggtype == "mean":
            neigh_mean = tf.div_no_nan(neigh_sum,tf.tile(tf.expand_dims(tf.cast(tf.count_nonzero(Adj,axis=-1),tf.float32),-1),(1,1,self.input_dim)))
        
        batch_neigh_weights = tf.tile(tf.expand_dims(self.neigh_weights,axis=0),(batch_size,1,1))
        ag_neighbours = tf.matmul(neigh_mean,batch_neigh_weights) 
        
        batch_self_weights = tf.tile(tf.expand_dims(self.self_weights,axis=0),(batch_size,1,1))
        ag_self = tf.matmul(h,batch_self_weights)
        
        output = tf.concat([ag_self,ag_neighbours],axis=-1)
        output = tf.nn.bias_add(output,self.bias)
        
        return tf.nn.relu(output)
    
class CNN(object):
    
    def __init__(self):
        self.activation = tf.nn.relu
        self.hidden_layer_size = model_utils.hidden_layer_size
        
        self.cnn1 = tf.layers.Conv2D(filters=64, 
                                    kernel_size=[1,3],
                                    strides=(1,1),
                                    padding="SAME",
                                    activation=tf.nn.relu,
                                    use_bias=True)
        self.cnn2 = tf.layers.Conv2D(filters=64, 
                                    kernel_size=[1,2],
                                    strides=(1,1),
                                    padding="SAME",
                                    activation=tf.nn.relu,
                                    use_bias=True)
        
        self.dense1 = tf.layers.Dense(self.hidden_layer_size,
                                         activation=self.activation,
                                         use_bias=True)
        
        self.dense2 = tf.layers.Dense(self.hidden_layer_size,
                                         activation=self.activation,
                                         use_bias=True)
        
    def __call__(self,aa_windows,shape):
        
        aa_windows = tf.reshape(aa_windows,
                                shape)
 
        aa_windows = tf.transpose(aa_windows,(0,2,3,1))
        
        
        window_encodings = self.cnn1(aa_windows)
        window_encodings = self.cnn2(window_encodings)
        window_encodings = tf.nn.max_pool(window_encodings,
                                          ksize=[1,1,3,1],
                                          strides=[1,1,2,1],
                                          padding="SAME")
        
        dense_input_size = shape[-2] * (model_utils.WINDOW_SIZE // 2) * 64
        
        window_encodings = tf.reshape(window_encodings,[-1,dense_input_size])
        window_encodings = self.dense1(window_encodings)
        window_encodings = tf.layers.batch_normalization(window_encodings)
        window_encodings = self.dense2(window_encodings)
        window_encodings = tf.layers.batch_normalization(window_encodings)
        
        return window_encodings
    
class model():
    
    
    def __init__(self,session=None,mode="train",direction="f"):
      
        self.mode=mode
      
        self.global_step = tf.Variable(0,trainable=False)
        
        self.num_edge_types = model_utils.vocab_size-3
        
        self.dropout = tf.placeholder(tf.float32,
                                      None,
                                      name="dropout")
        
        ##################### Data Placeholders

        self.by_input = tf.placeholder(dtype=tf.float32,
                                       shape=[None,
                                              model_utils.max_num_peaks])

        self.input_mask = tf.placeholder(dtype=tf.float32,
                                       shape=[None,
                                              model_utils.max_num_peaks])
        
        self.AA_windows = tf.placeholder(dtype=tf.float32,
                                        shape=[None,
                                               model_utils.max_num_peaks,
                                               model_utils.vocab_size,
                                               model_utils.num_ion,
                                               model_utils.WINDOW_SIZE],
                                        name="intensity_window")
        
        self.AA_windows_bw = tf.placeholder(dtype=tf.float32,
                                        shape=[None,
                                               model_utils.max_num_peaks,
                                               model_utils.vocab_size,
                                               model_utils.num_ion,
                                               model_utils.WINDOW_SIZE],
                                        name="intensity_window_bw")
        
        self.features = tf.placeholder(dtype=tf.float32,
                                       shape=[None,model_utils.max_num_peaks,model_utils.num_features],
                                       name="added_features")
        
        self.ion_windows = tf.placeholder(dtype=tf.float32,
                                       shape=[None,model_utils.max_num_peaks,model_utils.num_ion-1,10],
                                       name="ion_windows")
        
        self.adjacencies = tf.placeholder(dtype=tf.float32,
                                            shape=[None,None,None,model_utils.vocab_size-3],
                                            name="adj_matrices")
        
        
        
        self.JoinDense = tf.layers.Dense(model_utils.hidden_layer_size,
                                          use_bias=True,
                                          name="JoinDense")
        
        
        
        
        ########  Tensors
        
        self.zero = tf.constant(0)
        
        self.OutputLayer = tf.layers.Dense(1,
                                            activation=None,
                                            use_bias=False,
                                            name="OutputLayer")
        
        self.output = self.train()
        
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
    
    def encode(self):
        b = tf.shape(self.adjacencies)[0]
        v = tf.shape(self.adjacencies)[1]
        h_dim = model_utils.hidden_layer_size
        
        CNN1 = CNN()
        CNN2 = CNN()
        CNN3 = CNN()
        
        feature_encodings = self.Feature_Dense()

        window_encodings = tf.concat((CNN1(self.AA_windows,
                                           [-1, model_utils.vocab_size,
                                                model_utils.num_ion,
                                                model_utils.WINDOW_SIZE]),
                                      CNN2(self.AA_windows_bw,
                                           [-1, model_utils.vocab_size,
                                                model_utils.num_ion,
                                                model_utils.WINDOW_SIZE]),
                                      CNN3(self.ion_windows,
                                           [-1, 1,
                                                model_utils.num_ion-1,
                                                model_utils.WINDOW_SIZE]),
                                      feature_encodings
                                      ),-1)
        

        window_encodings = self.JoinDense(window_encodings)
        window_encodings = tf.reshape(window_encodings,
                                      [-1,model_utils.max_num_peaks,
                                          model_utils.hidden_layer_size],
                                      name="window_emb_reshape_join")
        
        
        initial_encodings_fw = window_encodings
        initial_encodings_bw = window_encodings        
        
        
         #####################################

        
        h_fw = initial_encodings_fw
        h_bw = initial_encodings_bw

        adjacency_matrices_fw = tf.transpose(self.adjacencies,perm=[3,0,1,2])
        adjacency_matrices_bw = tf.transpose(self.adjacencies,perm=[3,0,2,1])
        
        self.fw_aggregators = []
        self.bw_aggregators = []
        
        for i in range(model_utils.aggregate_path_length):
       

             
            if i==0:
                mul_factor=1
            else:
                mul_factor=2
                
            fw_agg = neigh_agg(input_dim=mul_factor*h_dim, 
                                    output_dim=h_dim, 
                                    v=v, 
                                    h_dim=h_dim, 
                                    name="fw_agg_{0}".format(i))
            
            bw_agg = neigh_agg(input_dim=mul_factor*h_dim, 
                                    output_dim=h_dim, 
                                    v=v, 
                                    h_dim=h_dim, 
                                    name="bw_agg_{0}".format(i))
            
            h_fw = fw_agg(node_encodings=h_fw,
                          adj_matrices=adjacency_matrices_fw,
                          include_Adj_weights=False)
        
            h_bw = bw_agg(node_encodings=h_bw,
                          adj_matrices=adjacency_matrices_bw,
                          include_Adj_weights=False)
        
        fw_hidden = h_fw
        bw_hidden = h_bw
        encoded_states = tf.nn.relu(
                                tf.concat([fw_hidden,
                                           bw_hidden], 
                                          axis=-1))
        
        graph_encoding = tf.reduce_max(encoded_states,axis=1) 

        graph_LSTM_states = tf.nn.rnn_cell.LSTMStateTuple(c=graph_encoding,h=graph_encoding)
        
        return encoded_states, graph_LSTM_states


    def train(self):
        
        
        encoded_outputs, encoder_states = self.encode()
        
        batch_size, num_nodes, _, _aa = tf.unstack(tf.shape(self.adjacencies))
        
        
        
        
        logits = self.OutputLayer(encoded_outputs)
        logits = tf.minimum(logits, 80)
        self.train_logits=logits
        
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
             adj_matrix,
             AA_windows_fw=None,
             AA_windows_bw=None,
             ion_windows=None,
             features=None):
             
             
        
        
        input_feed = {}
        
        input_feed[self.by_input.name] = targets
        input_feed[self.input_mask.name] = mask
        
        input_feed[self.AA_windows.name] = AA_windows_fw
        input_feed[self.AA_windows_bw.name] = AA_windows_bw
        input_feed[self.ion_windows.name] = ion_windows
        input_feed[self.features.name] = features
        
        input_feed[self.adjacencies.name] = adj_matrix

        
        ops = []
        

        if mode =="train":
            ops.append(self.train_logits)
            ops.append(self.loss)
            ops.append(self.train_op)

        if mode =="test":
            ops.append(self.train_logits)
            ops.append(self.loss)
            ops.append(self.zero)
            
        outputs = session.run(ops,input_feed)
        
        return outputs
        











