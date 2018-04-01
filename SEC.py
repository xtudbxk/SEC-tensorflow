import os
import sys
import time
import numpy as np
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

from dataset import dataset
from crf import crf_inference

class SEC():
    def __init__(self,config):
        self.config = config
        self.h,self.w = self.config.get("input_size",(321,321))
        self.category_num = self.config.get("category_num",21)
        self.accum_num = self.config.get("accum_num",1)
        self.data = self.config.get("data",None)
        self.min_prob = self.config.get("min_prob",0.0001)

        self.net = {}
        self.loss = {}
        self.saver = {}

        self.weights = {}
        self.stride = {}
        self.stride["input"] = 1
        self.trainable_list = []
        # different lr for different variable
        self.lr_1_list = []
        self.lr_2_list = []
        self.lr_10_list = []
        self.lr_20_list = []

    def build(self):
        if "output" not in self.net:
            with tf.name_scope("placeholder"):
                self.net["input"] = tf.placeholder(tf.float32,[None,self.h,self.w,self.config.get("input_channel",3)])
                self.net["label"] = tf.placeholder(tf.int32,[None,self.category_num])
                self.net["cues"] = tf.placeholder(tf.float32,[None,41,41,self.category_num])
                self.net["gt"] = tf.placeholder(tf.int32,[None,self.h,self.w,1])
                self.net["drop_prob"] = tf.placeholder(tf.float32)

            self.net["output"] = self.create_network()

        return self.net["output"]

    def create_network(self):
        if "init_model_path" in self.config:
            self.load_init_model()
        with tf.name_scope("deeplab") as scope:
            block = self.build_block("input",["conv1_1","relu1_1","conv1_2","relu1_2","pool1"])
            block = self.build_block(block,["conv2_1","relu2_1","conv2_2","relu2_2","pool2"])
            block = self.build_block(block,["conv3_1","relu3_1","conv3_2","relu3_2","conv3_3","relu3_3","pool3"])
            block = self.build_block(block,["conv4_1","relu4_1","conv4_2","relu4_2","conv4_3","relu4_3","pool4"])
            block = self.build_block(block,["conv5_1","relu5_1","conv5_2","relu5_2","conv5_3","relu5_3","pool5","pool5a"])
            fc = self.build_fc(block,["fc6","relu6","drop6","fc7","relu7","drop7","fc8"])

        with tf.name_scope("sec") as scope:
            softmax = self.build_sp_softmax(fc)
            crf = self.build_crf(fc,"input")

        return self.net[crf]

    def build_block(self,last_layer,layer_lists):
        for layer in layer_lists:
            if layer.startswith("conv"):
                if layer[4] != "5":
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = self.stride[last_layer]
                        weights,bias = self.get_weights_and_bias(layer)
                        self.net[layer] = tf.nn.conv2d( self.net[last_layer], weights, strides = [1,1,1,1], padding="SAME", name="conv")
                        self.net[layer] = tf.nn.bias_add( self.net[layer], bias, name="bias")
                        last_layer = layer
                if layer[4] == "5":
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = self.stride[last_layer]
                        weights,bias = self.get_weights_and_bias(layer)
                        self.net[layer] = tf.nn.atrous_conv2d( self.net[last_layer], weights, rate=2, padding="SAME", name="conv")
                        self.net[layer] = tf.nn.bias_add( self.net[layer], bias, name="bias")
                        last_layer = layer
            if layer.startswith("batch_norm"):
                with tf.name_scope(layer) as scope:
                    self.stride[layer] = self.stride[last_layer]
                    self.net[layer] = tf.contrib.layers.batch_norm(self.net[last_layer])
                    last_layer = layer
            if layer.startswith("relu"):
                with tf.name_scope(layer) as scope:
                    self.stride[layer] = self.stride[last_layer]
                    self.net[layer] = tf.nn.relu( self.net[last_layer],name="relu")
                    last_layer = layer
            elif layer.startswith("pool5a"):
                with tf.name_scope(layer) as scope:
                    self.stride[layer] = self.stride[last_layer]
                    self.net[layer] = tf.nn.avg_pool( self.net[last_layer], ksize=[1,3,3,1], strides=[1,1,1,1],padding="SAME",name="pool")
                    last_layer = layer
            elif layer.startswith("pool"):
                if layer[4] not in ["4","5"]:
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = 2 * self.stride[last_layer]
                        self.net[layer] = tf.nn.max_pool( self.net[last_layer], ksize=[1,3,3,1], strides=[1,2,2,1],padding="SAME",name="pool")
                        last_layer = layer
                if layer[4] in ["4","5"]:
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = self.stride[last_layer]
                        self.net[layer] = tf.nn.max_pool( self.net[last_layer], ksize=[1,3,3,1], strides=[1,1,1,1],padding="SAME",name="pool")
                        last_layer = layer
        return last_layer

    def build_fc(self,last_layer, layer_lists):
        for layer in layer_lists:
            if layer.startswith("fc"):
                with tf.name_scope(layer) as scope:
                    weights,bias = self.get_weights_and_bias(layer)
                    if layer.startswith("fc6"):
                        self.net[layer] = tf.nn.atrous_conv2d( self.net[last_layer], weights, rate=12, padding="SAME", name="conv")

                    else:
                        self.net[layer] = tf.nn.conv2d( self.net[last_layer], weights, strides = [1,1,1,1], padding="SAME", name="conv")
                    self.net[layer] = tf.nn.bias_add( self.net[layer], bias, name="bias")
                    last_layer = layer
            if layer.startswith("batch_norm"):
                with tf.name_scope(layer) as scope:
                    self.net[layer] = tf.contrib.layers.batch_norm(self.net[last_layer])
                    last_layer = layer
            if layer.startswith("relu"):
                with tf.name_scope(layer) as scope:
                    self.net[layer] = tf.nn.relu( self.net[last_layer])
                    last_layer = layer
            if layer.startswith("drop"):
                with tf.name_scope(layer) as scope:
                    self.net[layer] = tf.nn.dropout( self.net[last_layer],self.net["drop_prob"])
                    last_layer = layer

        return last_layer

    def build_sp_softmax(self,last_layer):
        layer = "fc8-softmax"
        preds_max = tf.reduce_max(self.net[last_layer],axis=3,keepdims=True)
        preds_exp = tf.exp(self.net[last_layer] - preds_max)
        self.net[layer] = preds_exp / tf.reduce_sum(preds_exp,axis=3,keepdims=True) + self.min_prob
        self.net[layer] = self.net[layer] / tf.reduce_sum(self.net[layer],axis=3,keepdims=True)
        return layer


    def build_crf(self,featemap_layer,img_layer):
        origin_image = self.net(img_layer) + self.data.img_mean
        origin_image_zoomed = tf.image.resize_bilinear(origin_image,(41,41))
        featemap = self.net[featemap_layer]
        def crf(featemap,image):
            crf_config = {"g_sxy":3/12,"g_compat":3,"bi_sxy":80/12,"bi_srgb":13,"bi_compat":10,"iterations":5}
            batch_size = featemap.shape[0]
            image = image.astype(np.uint8)
            ret = np.zeros(featemap.shape,dtype=np.float32)
            for i in range(batch_size):
                ret[i,:,:,:] = crf_inference(featemap[i],image[i],crf_config,self.category_num)

            ret[ret < self.min_prob] = self.min_prob
            ret /= np.sum(ret,axis=3,keepdims=True)
            ret = np.log(ret)
            return ret.astype(np.float32)

        layer = "crf"
        self.net[layer] = tf.py_func(crf,[featemap,origin_image_zoomed],tf.float32) # shape [N, h, w, C], RGB or BGR doesn't matter for crf
        return layer

    def load_init_model(self):
        model_path = self.config["init_model_path"]
        self.init_model = np.load(model_path,encoding="latin1").item()
        print("load init model success: %s" % model_path)
		
    def restore_from_model(self,saver,model_path,checkpoint=False):
        assert self.sess is not None
        if checkpoint is True:
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            saver.restore(self.sess, model_path)
			
    def get_weights_and_bias(self,layer):
        if layer.startswith("conv"):
            shape = [3,3,0,0]
            if layer == "conv1_1":
                shape[2] = 3
            else:
                shape[2] = 64 * self.stride[layer]
                if shape[2] > 512: shape[2] = 512
                if layer in ["conv2_1","conv3_1","conv4_1"]: shape[2] = int(shape[2]/2)
            shape[3] = 64 * self.stride[layer]
            if shape[3] > 512: shape[3] = 512
        if layer.startswith("fc"):
            if layer == "fc6":
                shape = [3,3,512,1024]
            if layer == "fc7":
                shape = [1,1,1024,1024]
            if layer == "fc8": 
                shape = [1,1,1024,self.category_num]
        if "init_model_path" not in self.config:
            init = tf.random_normal_initializer(stddev=0.01)
            weights = tf.get_variable(name="%s_weights" % layer,initializer=init, shape = shape)
            init = tf.constant_initializer(0)
            bias = tf.get_variable(name="%s_bias" % layer,initializer=init, shape = [shape[-1]])
        else: # restroe from init.npy
            if layer == "fc8": # using random initializer for the last layer
                init = tf.contrib.layers.xavier_initializer(uniform=True)
            else:
                init = tf.constant_initializer(self.init_model[layer]["w"])
            weights = tf.get_variable(name="%s_weights" % layer,initializer=init,shape = shape)
            if layer == "fc8":
                init = tf.constant_initializer(0)
            else:
                init = tf.constant_initializer(self.init_model[layer]["b"])
            bias = tf.get_variable(name="%s_bias" % layer,initializer=init,shape = [shape[-1]])
        self.weights[layer] = (weights,bias)
        if layer != "fc8":
            self.lr_1_list.append(weights)
            self.lr_2_list.append(bias)
        else: # the lr is larger in the last layer
            self.lr_10_list.append(weights)
            self.lr_20_list.append(bias)
        self.trainable_list.append(weights)
        self.trainable_list.append(bias)

        return weights,bias

    def getloss(self):
        seed_loss = self.get_seed_loss(self.net["fc8-softmax"],self.net["cues"])
        expand_loss = self.get_expand_loss(self.net["fc8-softmax"],self.net["label"])
        constrain_loss = self.get_constrain_loss(self.net["fc8-softmax"],self.net["crf"])
        self.loss["seed"] = seed_loss
        self.loss["expand"] = expand_loss
        self.loss["constrain"] = constrain_loss

        loss = seed_loss + expand_loss + constrain_loss

        return loss

    def get_seed_loss(self,softmax,cues):
        count = tf.reduce_sum(cues,axis=(1,2,3),keepdims=True)
        loss = -tf.reduce_mean(tf.reduce_sum( cues*tf.log(softmax), axis=(1,2,3), keepdims=True)/count)
        return loss

    def get_expand_loss(self,softmax,labels):
        stat = labels[:,1:]
        probs_bg = softmax[:,:,:,0]
        probs = softmax[:,:,:,1:]
        probs_max = tf.reduce_max(probs,axis=(1,2))

        q_fg = 0.996
        probs_sort = tf.contrib.framework.sort( tf.reshape(probs,(-1,41*41,20)), axis=1)
        weights = np.array([ q_fg ** i for i in range(41*41 -1, -1, -1)])
        weights = np.reshape(weights,(1,-1,1))
        Z_fg = np.sum(weights)
        probs_mean = tf.reduce_sum((probs_sort*weights)/Z_fg, axis=1)

        q_bg = 0.999
        probs_bg_sort = tf.contrib.framework.sort( tf.reshape(probs_bg,(-1,41*41)), axis=1)
        weights_bg = np.array([ q_bg ** i for i in range(41*41 -1, -1, -1)])
        weights_bg = np.reshape(weights_bg,(1,-1))
        Z_bg = np.sum(weights_bg)
        probs_bg_mean = tf.reduce_sum((probs_bg_sort*weights_bg)/Z_bg, axis=1)

        stat_2d = tf.greater( stat, 0)
        stat_2d = tf.cast(stat_2d,tf.float32)
        
        self.loss_1 = -tf.reduce_mean( tf.reduce_sum( ( stat_2d*tf.log(probs_mean) / tf.reduce_sum(stat_2d,axis=1,keepdims=True)), axis=1))
        self.loss_2 = -tf.reduce_mean( tf.reduce_sum( ( (1-stat_2d)*tf.log(1-probs_max) / tf.reduce_sum((1-stat_2d),axis=1,keepdims=True)), axis=1))
        self.loss_3 = -tf.reduce_mean( tf.log(probs_bg_mean) )

        loss = self.loss_1 + self.loss_2 + self.loss_3
        return loss

    def get_constrain_loss(self,softmax,crf):
        probs_smooth = tf.exp(crf)
        loss = tf.reduce_mean(tf.reduce_sum(probs_smooth * tf.log(probs_smooth/softmax), axis=3))
        return loss

    def optimize(self,base_lr,momentum):
        self.net["lr"] = tf.Variable(base_lr, trainable=False, dtype=tf.float32)
        opt = tf.train.MomentumOptimizer(self.net["lr"],momentum)
        gradients = opt.compute_gradients(self.loss["total"],var_list=self.trainable_list)
        self.grad = {}
        self.net["accum_gradient"] = []
        self.net["accum_gradient_accum"] = []
        new_gradients = []
        for (g,v) in gradients:
            if v in self.lr_2_list:
                g = 2*g
            if v in self.lr_10_list:
                g = 10*g
            if v in self.lr_20_list:
                g = 20*g
            self.net["accum_gradient"].append(tf.Variable(tf.zeros_like(g),trainable=False))
            self.net["accum_gradient_accum"].append(self.net["accum_gradient"][-1].assign_add( g/self.accum_num, use_locking=True))
            new_gradients.append((self.net["accum_gradient"][-1],v))

        self.net["accum_gradient_clean"] = [g.assign(tf.zeros_like(g)) for g in self.net["accum_gradient"]]
        self.net["accum_gradient_update"]  = opt.apply_gradients(new_gradients)

    def train(self,base_lr,weight_decay,momentum,batch_size,epoches):
        gpu_options = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.30))
        self.sess = tf.Session(config=gpu_options)
        x,gt,y,c,id_of_image,iterator_train = self.data.next_batch(category="train",batch_size=batch_size,epoches=-1)
        self.build()
		self.optimize(base_lr,momentum)
        self.saver["norm"] = tf.train.Saver(max_to_keep=2,var_list=self.trainable_list)
        self.saver["lr"] = tf.train.Saver(var_list=self.trainable_list)
        self.saver["best"] = tf.train.Saver(var_list=self.trainable_list,max_to_keep=2)

        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            self.sess.run(iterator_train.initializer)

            if self.config.get("model_path",False) is not False:
                print("start to load model: %s" % self.config.get("model_path"))
                print("before l2 loss:%f" % self.sess.run(self.loss["l2"]))
                self.restore_from_model(self.saver["norm"],self.config.get("model_path"),checkpoint=False)
                print("model loaded ...")
                print("after l2 loss:%f" % self.sess.run(self.loss["l2"]))

            start_time = time.time()
            print("start_time: %f" % start_time)
            print("config -- lr:%f weight_decay:%f momentum:%f batch_size:%f epoches:%f" % (base_lr,weight_decay,momentum,batch_size,epoches))

            epoch,i = 0.0,0
            iterations_per_epoch_train = self.data.get_data_len() // batch_size
            while epoch < epoches:
                if i == 0: # in case for restoring
                    self.sess.run(tf.assign(self.net["lr"],base_lr))
                if i == 10*iterations_per_epoch_train:
                    new_lr = 1e-4
                    print("save model before new_lr:%f" % new_lr)
                    self.saver["lr"].save(self.sess,os.path.join(self.config.get("saver_path","saver"),"lr-%f" % base_lr),global_step=i)
                    self.sess.run(tf.assign(self.net["lr"],new_lr))
                    base_lr = new_lr
                if i == 20*iterations_per_epoch_train:
                    new_lr = 1e-5
                    print("save model before new_lr:%f" % new_lr)
                    self.saver["lr"].save(self.sess,os.path.join(self.config.get("saver_path","saver"),"lr-%f" % base_lr),global_step=i)
                    self.sess.run(tf.assign(self.net["lr"],new_lr))
                    base_lr = new_lr
                data_x,data_gt,data_y,data_c,data_id_of_image = self.sess.run([x,gt,y,c,id_of_image])
                params = {self.net["input"]:data_x,self.net["gt"]:data_gt,self.net["label"]:data_y,self.net["cues"]:data_c,self.net["drop_prob"]:0.5}
                self.sess.run(self.net["accum_gradient_accum"],feed_dict=params)
                if i % self.accum_num == self.accum_num - 1:
                    _ = self.sess.run(self.net["accum_gradient_update"])
                    _ = self.sess.run(self.net["accum_gradient_clean"])
                if i%500 == 0:
                    l1,l2,l3,seed_l,expand_l,constrain_l,loss,lr = self.sess.run([self.loss_1,self.loss_2,self.loss_3,self.loss["seed"],self.loss["expand"],self.loss["constrain"],self.loss["total"],self.net["lr"]],feed_dict=params)
                    print("epoch:%f, iteration:%f, lr:%f, loss:%f" % (epoch,i,lr,loss))
                    print("seed_loss:%f,expand_loss:%f,constrain_loss:%f" % (seed_l,expand_l,constrain_l))
                    print("l1:%f l2:%f l3:%f" % (l1,l2,l3))

                if i%3000 == 2999:
                    self.saver["norm"].save(self.sess,os.path.join(self.config.get("saver_path","saver"),"norm"),global_step=i)
                i+=1
                epoch = i / iterations_per_epoch_train

            end_time = time.time()
            print("end_time:%f" % end_time)
            print("duration time:%f" %  (end_time-start_time))

if __name__ == "__main__":
    batch_size = 4 # the actual batch size is  batch_size * accum_num
    input_size = (321,321)
    category_num = 21
    epoches = 30
    data = dataset({"batch_size":batch_size,"input_size":input_size,"epoches":epoches,"category_num":category_num,"categorys":["train"]})
    sec = SEC({"data":data,"batch_size":batch_size,"input_size":input_size,"epoches":epoches,"category_num":category_num,"init_model_path":"./model/init.npy","accum_num":16})
    #sec = SEC({"data":data,"batch_size":batch_size,"input_size":input_size,"epoches":epoches,"category_num":category_num,"model_path":"./old_saver/20180315-2-5/norm-23999","accum_num":16}) # this for continuing train after interupting

    lr = 1e-3
    sec.train(base_lr=lr,weight_decay=5e-5,momentum=0.9,batch_size=batch_size,epoches=epoches)
