from . import dlBase
from .dlBase import Data,WDNN,DeepAMR
from .dlCallback import CustomMCP
from .clr_callback import CyclicLR
from .dlTrainEvaluate import Evaluate

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Input, Dense, GaussianDropout
from tensorflow.keras.callbacks import  TensorBoard, EarlyStopping
from tensorflow.keras import regularizers,Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.math as tfmath

import numpy as np
import sys,os

class WDNN_modified1(WDNN):
  def preprocess(self):
    #Part I: assign 0.5 to each unknown loci
    self.feature[self.feature == -1] = 0.5

  @staticmethod
  def masked_weighted_accuracy(y_true, y_pred):
    return tf.reduce_sum(y_true,axis=-1) / tf.reduce_sum(y_true,axis=-1)
    '''
    total   = tf.reduce_sum(
                tf.cast(tf.not_equal(alpha, 0.), tf.float32),
                axis=-1
              )
    y_label = tf.cast(  tf.greater(alpha, 0.), tf.float32)
    mask    = tf.cast(tf.not_equal(alpha, 0.), tf.float32)
    correct = tf.reduce_sum(
                tf.cast(tf.equal(y_label, tf.round(y_pred)), tf.float32) * mask,
                axis=-1
              )
    return correct / total
    '''
class WDNN_modified(WDNN):
    def preprocess(self):  # original,20200812
      return
      # Part I: assign 0.5 to each unknown loci
      #self.feature[self.feature == -1] = 0.5

      # part II: filter out loci whose counts < 30
      # 1. get the indices of the elements that we want.
      #indices = np.where((self.feature == 1).sum(axis=0) >= 30)[0]
      # 2. get the elements using the indices
      #self.feature = self.feature[:, indices]
class WDNN_cnngwp(WDNN):
  def build_model(self,nfilter=64, nkernel=27, nlambda=1e-8): #original 20200919
    input_data = Input(shape=(self.feature.shape[1],1),name="input")
    x = Conv1D(filters=nfilter,
               kernel_size=nkernel,
               activation='relu',
               strides=2,
               padding="same")(input_data)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    final = Dense(units=self.label.shape[1],
                  activation='sigmoid',
                  kernel_regularizer=regularizers.l1(l=nlambda))(x)
    self.wdnn = Model(inputs=input_data,outputs=final,name="cnngwp")
    opt = Adam(lr=0.00025)
    self.wdnn.compile(optimizer=opt,
                        loss=[self.masked_multi_weighted_bce],
                        metrics=[self.masked_weighted_accuracy])


class DeepAMR_modified(DeepAMR):
  def __init__(self):
    super().__init__()

  #autoencoder and deepamr will use this
  @staticmethod
  def bce_without_nan(y_true,y_pred):
    epsilon = 1e-07
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    y_true = tf.cast(y_true,tf.float32)
    pheno_mask = tf.cast(tfmath.not_equal(y_true,-1.0),tf.float32)
    ind_pheno_n = tfmath.reduce_sum(pheno_mask,axis = 1)

    #avoid zero to be divided
    max = tf.reduce_max(ind_pheno_n)
    ind_pheno_n = tf.clip_by_value(ind_pheno_n,epsilon,max)

    bce = - y_true * tfmath.log(y_pred) - (1 - y_true) * tfmath.log(1 - y_pred)
    masked_bce = bce * pheno_mask
    return tfmath.reduce_sum(masked_bce,axis = 1) / ind_pheno_n

  @staticmethod
  def acc_without_nan(y_true,y_pred):
    #for each sample, count its label
    total = tf.cast(tfmath.reduce_sum(
            tf.cast(tf.not_equal(y_true,-1.0),tf.float32),
            axis=1
            ),tf.float32)
    # avoid zero to be divided
    epsilon = 1e-07
    max = tfmath.reduce_max(total)
    total = tf.clip_by_value(total,epsilon,max)

    pheno_mask = tf.cast(tfmath.not_equal(y_true,-1.0), tf.float32)
    correct = tf.reduce_sum(
              tf.cast(tf.equal(y_true,tf.round(y_pred)),tf.float32) * pheno_mask,
              axis=1
              )
    return correct / total

  def build_autoencoder(self, dims=[500, 1000, 20],
                        act='relu', init='uniform',
                        Optimizer='SGD',
                        Loss='binary_crossentropy',
                        drop_rate=0.3):
    n_stacks = len(dims)
    # input
    input_snp = Input(shape=(self.feature.shape[1],), name='input')
    x = input_snp
    x = GaussianDropout(drop_rate)(x)
    # internal layers in encoder
    # 500, 1000
    for i in range(n_stacks - 1):  #0, 1
      x = Dense(dims[i],
                activation=act,
                kernel_initializer=init,
                name='encoder_%d' % i)(x)

    # hidden layer, features are extracted from here
    # 20
    encoded = Dense(dims[n_stacks - 1],
                    kernel_initializer=init,
                    name='encoder_%d' % (n_stacks - 1))(x)

    x = encoded
    # internal layers in decoder
    # 1000,500
    for i in range(n_stacks - 2, -1, -1):  #1, 0
      x = Dense(dims[i],
                activation=act,
                kernel_initializer=init,
                name='decoder_%d' % i)(x)  #original,20200817

    # output
    x = Dense(self.feature.shape[1],
              kernel_initializer=init,
              activation='sigmoid',
              name='decoder')(x)       #original,20200815

    decoded = x
    self.autoencoder = Model(inputs=input_snp, outputs=decoded, name='AE')
    self.encoder     = Model(inputs=input_snp, outputs=encoded, name='encoder')

    #compile
    self.autoencoder.compile(optimizer=Optimizer,
                             loss=[self.bce_without_nan],
                             metrics=[self.acc_without_nan])      #original, 20200815


  def build_deepamr(self):
    super().build_deepamr(loss_func=[self.bce_without_nan],metric=[self.acc_without_nan])
  def train_deepamr(self,
                    Epochs=100,
                    nverbose=2,
                    custom_objects={'CustomMCP': CustomMCP,
                                    'CyclicLR': CyclicLR}):
      super().train_deepamr(Epochs=Epochs,nverbose=nverbose,
                            custom_objects={'CustomMCP': CustomMCP,
                                            'CyclicLR': CyclicLR,
                                            'acc_without_nan':self.acc_without_nan,
                                            'bce_without_nan':self.bce_without_nan})
  def train(self,ae_epoch=10,deepamr_epoch=5):  #original,20200815
    self.train_autoencoder(Epochs=ae_epoch)
    print("\n\n\nstart to train deepamr\n\n")
    self.train_deepamr    (Epochs=deepamr_epoch)

class DeepAMR_modified1(DeepAMR):
  def __init__(self):
    super().__init__()
    self.alpha = None
    self.alpha_train = None
    self.alpha_val = None
    self.alpha_test = None

  def alpha_init(self):
    self.alpha = np.zeros(self.label.shape,dtype='float32')
    self.alpha[self.label == 0] = (-1.0)
    self.alpha[self.label == 1] = (+1.0)

  def init(self):
    super().init()
    self.alpha_init()

  def train_deepamr(self,
                    Epochs=100,
                    ): #original,20200815
    self.batch_size=1
    super().train_deepamr(Epochs=100,nverbose=1)
  def build_deepamr(self,
                    act='sigmoid',
                    init='uniform',
                    loss_func=['binary_crossentropy'],
                    metric=['accuracy'],
                    Optimizer='Nadam'):
    super().build_deepamr(metric=[self.acc_without_nan],loss_func=[self.bce_without_nan])
    #tf.keras.metrics.BinaryAccuracy
    #super().build_deepamr(metric=[tf.keras.metrics.BinaryAccuracy()])
    #super().build_deepamr(metric=[tf.keras.metrics.CategoricalAccuracy()])
    #super().build_deepamr(metric=['accuracy'])


  @staticmethod
  def acc_without_nan(y_true,y_pred):
    #critical!
    total = tf.reduce_sum(
              tf.cast(
                tf.not_equal(y_true,0.),
                tf.float32
              ),
            axis=-1)

    y_label = tf.cast(tf.greater(y_true, 0.), tf.float32)
    keep    = tf.cast(tf.not_equal(y_true,0.), tf.float32)
    correct = tf.reduce_sum(
                tf.cast(
                  tf.equal(
                    y_label,
                    tf.round(y_pred)
                  ),
                  tf.float32
                ) * keep,
                axis=-1)
    return correct / total

  @staticmethod
  def bce_without_nan(y_true,y_pred):
    #critical!
    y_true_ = tf.cast(tf.greater(y_true, 0.), tf.float32)
    keep    = tf.cast(tf.not_equal(y_true, 0.), tf.float32)
    num_not_missing = tf.reduce_sum(keep)
    bce = - y_true_ * tf.math.log(y_pred) - (1.0 - y_true_) * tf.math.log(1.0 - y_pred)
    masked_bce = bce * keep
    return tf.reduce_sum(masked_bce) / num_not_missing


  def cv_KFold_assignment(self, train, val):
    super().cv_KFold_assignment(train,val)
    self.alpha_train = self.alpha[train]
    self.alpha_val   = self.alpha[val]

  @staticmethod
  def acc_without_nan(y_true,y_pred):
    #critical!
    total = tf.reduce_sum(
              tf.cast(
                tf.not_equal(y_true,0.),
                tf.float32
              ),
            axis=-1)

    y_label = tf.cast(tf.greater(y_true, 0.), tf.float32)
    keep    = tf.cast(tf.not_equal(y_true,0.), tf.float32)
    correct = tf.reduce_sum(
                tf.cast(
                  tf.equal(
                    y_label,
                    tf.round(y_pred)
                  ),
                  tf.float32
                ) * keep,
                axis=-1)
    return correct / total





  def prepare_callbacks(self):
    # 1. for autoencoder
    esp = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto') #original, 20200816
    clr = CyclicLR(base_lr=0.001, max_lr=0.9,
                   step_size=100., mode='triangular2')          #original,20200816
    #self.ae_cb = [clr, esp]

    # 2. for deepamr
    esp = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')  #original,20200816
    clr = CyclicLR(base_lr=0.0001, max_lr=0.003,
                   step_size=100., mode='triangular2')  #original,20200816
    #self.deepamr_cb = [clr, esp]
  '''
  def build_deepamr(self, act='sigmoid',
                    init='uniform',
                    loss_func='binary_crossentropy',
                    metric=['accuracy'],
                    Optimizer='Nadam'):
    super().build_deepamr(loss_func=DeepAMR_modified.bce_without_nan,
                          metric=[DeepAMR_modified.acc_without_nan])
  '''
  def build_deepamr(self,
                    act='sigmoid',
                    init='uniform',
                    loss_func=['binary_crossentropy'],
                    metric=['accuracy'],
                    Optimizer='Nadam'):
    pred = []
    losses = []
    metrics = []
    loss_weights = []
    for i in range(self.label.shape[1]):
      # 1.prepare loss and metrics
      losses.append(loss_func)
      metrics.append(metric)
      loss_weights.append(1)

      # 2.build layers
      item = Dense(1,
                   kernel_initializer=init,
                   activation=act,
                   name='task' + str(i))(self.encoder.output)
      pred.append(item)

    # for autoencoder
    losses.append(loss_func)
    metrics.append(metric)
    loss_weights.append(1)
    pred.append(self.autoencoder.output)

    # Compile model
    self.deepamr = Model(inputs=self.encoder.input,
                         outputs=pred,
                         name='deepamr')
    self.deepamr.compile(loss=[self.bce_without_nan],
                         loss_weights=loss_weights,
                         optimizer=Optimizer,
                         metrics=[self.acc_without_nan])              #original

  def train_deepamr(self,
                    Epochs=100,
                    ): #original,20200815
    # prepare label inputs
    y_train_list = []
    y_val_list = []
    y_tv_list = []
    for i in range(self.alpha_train.shape[1]):
      y_train_list.append(self.alpha_train[:, i])
      y_val_list.append(self.alpha_val[:, i])
      y_tv_list.append(np.concatenate((self.alpha_train[:, i],
                                       self.alpha_val[:, i]),axis = 0))
    y_train_list.append(self.x_train)
    y_val_list.append(self.x_val)
    y_tv_list.append(np.concatenate((self.x_train,self.x_val),axis=0))

    self.deepamr.fit(self.x_train, y_train_list,
                     validation_data=(self.x_val, y_val_list),
                     shuffle=False,
                     epochs=Epochs,
                     verbose=2,
                     batch_size=self.batch_size,
                     callbacks=self.deepamr_cb)
    self.final_model = self.deepamr
    #for thresholds
    x_tv = np.concatenate((self.x_train,self.x_val),axis=0)
    self.th_x = x_tv
    self.th_y = y_tv_list

def create_dl_model_obj(para):
  # 1. model
  # create an object of the specified deep learning model
  #class name
  my_Model = {
    "wdnn": WDNN,
    "deepamr": DeepAMR,
    "wdnn_cnngwp": WDNN_cnngwp,
    "wdnn_modified": WDNN_modified,
    "deepamr_modified": DeepAMR_modified
  }

  if 'model' in para and para['model'] != None:
    if para['model'] in my_Model:
      # todo: avoid write down the exact names of each model. try to find models dynamically.
      # reason: If we write like this, each time we add one more Model, we have to modified this model.
      model = my_Model[para['model']]()  # checked,20200820
      model.select_model = para['model']
      print("The model you choose is: %s" % (model.select_model))
    else:
      sys.exit("Error: The model " + para['model'] + " is not supported yet. Up to now, we only support \"" + ",".join(my_Model.keys())+"\"")
  else:
    sys.exit("Error: Please choose a model!")

  # 2. cv_strategy
  #function name
  my_CV = {
        'msss': Evaluate.MSSS,
        'wdnn_kfold':Evaluate.wdnn_KFold,
  }

  if 'cv_strategy' in para and para['cv_strategy'] != None:
    if para['cv_strategy'] in my_CV:
      model.cv_strategy = my_CV[para['cv_strategy']]  # checked,20200820
    else:
      sys.exit("Error: The cv_strategy "+para['cv_strategy']+" is not supported yet. Up to now, we only support \""+",".join(my_CV.keys())+"\"")
  else:
    sys.exit("Error: Please choose a cross-validation strategy!")

  # 3. prefix
  if para['prefix'] == None:
    model.output_name = "model_metrics.csv"
  else:
    model.output_name = para['prefix'] + ".model_metrics.csv"

  # 4. debug
  if para['debug'] == str(0):
    model.debug = 0
  else:
    if para['prefix'] == None:
      model.debug = "debug_info.csv"
    else:
      model.debug = para['prefix']+".debug_info.csv"

  #5. parallel
  # supported parallel methods
  my_Parallel = ["shmd"]

  if 'parallel' in para and para['parallel'] != None:
    assert para['parallel'] in my_Parallel,"Please choose a right parallel strategy!"
    model.parallel = para['parallel']

  #model,fpath,lpath
  # todo: write my own exception class
  #6. file path
  assert para['fpath'] != None, "Please provide the path of the feature file!"
  model.feature_path = para['fpath']
  assert para['lpath'] != None, "Please provide the path of the label file!"
  model.label_path = para['lpath']

  return model

def main():
  #os.environ['AUTOGRAPH_VERBOSITY']="10"
# 1.get parameters
  para = Data.get_parameter()
# 2.initialize an instance of a class object.
  model = create_dl_model_obj(para)
# 3.prepare things for training according to the parameters users provide.
  model.prepare_for_train()
# 4.cross validate model
  model.cv_strategy(model)

  '''
  #5. find contributor
  contributor = Find_contributor()

  contributor.saliency_map()
  contributor.grad_cam()
  contributor.permutation()
  '''