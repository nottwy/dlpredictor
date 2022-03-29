import os
import sys
from argparse import ArgumentParser,RawDescriptionHelpFormatter
from functools import partial

from argparse import RawTextHelpFormatter

#specific
from .clr_callback import CyclicLR
from .dlCallback import CustomMCP

import numpy  as np
import pandas as pd

#sklearn
from sklearn.metrics import roc_curve

from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, concatenate
from tensorflow.keras.layers import GaussianDropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers,Model
from tensorflow.keras.callbacks import  EarlyStopping,ModelCheckpoint,TensorBoard
from tensorflow.keras.models import load_model

import tensorflow.keras.backend as K
import tensorflow as tf
# noinspection PyUnresolvedReferences
import tensorflow.math as tfmath

class Data:

  print = partial(print, flush=True)
  def __init__(self):
    #1. from parameters
    self.select_model = None
    self.cv_strategy = None
    self.feature_path = None
    self.label_path   = None
    self.output_name = None
    self.drug_list = None
    self.debug = None

    self.feature      = None
    self.label        = None

    #parameter for deep learning
    self.final_model = None
    self.thresholds = []
    self.parallel = None
    self.batch_size_per_gpu = 32

    #cross validation
    #control fit batch size
    self.batch_size = 32        #original, 20200815

    #Datasets
    self.x_train = None
    self.y_train = None
    self.x_val = None
    self.y_val = None

  def predict(self,index):
    return self.final_model.predict(self.feature[index])

  # check at 20210406
  # todo: improve: now we have to make sure file ends with .npy or .csv
  def read(self,fn):
    suffix = os.path.splitext(fn)[-1]
    if suffix == ".csv":
      self.feature = np.loadtxt(fn,delimiter=",")
    elif suffix == ".npy": #save as .npy for quick loading
      self.feature = np.load(fn)

  @staticmethod
  # check at 20210406
  def remove_invalid_label_column(label):
    ncol = []

    for i in range(label.shape[1]):
      index_list = list(pd.value_counts(label.iloc[:, i]).index)
      # samples have to be made up with positive and negative.
      if 0 not in index_list or 1 not in index_list:
        ncol.append(label.columns[i])
    if len(ncol) != 0:
      print("We have removed following column(s):")
      print(ncol)
    return label.drop(columns=ncol)

  #check at 20210406
  def load_data(self):
    #load feature
    print("1. loading feature files ...")
    self.read(self.feature_path)
    print("Finish reading: ",self.feature_path)

    #load label
    print("2. loading label files ...")
    # todo: drug_list is deprecated
    if self.drug_list != None:
      label = pd.read_csv(self.label_path,
                          header=None,
                          names=self.drug_list)
    else:
      label = pd.read_csv(self.label_path)
    print("Finish reading",self.label_path)

    # quality control for label
    label = Data.remove_invalid_label_column(label)
    label = label.astype('float32')

    # create .label and .drug_list
    self.label = np.array(label)
    self.drug_list = label.columns.tolist()

  #assignment part
  #the values assigned here are only used for training
  def cv_MSSS_assignment(self, all_train_index, test_index, train_index, val_index):
      x_train_tmp, x_test = self.feature[all_train_index], self.feature[test_index]
      y_train_tmp, y_test = self.label[all_train_index], self.label[test_index]

      x_train, x_val = x_train_tmp[train_index], x_train_tmp[val_index]
      y_train, y_val = y_train_tmp[train_index], y_train_tmp[val_index]

      self.x_train = x_train
      self.y_train = y_train
      self.x_val = x_val
      self.y_val = y_val
      self.x_test = x_test
      self.y_test = y_test

  def cv_KFold_assignment(self, train, val):
      self.x_train = self.feature[train]
      self.y_train = self.label[train]
      self.x_val = self.feature[val]
      self.y_val = self.label[val]


#Function
#1. dealing with parameters
  @staticmethod
  def read_from_file(conf_file):
    parameters = {}
    with open(conf_file,"r") as input:
      for i in input:
        i = i.strip()
        if i.startswith('#') or not len(i): #skip comment line and blank line
          continue
        para = i.split('=')
        parameters[para[0].strip()] = para[1].strip('\t\n \'\"')
    return parameters
  @staticmethod
  def read_from_commandline(args):
    parameters = {}
    parameters['model'] = args.model
    parameters['lpath'] = args.lpath
    parameters['fpath'] = args.fpath
    parameters['debug'] = args.debug
    parameters['prefix'] = args.prefix
    parameters['cv_strategy'] = args.cv_strategy
    parameters['parallel'] = args.parallel
    return parameters

  @staticmethod
  def get_parameter():
    #1. prepare ArgumentParser
    header = '''
==============================
  Deep Learning framework
==============================
         Author: Yu Wang
        version: 1.0
latest modified: Aug. 11,2020
===============================
'''
    parser = ArgumentParser(description=header,
                            formatter_class=RawDescriptionHelpFormatter)

    # group 1: configure file
    conf = parser.add_argument_group('Parameters from a file')
    conf.add_argument('--conf',action="store",default=None,required=False,
                       help="config file")

    # group 2: model, CV and Data
    data = parser.add_argument_group('About Data')
    data.add_argument('--model',action="store",default=None,required=False,
                       help="the model you want to apply to your data. The names of choosable models include: wdnn, wdnn_modified, deepamr, deepamr_modified.(required)")
    data.add_argument('--cv_strategy',action="store",default=None,required=False,
                       help="the cross-validation strategy you want to apply to your model. The names of choosable models include: kfold, msss.(required)")
    data.add_argument('--fpath',action="store",default=None,required=False,
                         help="path of the file storing features.(required))")
    data.add_argument('--lpath',action="store",default=None,required=False,
                         help="path of the file storing labels.(required)")
    # group 3: Output
    output = parser.add_argument_group('About Output')
    output.add_argument('--prefix',action="store",default=None, required=False,
                         help="The prefix of the output file storing metrics evaluating model performance like sensitivity,accuracy, auc and so on.")
    output.add_argument('--debug',action="store",default=None, required=False,
                         help="Output extra information used to debug.")
    # group 4: parallel
    parallel = parser.add_argument_group('About Parallel')
    parallel.add_argument('--parallel',action="store",default=None,required=False,
                          help="the parallel strategy you want to use. The choosable parallel strategies include: shmd (single-host, multi-device(gpu) synchronous training)")


    #no argument print help
    if len(sys.argv) == 1:
      parser.print_help(sys.stderr)
      sys.exit("Error: Please provide arguments!")

    arg = parser.parse_args()

    #commandline or file
    if arg.conf == None:
      parameters = Data.read_from_commandline(arg)
    else:
      parameters = Data.read_from_file(arg.conf)
    return parameters

class WDNN(Data):
  def __init__(self):
    #general
    Data.__init__(self)

    #WDNN specific
    self.nepoch = 100            #original,20200812
    self.batch_size_per_gpu = 32 #original,20200820

    #WDNN specific datasets
    self.alpha        = None
    self.alpha_train = None
    self.alpha_val = None
    self.alpha_test = None

    self.best_wdnn_model_fn = None
    self.wdnn_cb = []

    self.wdnn = None


# init code block
  #check at 20210406
  def preprocess(self):             #original,20200812
    # Part I: assign 0.5 to each unknown loci
    self.feature[self.feature == -1] = 0.5

    # part II: filter out loci whose counts < 30
    # 1. get the indices of the elements that we want.
    indices = (self.feature == 1).sum(axis=0) >= 30
    # 2. get the elements using the indices
    self.feature = self.feature[:,indices]
  #checked at 20210406
  def alpha_init(self):
    self.alpha = np.zeros(self.label.shape,dtype='float32')
    for i in range(self.label.shape[1]):
      x1 = np.count_nonzero(self.label[:,i] == 0) #0 susceptive
      x2 = np.count_nonzero(self.label[:,i] == 1) #1 resistant
      alpha = 1 - x2 / (x1 + x2)
      #alpha is the coefficient of the positive sample in the custom loss function, herein x2.
      self.alpha[self.label[:,i] == 0,i] = (-1) * alpha
      self.alpha[self.label[:,i] == 1,i] = (+1) * alpha
      self.alpha[self.label[:,i] == -1,i] = 0.000
  def prepare_for_train(self):
    self.load_data()
    self.preprocess()
    self.alpha_init()

#build and train code block
  @staticmethod
  def masked_weighted_accuracy(alpha, y_pred):
    total   = tfmath.reduce_sum(
                tf.cast(tf.not_equal(alpha, 0.), tf.float32),
                axis=1
              )
    y_true_mask = tf.cast(tfmath.greater(  alpha, 0.), tf.float32)
    pheno_mask  = tf.cast(tfmath.not_equal(alpha, 0.), tf.float32)
    correct = tf.reduce_sum(
                tf.cast(tf.equal(y_true_mask, tf.round(y_pred)), tf.float32) * pheno_mask,
                axis=1
              )
    return correct / total
  @staticmethod
  def masked_multi_weighted_bce(alpha, y_pred):
    #upweight sparser class
    #resistant ,+1>0,P=y_pred
    #susceptive,-1<0,P=1-y_pred
    epsilon = 1e-07
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    y_true_mask = tf.cast(tfmath.greater(alpha,0.),tf.float32)
    pheno_mask = tf.cast(tfmath.not_equal(alpha,0.),tf.float32) # 0 or 1
    ind_pheno_n = tfmath.reduce_sum(pheno_mask,axis=1)
    alpha = tfmath.abs(alpha)

    bce = - alpha * y_true_mask * tfmath.log(y_pred) - (1.0 - alpha) * (1 - y_true_mask) * tfmath.log(1 - y_pred)
    masked_bce = bce * pheno_mask
    # todo: every drug could own its own weight,so we could get a different sum
    return tfmath.reduce_sum(masked_bce,axis = 1) / ind_pheno_n
  def build_model(self,regularization_factor=1e-8):
    input_data = Input(shape=(self.feature.shape[1],))
    x = Dense(256, activation='relu',
              kernel_regularizer=regularizers.l2(regularization_factor))(input_data)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu',
              kernel_regularizer=regularizers.l2(regularization_factor))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu',
              kernel_regularizer=regularizers.l2(regularization_factor))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    wide_deep = concatenate([input_data, x])
    preds = Dense(self.label.shape[1],
                  activation='sigmoid',
                  kernel_regularizer=regularizers.l2(regularization_factor),
                  name="wdnn_final_layer")(wide_deep)
    wdnn = Model(inputs=input_data, outputs=preds,name = 'WDNN')
    opt = Adam(lr=np.exp(-1.0 * 9))
    wdnn.compile(optimizer=opt,
                  loss=[WDNN.masked_multi_weighted_bce],
                  metrics=[WDNN.masked_weighted_accuracy])
    self.wdnn = wdnn

  def prepare_callbacks(self):
    self.wdnn_cb = []
  def cv_KFold_assignment(self, train, val):
    super().cv_KFold_assignment(train,val)
    self.alpha_train = self.alpha[train]
    self.alpha_val   = self.alpha[val]

  def cv_MSSS_assignment(self, all_train_index, test_index, train_index, val_index):
    super().cv_MSSS_assignment(all_train_index, test_index, train_index, val_index)
    alpha_train_tmp, alpha_test = self.alpha[all_train_index], self.alpha[test_index]
    self.alpha_train, self.alpha_val = alpha_train_tmp[train_index], alpha_train_tmp[val_index]

  @staticmethod
  def optimal_threshold(ptrue, ppred):

    fpr, tpr, threshold = roc_curve(ptrue, ppred)
    tf_sum = pd.DataFrame(
      {'tf':        tpr + (1 - fpr),  # tpr+tnr
       'threshold': threshold,
       }
    )
    #wdnn solution
    #argsort: ascending order (from small to large)
    sort_index = (tf_sum.tf - 0).abs().argsort()
    max_sum = tf_sum.iloc[sort_index.iloc[-1]]  # sort |tpr + tnr|,select the max one
    return float(max_sum['threshold'])

    # todo: if tf equal,compare threshold
    #my solution
    #sorted = tf_sum.sort_values(by=['tf','threshold'],ascending=(False,False))
    #return float(sorted.iloc[0].threshold)


  def determine_threshold(self):
    probs = self.final_model.predict(self.x_train)
    for i,drug in enumerate(self.drug_list):
      non_missing_val = np.where(self.y_train[:,i] != -1)[0]
      trues = np.reshape(
        self.y_train[non_missing_val,i],
        (len(non_missing_val),1)
      )
      preds = np.reshape(
        probs[non_missing_val,i],
        (len(non_missing_val),1)
      )
      self.thresholds.insert(i,self.optimal_threshold(trues,preds))
  def extra_cb(self,cv_num,output_dir):
    self.best_wdnn_model_fn = output_dir + "/best_wdnn_model_"+str(cv_num)+".tf"

    save_wdnn_model = CustomMCP(
      filepath=self.best_wdnn_model_fn,
      save_weights_only=False,
      save_freq="train",
      monitor='val_loss',
      mode='min',
      verbose=1,
      save_best_only=True)
    save_wdnn_tb = TensorBoard(
      log_dir=self.best_wdnn_model_fn+"_tb",
      histogram_freq=1,
      write_graph=True,
      write_images=True
    )
    self.wdnn_cb.append(save_wdnn_tb)
    self.wdnn_cb.append(save_wdnn_model)
  def train(self, nverbose=2,BSIZE=32):
    self.wdnn.fit(
      self.x_train,
      self.alpha_train,
      epochs=self.nepoch,
      verbose=nverbose,
      batch_size=BSIZE,                               #original,20200820
      validation_data=(self.x_val, self.alpha_val),
      callbacks=self.wdnn_cb                         #save information
    ) #original,20200812


    self.final_model = self.wdnn

  def get_probs(self, probs, samples, drug):
    return probs[samples, drug]


class DeepAMR(Data):
  def __init__(self):
    Data.__init__(self)
    self.autoencoder = None
    self.encoder = None
    self.deepamr = None

    self.best_ae_model_fn = None
    self.best_deepamr_model_fn = None

    self.ae_cb = []
    self.deepamr_cb = []
    self.metrics = None
    self.loss = None
    self.loss_weights = None

    #datasets used for determine_threshold
    self.th_x = None
    self.th_y = None

    self.deepamr_cb_monitor = 'val_bce_without_nan'

#init code block
  def preprocess(self):
    return
  def prepare_for_train(self):
    self.load_data()
    self.preprocess()


#train code block
  #grouping and makedivisible
  def remove_extra_element(self, x, y):
    b_s = self.batch_size
    if x.shape[0] / b_s != 0:
      to_remove = x.shape[0] - int(x.shape[0] // b_s * b_s)  #python 3.0: '//'
      x = x[:-to_remove]
      y = y[:-to_remove]
    return x, y
  def makedivisible_to_all(self):
    self.x_train,self.y_train=self.remove_extra_element(self.x_train,self.y_train)
    self.x_test ,self.y_test =self.remove_extra_element(self.x_test ,self.y_test)
    self.x_val  ,self.y_val  =self.remove_extra_element(self.x_val  ,self.y_val)


  #build and train
  def cv_MSSS_assignment(self, all_train_index, test_index, train_index, val_index):
    super().cv_MSSS_assignment(all_train_index, test_index, train_index, val_index)
    self.makedivisible_to_all()
      
  '''
  include
  1. EarlyStopping
  2. CyclicLR
  for ae and deepamr
  '''
  def prepare_callbacks(self):
    # 1. for autoencoder
    esp = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto') #original, 20200816
    clr = CyclicLR(base_lr=0.001, max_lr=0.9,
                   step_size=100., mode='triangular2')          #original,20200816
    self.ae_cb = [clr, esp]

    # 2. for deepamr
    esp = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')  #original,20200816
    clr = CyclicLR(base_lr=0.0001, max_lr=0.003,
                   step_size=100., mode='triangular2')  #original,20200816
    self.deepamr_cb = [clr, esp]

  '''
  include
  1. CustomMCP
  2. TensorBoard
  for ae and deepamr
  '''
  def extra_cb(self,cv_num,od): # od: output directory
    self.best_ae_model_fn = od + "/best_ae_model_"+str(cv_num)+".tf"
    save_ae_model = CustomMCP(
      filepath=self.best_ae_model_fn,
      save_weights_only=False,
      monitor='val_loss',
      mode='min',
      save_freq="train",
      verbose=1,
      save_best_only=True)
    save_ae_tb = TensorBoard(
      log_dir=self.best_ae_model_fn+"_tb",
      histogram_freq=1,
      write_graph=True,
      write_images=True
    )
    self.ae_cb.append(save_ae_tb)
    self.ae_cb.append(save_ae_model)

    self.best_deepamr_model_fn = od + "/best_deepamr_model_"+str(cv_num)+".tf"
    save_deepamr_model = CustomMCP(
      filepath=self.best_deepamr_model_fn,
      save_weights_only=False,
      monitor='val_loss',
      mode='min',
      save_freq="train",
      verbose=1,
      save_best_only=True)
    save_deepamr_tb = TensorBoard(
      log_dir=self.best_deepamr_model_fn+"_tb",
      histogram_freq=1,
      write_graph=True,
      write_images=True
    )
    self.deepamr_cb.append(save_deepamr_tb)
    self.deepamr_cb.append(save_deepamr_model)

  def build_autoencoder(self, dims=[500, 1000, 20],
                        act='relu', init='uniform',
                        Optimizer='SGD',
                        Loss='binary_crossentropy',
                        drop_rate=0.3): #original, 20200815
    n_stacks = len(dims) #n_stacks = 3
    # input
    input_snp = Input(shape=(self.feature.shape[1],), name='input')
    x = input_snp
    x = GaussianDropout(drop_rate)(x)
    # internal layers in encoder
    # 500, 1000
    for i in range(n_stacks - 1):  #0->1
      x = Dense(dims[i],
                activation=act,
                kernel_initializer=init,
                name='encoder_%d' % i)(x)

    # hidden layer, features are extracted from here
    # 20
    encoded = Dense(dims[n_stacks - 1],
                    kernel_initializer=init,
                    name='encoder_%d' % (n_stacks - 1))(x) # 2

    x = encoded
    # internal layers in decoder
    # 1000,500
    for i in range(n_stacks - 2, -1, -1):  #1->0
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
    self.autoencoder.compile(optimizer=Optimizer, loss=Loss)      #original, 20200815

  #build and compile
  def build_deepamr(self,
                    act='sigmoid',
                    init='uniform',
                    loss_func='binary_crossentropy',
                    metric=['accuracy'],
                    Optimizer='Nadam'):
    pred = []
    losses = []
    metrics = {}
    loss_weights = []
    for i in range(self.label.shape[1]):
      taskname = 'task' + str(i)
      # 1.prepare loss and metrics
      losses.append(loss_func)
      metrics[taskname] = metric
      loss_weights.append(1)

      # 2.build layers
      item = Dense(1,
                   kernel_initializer=init,
                   activation=act,
                   name=taskname)(self.encoder.output)
      pred.append(item)

    # for autoencoder
    losses.append(loss_func)
    metrics['decoder']=metric
    loss_weights.append(1)
    pred.append(self.autoencoder.output)
    # Compile model
    self.deepamr = Model(inputs=self.encoder.input,
                         outputs=pred,
                         name='deepamr')
    self.deepamr.compile(loss=losses,
                         loss_weights=loss_weights,
                         optimizer=Optimizer,
                         metrics=metrics)              #original
  def build_model(self):
    self.build_autoencoder()
    self.build_deepamr()
  def train_autoencoder(self,Epochs=100):            #original, 20200815
    ae =self.autoencoder
    enc=self.encoder
    ae.fit(self.x_train, self.x_train,
            batch_size=self.batch_size,
            epochs=Epochs,
            shuffle=False,
            verbose=2,
            validation_data=(self.x_val,self.x_val),            #original is list, changed to tuple,20200817
            callbacks=self.ae_cb)                               #original, 20200815
    self.autoencoder=ae
    self.encoder=enc
  @staticmethod
  def optimal_threshold(ptrue, ppred):
    fpr, tpr, threshold = roc_curve(ptrue, ppred)
    i = np.arange(len(tpr))

    # wdnn: max tpr+tnr
    # deepamr: min tpr-tnr
    diff = pd.DataFrame(
      {'tf':        tpr - (1 - fpr),  # tpr-tnr
       'threshold': threshold,
       }
    )
    # python data science book,page 97
    sorted_index = (diff.tf - 0).abs().argsort()
    min_diff = diff.iloc[sorted_index.iloc[0]]  # sort |tpr - tnr|,select the minimal one

    return float(min_diff['threshold'])

  def determine_threshold(self):
    probs = self.final_model.predict(self.th_x)

    for i,drug in enumerate(self.drug_list):
      non_missing_val = np.where(self.th_y[i] != -1)[0]
      trues = np.reshape(
        self.th_y[i][non_missing_val],
        (len(non_missing_val),1)
      )
      preds = np.reshape(
        probs[i][non_missing_val],
        (len(non_missing_val),1)
      )
      self.thresholds.insert(i, self.optimal_threshold(trues,preds))

  def train_deepamr(self,
                    Epochs=100,
                    nverbose=2,
                    custom_objects={'CustomMCP': CustomMCP,
                                    'CyclicLR': CyclicLR}
                    ): #original,20200815
    # prepare label inputs
    y_train_list = []
    y_val_list = []
    y_tv_list = []
    for i in range(self.y_train.shape[1]):
      y_train_list.append(self.y_train[:, i])
      y_val_list.append(self.y_val[:, i])
      y_tv_list.append(np.concatenate((self.y_train[:, i],self.y_val[:, i]),axis = 0))
    y_train_list.append(self.x_train)
    y_val_list.append(self.x_val)
    y_tv_list.append(np.concatenate((self.x_train,self.x_val),axis=0))

    self.deepamr.fit(self.x_train, y_train_list,
                     validation_data=(self.x_val, y_val_list),
                     shuffle=False,
                     epochs=Epochs,
                     verbose=nverbose,
                     batch_size=self.batch_size,
                     callbacks=self.deepamr_cb)

    #load best model: 20210421
    self.final_model = load_model(self.best_deepamr_model_fn,custom_objects=custom_objects)

    #for thresholds
    x_tv = np.concatenate((self.x_train,self.x_val),axis=0)
    self.th_x = x_tv
    self.th_y = y_tv_list




  def train(self,ae_epoch=5,deepamr_epoch=5):  #original,20200815
    self.train_autoencoder(Epochs=ae_epoch)
    self.train_deepamr    (Epochs=deepamr_epoch)

  def get_probs(self,probs,samples,drug):
    return probs[drug][samples]

class CNNGWP(Data):
  def __init__(self):
    Data.__init__(self)

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
    self.cnngwp = Model(inputs=input_data,outputs=final,name="cnngwp")