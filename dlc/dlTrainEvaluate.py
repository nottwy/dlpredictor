#wdnn
from sklearn.model_selection import KFold
#deepamr
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix

import os,sys,shutil

class Evaluate:
  def __init__(self):
    self.cv_splits = None

  @staticmethod
  def calc_metrics(trues,preds,md,i,index):
    auc = roc_auc_score(trues, preds)
    auc_pr = average_precision_score(trues, preds)

    # th
    th = md.thresholds[i]
    pred_prob_tf = np.where(preds >= th, 1, 0)

    tn, fp, fn, tp = confusion_matrix(trues, pred_prob_tf).ravel()

    #*_num == 0 equals to tp/tn == 0, so sen/sps/npv/precision must be equal to zero.
    predP_num = (tp+fp) if (tp+fp) != 0 else 0.00001
    trueP_num = (tp+fn) if (tp+fn) != 0 else 0.00001
    trueN_num = (tn+fp) if (tn+fp) != 0 else 0.00001
    predN_num = (tn+fn) if (tn+fn) != 0 else 0.00001

    sen = tp / trueP_num  # or recall
    sps = tn / trueN_num
    npv = tn / predN_num
    precision = tp / predP_num

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    p_s = precision + sen if (precision + sen) != 0 else 0.00001
    f1_score = 2 * (precision * sen) / p_s

    # 4. count other information
    nresis = len(np.where(md.label[:, i] == 1)[0])
    nsus = len(np.where(md.label[:, i] == 0)[0])

    val_nresis = np.count_nonzero(md.label[index, i] == 1)
    val_nsus   = np.count_nonzero(md.label[index, i] == 0)
    return ([auc,auc_pr,sen,sps,precision,npv,accuracy,f1_score,nresis,nsus],[th,tn,fp,fn,tp,val_nresis,val_nsus])

  @staticmethod
  def get_metrics(md,index,cv_num):#md: model;index:index of data used for predicting
    #1. predict using datasets corresponding to index
    probs = md.predict(index)
    
    #2.evaluate the performance of the model based on the output of predicting
    #prepare table header
    column_names = ['cv_num','Algorithm','Drug','AUC','AUC_PR',"sen","sps","precision","npv","accuracy","f1_score","nresis","nsus"]
    results = pd.DataFrame(columns=column_names)
    results_index = 0
    if md.debug != 0:
      column_names = ['cv_num','Drug','th','tn','fp','fn','tp','val_nresis','val_nsus']
      debug = pd.DataFrame(columns=column_names)
      debug_index = 0

    #3. calculate metrics for each drug
    for i,drug in enumerate(md.drug_list):
      #1. get content of val label data
      label_val = md.label[index]

      #2. remove missing_value
      # different NN architecture, different probs data structure
      # so I prepare the model-specific get_probs method for each model.
      non_missing_val = np.where(label_val[:,i] != -1)[0]
      trues = np.reshape(
        label_val[non_missing_val,i],
        (len(non_missing_val), 1)
        )
      preds = np.reshape(
        md.get_probs(probs,non_missing_val,i),
        (len(non_missing_val), 1)
        )

      #3. calculate metrics using trues and preds
      # ignore debug
      (performance_metrics,check_metrics) = Evaluate.calc_metrics(trues,preds,md,i,index)

      #4. save
      #save result
      results.loc[results_index] = [cv_num,md.select_model,drug]+performance_metrics
      results_index += 1

      #save debug
      if md.debug != 0:
        debug.loc[debug_index] = [cv_num, drug] + check_metrics
        debug_index += 1

    #4. output results
    results.to_csv(md.output_name,index=False,mode='a')
    if md.debug != 0:
      debug.to_csv(md.debug,index=False,mode='a')
  @staticmethod
  def prepare_for_KFold(md):
    md.cv_splits = 10

  @staticmethod
  def wdnn_KFold(md):# md: model
    #datasets: only train and validation
    Evaluate.prepare_for_KFold(md)
    kf = KFold(n_splits=md.cv_splits, shuffle=True,
               random_state = 333)
    cv_num = 0
    for train, val in kf.split(md.feature):  # kf.split returns ndarray.
      cv_num += 1
      #callback
      #check 20210407
      md.prepare_callbacks()  #clear
      if md.debug != 0:
        # 1. prepare a directory for storing debugging information
        output_dir = "debug_info_" + str(cv_num)
        if os.path.exists(output_dir):
          shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        # 2. Store the index used to split samples into training/validation
        train_index_fn = output_dir + "/train_index_" + str(cv_num) + ".npy"
        val_index_fn   = output_dir + "/val_index_" + str(cv_num) + ".npy"
        np.save(train_index_fn, train)
        np.save(val_index_fn, val)

        # 3. For callback
        md.extra_cb(cv_num,output_dir)
      #parallel mode will affect two parts: model building and training
      if md.parallel == "shmd":
        strategy = tf.distribute.MirroredStrategy()
        ngpus =  strategy.num_replicas_in_sync
        batch_size_per_gpu = md.batch_size_per_gpu
        batch_size = ngpus * batch_size_per_gpu
        with strategy.scope():
          md.build_model()
      else:
        md.build_model()
      md.cv_KFold_assignment(train,val)
      # todo: make deepamr support BSIZE
      if md.parallel == "shmd":
        md.train(BSIZE=batch_size)
      else:
        md.train()
      md.determine_threshold()
      Evaluate.get_metrics(md,val,cv_num)

  @staticmethod
  def prepare_for_MSSS(md):
    #control train/test split
    # default ratio: ((training_7(training_8:validation_2)):(test_3))
    # todo: modify this function to make it support customizing. 20210518
    md.N_splits_tt = 1       # original, 20200815
    md.N_splits_tv = 1       # original, 20200815
    # control test/(train+validation)
    # control validation/train
    md.Test_size = 0.3        # original, 20200815
    md.Val_size  = 0.2        # original, 20200815
    #for MultilabelStratifiedShuffleSplit
    md.rand_sta = 333         # original, 20200815

  @staticmethod
  def MSSS(md):
    #strategy: MultilabelStratifiedShuffleSplit
    #datasets: three parts: train,validation and test
    cv_num = 0
    Evaluate.prepare_for_MSSS(md)
    msss1 = MultilabelStratifiedShuffleSplit(n_splits=md.N_splits_tt,
                                             test_size=md.Test_size,
                                             random_state = md.rand_sta)
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=md.N_splits_tv,
                                             test_size=md.Val_size,
                                             random_state=md.rand_sta)
    # msss1.split returns ndarray
    # todo: make MSSS support parallel mode
    for all_train_index, test_index in msss1.split(md.feature,md.label):
      x_train_tmp,y_train_tmp = md.feature[all_train_index],md.label[all_train_index]
      cv_num += 1
      for train_index, val_index in msss2.split(x_train_tmp, y_train_tmp):
        md.prepare_callbacks() # clear callback list through assignment in this function.
        if md.debug != 0:
          output_dir = "debug_info_"+str(cv_num)
          if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
          os.mkdir(output_dir)

          all_train_index_fn = output_dir + "/all_train_index_" + str(cv_num) + ".npy"
          test_index_fn      = output_dir + "/test_index_"      + str(cv_num) + ".npy"
          train_index_fn     = output_dir + "/train_index_"     + str(cv_num) + ".npy"
          val_index_fn       = output_dir + "/val_index_"       + str(cv_num) + ".npy"
          np.save(all_train_index_fn, all_train_index)
          np.save(test_index_fn     , test_index)
          np.save(train_index_fn    , train_index)
          np.save(val_index_fn      , val_index)

          md.extra_cb(cv_num,output_dir)
        md.build_model()
        md.cv_MSSS_assignment(all_train_index,
                              test_index,
                              train_index,
                              val_index)#checked, 20200815
        md.train()
        md.determine_threshold()
        Evaluate.get_metrics(md,test_index,cv_num)