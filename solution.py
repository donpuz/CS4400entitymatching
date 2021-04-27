import pandas as pd
import numpy as np
from os.path import join
import torch
import deepmatcher as dm
import torch.nn as nn
import py_entitymatching as em

# Read tables, rename columns, and perform left joins
ltable = pd.read_csv(join('data', "ltable.csv"))
rtable = pd.read_csv(join('data', "rtable.csv"))
train = pd.read_csv(join('data', "train.csv"))

# Rename columns
ltable.columns = ['left_id', 'left_title', 'left_category', 'left_brand', 'left_modelno', 'left_price']
rtable.columns = ['right_id', 'right_title', 'right_category', 'right_brand', 'right_modelno', 'right_price']
train.columns = ['left_id', 'right_id', 'label']

# Perform left joins to combine tables (creates a labeled table)
train = train.merge(ltable, how="left", on="left_id")
train = train.merge(rtable, how="left", on="right_id")
train.insert(0, 'id', range(0, len(train)))

# Split data into training, validation, and test sets
dm.data.split(train, '/content/drive/My Drive/CS 4400', 'train.csv', 'validation.csv', 'test.csv', split_ratio=[0.6, 0.2, 0.2]) #changeeeeee

# Process data: tokenization, embedding
train, validation = dm.data.process(
    path='/content/drive/My Drive/CS 4400',
    train='train.csv',
    validation='validation.csv',
    test='test.csv'
    ignore_columns=['left_id, right_id'])

# Model definition: hybrid architecture
model = dm.MatchingModel(attr_summarizer="hybrid")

# Model training: hyper-parameter tuned
model.run_train(
    train,
    validation,
    epochs=15,
    batch_size=32,
    pos_neg_ratio=8.65,
    best_save_path='hybrid_model.pth')
    
# Run evaluation on the test set
model.run_eval(test)

# Load already trained model from file
# model.load_state('hybrid_model.pth')

# Restore column names
ltable.columns = ['id', 'title', 'category', 
    'brand', 'modelno', 'price']  
rtable.columns = ['id', 'title', 'category', 
    'brand', 'modelno', 'price']  

# Blocking step: uses an overlap blocking method
A = em.read_csv_metadata('data/ltable.csv', key='id')
B = em.read_csv_metadata('data/rtable.csv', key='id')
ob = em.OverlapBlocker()
candset_df = ob.block_tables(A, B, 'title', 'brand', overlap_size=1, 
    l_output_attrs=['id', 'title', 'category', 'brand', 'modelno', 'price'], 
    r_output_attrs=['id', 'title', 'category', 'brand', 'modelno', 'price'],
    l_output_prefix = 'left_', r_output_prefix = 'right_', allow_missing = True,
    n_jobs = 2)
candset_df.columns.values[0] = "id"
candset_df.to_csv('unlabeled.csv')

# Process unlabeled data
unlabeled = dm.data.process_unlabeled(
    path='unlabeled.csv',
    trained_model = model,
    ignore_columns=['left_id, right_id'])

# Predict unlabeled data
predictions = model.run_prediction(unlabeled)

# Split prediction output [model_score] to a two column dataframe [id, model_score]
predictions.to_csv('predictions.csv')
predictions = pd.read_csv("predictions.csv")

# Modified pairs2LR from sample solution
def pairs2LR(ltable, rtable, candset):
    ltable.index = ltable.id
    rtable.index = rtable.id
    pairs = np.array(candset)
    tpls_l = ltable.loc[pairs[:, 0], :]
    tpls_r = rtable.loc[pairs[:, 1], :]
    tpls_l.columns = ["left_" + col for col in tpls_l.columns]
    tpls_r.columns = ["right_" + col for col in tpls_r.columns]
    tpls_l.reset_index(inplace=True, drop=True)
    tpls_r.reset_index(inplace=True, drop=True)
    LR = pd.concat([tpls_l, tpls_r], axis=1)
    return LR

# Apply pairs2LR to labeled data
training_pairs = list(map(tuple, train[["left_id", "right_id"]].values))
training_df = pairs2LR(ltable, rtable, training_pairs)

# Re-join attributes to predictions dataframe
predictions["id"] = predictions["id"].astype(int)
candset_df["id"] = candset_df["id"].astype(int)
predictions = predictions.merge(candset_df, how = "left", on = "id")

# Find pairs with a match_score >= 0.5
matching_pairs = predictions.loc[predictions.match_score.values >= 0.5, ["left_id", "right_id"]]
matching_pairs = list(map(tuple, matching_pairs.values))

# Find positive matches in labeled data
matching_pairs_in_training = training_df.loc[train.label.values == 1, ["left_id", "right_id"]] # changeee train0
matching_pairs_in_training = set(list(map(tuple, matching_pairs_in_training.values)))

# Remove the matching pairs already in training
pred_pairs = [pair for pair in matching_pairs if
              pair not in matching_pairs_in_training]

# output
pred_pairs = np.array(pred_pairs)
pred_df = pd.DataFrame(pred_pairs, columns=["ltable_id", "rtable_id"])
pred_df.to_csv("output.csv", index=False)




