#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from textwrap import dedent
from Bio import SeqIO
import numpy as np
import random
import os
import time
import gc
import tensorflow as tf
from layers import Attention, ScaledDotProductAttention, MultiHeadAttention
from keras.models import Model
from keras.layers import Masking, Dense, LSTM, Bidirectional, Input, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score, confusion_matrix
from keras import backend as K

MAX_LEN = 200

def one_hot_padding(seq_list, padding):
    feat_list = []
    one_hot = {}
    aa = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    for i in range(len(aa)):
        one_hot[aa[i]] = [0]*20
        one_hot[aa[i]][i] = 1
    for i in range(len(seq_list)):
        feat = []
        for j in range(len(seq_list[i])):
            feat.append(one_hot[seq_list[i][j]])
        feat = feat + [[0]*20]*(padding-len(seq_list[i]))
        feat_list.append(feat)
    return np.array(feat_list)

def focal_loss(alpha, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return K.mean(-alpha_t * K.pow(1 - pt, gamma) * K.log(pt))
    return loss

def build_model(loss_fn):
    inputs = Input(shape=(MAX_LEN, 20), name='Input')
    masking = Masking(mask_value=0.0, input_shape=(MAX_LEN, 20), name='Masking')(inputs)
    hidden = Bidirectional(LSTM(512, use_bias=True, dropout=0.5, return_sequences=True), name='Bidirectional-LSTM')(masking)
    hidden = MultiHeadAttention(head_num=32, activation='relu', use_bias=True,
                                return_multi_attention=False, name='Multi-Head-Attention')(hidden)
    hidden = Dropout(0.2, name='Dropout_1')(hidden)
    hidden = Attention(name='Attention')(hidden)
    prediction = Dense(1, activation='sigmoid', name='Output')(hidden)
    model = Model(inputs=inputs, outputs=prediction)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=loss_fn, optimizer=adam, metrics=['accuracy'])
    return model

def main():
    parser = argparse.ArgumentParser(description=dedent('''
        AMPlify v3.0 training
        ------------------------------------------------------
        Given training sets with two labels: AMP and non-AMP,
        train the AMP prediction model.
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-amp_tr', required=True)
    parser.add_argument('-non_amp_tr', required=True)
    parser.add_argument('-amp_te', default=None, required=False)
    parser.add_argument('-non_amp_te', default=None, required=False)
    parser.add_argument('-sample_ratio', choices=['balanced', 'imbalanced'], default='balanced', required=False)
    parser.add_argument('-out_dir', required=True)
    parser.add_argument('-model_name', required=True)
    parser.add_argument('--fold_id', type=int, required=True)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load training sequences ───────────────────────────────────────
    AMP_train, non_AMP_train = [], []
    for seq_record in SeqIO.parse(args.amp_tr, 'fasta'):
        seq = str(seq_record.seq)
        AMP_train.append(seq[:-1] if seq[-1] == '*' else seq)
    for seq_record in SeqIO.parse(args.non_amp_tr, 'fasta'):
        seq = str(seq_record.seq)
        non_AMP_train.append(seq[:-1] if seq[-1] == '*' else seq)

    train_seq = AMP_train + non_AMP_train
    y_train = np.array([1]*len(AMP_train) + [0]*len(non_AMP_train))

    train = list(zip(train_seq, y_train))
    random.Random(123).shuffle(train)
    train_seq, y_train = zip(*train)
    train_seq = list(train_seq)
    y_train = np.array(y_train)
    np.save(f"{args.out_dir}/y_train_labels.npy", y_train)

    X_train = one_hot_padding(train_seq, MAX_LEN)

    # ── Load test sequences if provided ──────────────────────────────
    if args.amp_te is not None and args.non_amp_te is not None:
        AMP_test, non_AMP_test = [], []
        for seq_record in SeqIO.parse(args.amp_te, 'fasta'):
            seq = str(seq_record.seq)
            AMP_test.append(seq[:-1] if seq[-1] == '*' else seq)
        for seq_record in SeqIO.parse(args.non_amp_te, 'fasta'):
            seq = str(seq_record.seq)
            non_AMP_test.append(seq[:-1] if seq[-1] == '*' else seq)
        test_seq = AMP_test + non_AMP_test
        y_test = np.array([1]*len(AMP_test) + [0]*len(non_AMP_test))
        np.save(f"{args.out_dir}/y_test_labels.npy", y_test)
        X_test = one_hot_padding(test_seq, MAX_LEN)

    # ── Fold split ────────────────────────────────────────────────────
    ensemble = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)
    for i, (tr_idx, te_idx) in enumerate(ensemble.split(X_train, y_train)):
        if i == args.fold_id:
            X_tr_fold, X_val_fold = X_train[tr_idx], X_train[te_idx]
            y_tr_fold, y_val_fold = y_train[tr_idx], y_train[te_idx]

            # Compute loss — alpha derived from fold training split only (no leakage)
            if args.sample_ratio == 'balanced':
                loss_fn = 'binary_crossentropy'
            else:
                n_pos = int(y_tr_fold.sum())
                n_neg = len(y_tr_fold) - n_pos
                alpha = n_neg / (n_pos + n_neg)
                loss_fn = focal_loss(alpha=alpha, gamma=2.0)
                print(f"Fold {args.fold_id} alpha: {alpha:.4f}")
            break

    # ── Train ─────────────────────────────────────────────────────────
    gc.collect()
    model = build_model(loss_fn)
    start = time.time()

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50,
                                   mode='min', restore_best_weights=True)
    model.fit(X_tr_fold, y_tr_fold, epochs=1000, batch_size=32,
              validation_data=(X_val_fold, y_val_fold), verbose=2,
              callbacks=[early_stopping], class_weight=None)

    print(f"Fold {args.fold_id} training time: {time.time() - start:.2f} seconds")
    val_loss, val_acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    print(f"Fold {args.fold_id} validation loss: {val_loss}")

    # ── Save model and train predictions ──────────────────────────────
    temp_pred_train = model.predict(X_train).flatten()
    np.save(f"{args.out_dir}/temp_pred_train{args.fold_id}.npy", temp_pred_train)
    model.save(f"{args.out_dir}/{args.model_name}_{args.fold_id}.h5")
    model.save_weights(f"{args.out_dir}/{args.model_name}_weights_{args.fold_id}.h5")

    # ── Threshold from validation fold ───────────────────────────────
    y_val_prob = model.predict(X_val_fold).flatten()
    precision, recall, thresholds_pr = precision_recall_curve(y_val_fold, y_val_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    threshold = thresholds_pr[np.argmax(f1_scores[:-1])]
    np.save(f"{args.out_dir}/optimal_threshold_{args.fold_id}.npy", threshold)
    print(f"Fold {args.fold_id} optimal threshold (F1): {threshold:.4f}")

    # ── Per-fold metrics ──────────────────────────────────────────────
    temp_pred_class_train_curr = (model.predict(X_tr_fold).flatten() > threshold).astype(int)
    temp_pred_class_val        = (y_val_prob > threshold).astype(int)

    print('*************************** current model ***************************')
    print('current train acc: ', accuracy_score(y_tr_fold, temp_pred_class_train_curr))
    print('current val acc:   ', accuracy_score(y_val_fold, temp_pred_class_val))

    if args.amp_te is not None and args.non_amp_te is not None:
        temp_pred_test = model.predict(X_test).flatten()
        np.save(f"{args.out_dir}/temp_pred_test{args.fold_id}.npy", temp_pred_test)
        temp_pred_class_test = (temp_pred_test > threshold).astype(int)
        tn_indv, fp_indv, fn_indv, tp_indv = confusion_matrix(y_test, temp_pred_class_test).ravel()
        print('test acc:  ', accuracy_score(y_test, temp_pred_class_test))
        print('test sens: ', tp_indv/(tp_indv+fn_indv))
        print('test spec: ', tn_indv/(tn_indv+fp_indv))
        print('test f1:   ', f1_score(y_test, temp_pred_class_test))
        print('test roc_auc: ', roc_auc_score(y_test, temp_pred_test))

    print('*********************************************************************')

if __name__ == "__main__":
    main()