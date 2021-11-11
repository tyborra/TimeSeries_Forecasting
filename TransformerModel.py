# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 10:15:46 2021

@author: ME
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, Conv1D, Layer, TimeDistributed, MultiHeadAttention

##############################################################################
# https://towardsdatascience.com/the-time-series-transformer-2a521a0efad3


class Time2Vec(Layer):
    def __init__(self, kernel_size=1):
        super(Time2Vec, self).__init__(trainable=True, name='Time2VecLayer')
        self.k = kernel_size
    
    def build(self, input_shape):
        # trend
        self.wb = self.add_weight(name='wb',shape=(input_shape[1],),initializer='uniform',trainable=True)
        self.bb = self.add_weight(name='bb',shape=(input_shape[1],),initializer='uniform',trainable=True)
        # periodic
        self.wa = self.add_weight(name='wa',shape=(1, input_shape[1], self.k),initializer='uniform',trainable=True)
        self.ba = self.add_weight(name='ba',shape=(1, input_shape[1], self.k),initializer='uniform',trainable=True)
        super(Time2Vec, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        bias = self.wb * inputs + self.bb
        dp = K.dot(inputs, self.wa) + self.ba
        wgts = K.sin(dp) # or K.cos(.)

        ret = K.concatenate([K.expand_dims(bias, -1), wgts], -1)
        ret = K.reshape(ret, (-1, inputs.shape[1]*(self.k+1)))
        return ret
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]*(self.k + 1))


class AttentionBlock(K.Model):
    def __init__(self, name='AttentionBlock', num_heads=2, head_size=128, ff_dim=None, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)

        if ff_dim is None:
            ff_dim = head_size

        self.attention = MultiHeadAttention(num_heads=num_heads, head_size=head_size, dropout=dropout)
        self.attention_dropout = Dropout(dropout)
        self.attention_norm = LayerNormalization(epsilon=1e-6)

        self.ff_conv1 = Conv1D(filters=ff_dim, kernel_size=1, activation='relu')
        # self.ff_conv2 at build()
        self.ff_dropout = Dropout(dropout)
        self.ff_norm = LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.ff_conv2 = Conv1D(filters=input_shape[-1], kernel_size=1) 

    def call(self, inputs):
        x = self.attention([inputs, inputs])
        x = self.attention_dropout(x)
        x = self.attention_norm(inputs + x)

        x = self.ff_conv1(x)
        x = self.ff_conv2(x)
        x = self.ff_dropout(x)

        x = self.ff_norm(inputs + x)
        return x


class ModelTrunk(K.Model):
    def __init__(self, name='ModelTrunk', time2vec_dim=1, num_heads=2, head_size=128, ff_dim=None, num_layers=1, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.time2vec = Time2Vec(kernel_size=time2vec_dim)
        if ff_dim is None:
            ff_dim = head_size
        self.dropout = dropout
        self.attention_layers = [AttentionBlock(num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)]

    
    def call(self, inputs):
        time_embedding = TimeDistributed(self.time2vec)(inputs)
        x = K.concatenate([inputs, time_embedding], -1)
        for attention_layer in self.attention_layers:
            x = attention_layer(x)
    
        return K.reshape(x, (-1, x.shape[1] * x.shape[2])) # flat vector of features out
    
    
    

##############################################################################
# https://github.com/JanSchm/CapMarket/blob/master/bot_experiments/IBM_Transformer%2BTimeEmbedding.ipynb



class Time2Vector(Layer):
    
    def __init__(self, seq_len, **kwargs):
      super(Time2Vector, self).__init__()
      self.seq_len = seq_len
    
    def build(self, input_shape):
      '''Initialize weights and biases with shape (batch, seq_len)'''
      self.weights_linear = self.add_weight(name='weight_linear',
                                  shape=(int(self.seq_len),),
                                  initializer='uniform',
                                  trainable=True)
      
      self.bias_linear = self.add_weight(name='bias_linear',
                                  shape=(int(self.seq_len),),
                                  initializer='uniform',
                                  trainable=True)
      
      self.weights_periodic = self.add_weight(name='weight_periodic',
                                  shape=(int(self.seq_len),),
                                  initializer='uniform',
                                  trainable=True)
    
      self.bias_periodic = self.add_weight(name='bias_periodic',
                                  shape=(int(self.seq_len),),
                                  initializer='uniform',
                                  trainable=True)
    
    def call(self, x):
      '''Calculate linear and periodic time features'''
      x = tf.math.reduce_mean(x[:,:,:4], axis=-1) 
      time_linear = self.weights_linear * x + self.bias_linear # Linear time feature
      time_linear = tf.expand_dims(time_linear, axis=-1) # Add dimension (batch, seq_len, 1)
      
      time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
      time_periodic = tf.expand_dims(time_periodic, axis=-1) # Add dimension (batch, seq_len, 1)
      return tf.concat([time_linear, time_periodic], axis=-1) # shape = (batch, seq_len, 2)
     
    def get_config(self): # Needed for saving and loading model with custom layer
      config = super().get_config().copy()
      config.update({'seq_len': self.seq_len})
      return config
  
    

class SingleAttention(Layer):
    def __init__(self, d_k, d_v):
      super(SingleAttention, self).__init__()
      self.d_k = d_k
      self.d_v = d_v
    
    def build(self, input_shape):
      self.query = Dense(self.d_k, 
                         input_shape=input_shape, 
                         kernel_initializer='glorot_uniform', 
                         bias_initializer='glorot_uniform')
      
      self.key = Dense(self.d_k, 
                       input_shape=input_shape, 
                       kernel_initializer='glorot_uniform', 
                       bias_initializer='glorot_uniform')
      
      self.value = Dense(self.d_v, 
                         input_shape=input_shape, 
                         kernel_initializer='glorot_uniform', 
                         bias_initializer='glorot_uniform')
    
    def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
      q = self.query(inputs[0])
      k = self.key(inputs[1])
    
      attn_weights = tf.matmul(q, k, transpose_b=True)
      attn_weights = tf.map_fn(lambda x: x/np.sqrt(self.d_k), attn_weights)
      attn_weights = tf.nn.softmax(attn_weights, axis=-1)
      
      v = self.value(inputs[2])
      attn_out = tf.matmul(attn_weights, v)
      return attn_out
  
    
class MultiAttention(Layer):
  def __init__(self, d_k, d_v, n_heads):
    super(MultiAttention, self).__init__()
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.attn_heads = list()

  def build(self, input_shape):
    for n in range(self.n_heads):
      self.attn_heads.append(SingleAttention(self.d_k, self.d_v))  
    
    # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7 
    self.linear = Dense(input_shape[0][-1], 
                        input_shape=input_shape, 
                        kernel_initializer='glorot_uniform', 
                        bias_initializer='glorot_uniform')

  def call(self, inputs):
    attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
    concat_attn = tf.concat(attn, axis=-1)
    multi_linear = self.linear(concat_attn)
    return multi_linear   


class TransformerEncoder(Layer):
  def __init__(self, d_k, d_v, n_heads, ff_dim, dropout=0.1, **kwargs):
    super(TransformerEncoder, self).__init__()
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.ff_dim = ff_dim
    self.attn_heads = list()
    self.dropout_rate = dropout

  def build(self, input_shape):
    self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
    self.attn_dropout = Dropout(self.dropout_rate)
    self.attn_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    self.ff_conv1D_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
    # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1] = 7 
    self.ff_conv1D_2 = Conv1D(filters=input_shape[0][-1], kernel_size=1) 
    self.ff_dropout = Dropout(self.dropout_rate)
    self.ff_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)    
  
  def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
    attn_layer = self.attn_multi(inputs)
    attn_layer = self.attn_dropout(attn_layer)
    attn_layer = self.attn_normalize(inputs[0] + attn_layer)

    ff_layer = self.ff_conv1D_1(attn_layer)
    ff_layer = self.ff_conv1D_2(ff_layer)
    ff_layer = self.ff_dropout(ff_layer)
    ff_layer = self.ff_normalize(inputs[0] + ff_layer)
    return ff_layer 

  def get_config(self): # Needed for saving and loading model with custom layer
    config = super().get_config().copy()
    config.update({'d_k': self.d_k,
                   'd_v': self.d_v,
                   'n_heads': self.n_heads,
                   'ff_dim': self.ff_dim,
                   'attn_heads': self.attn_heads,
                   'dropout_rate': self.dropout_rate})
    return config