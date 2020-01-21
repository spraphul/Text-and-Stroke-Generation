import numpy as np
import numpy
import re
import os
import pickle as pkl
import tensorflow as tf
from matplotlib import pyplot
from random import shuffle
from collections import Counter 
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K



strokes = np.load('strokes-py3.npy', allow_pickle=True)
with open('sentences.txt') as f:
    texts = f.readlines()
    
    
 
#    Used from the link:
# https://github.com/thushv89/attention_keras/blob/f7c6f40cb207431d0229c38992eb93ad17d38e20/layers/attention.py#L7

class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state """

            assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch_size*en_seq_len, latent_dim
            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
            # <= batch_size*en_seq_len, latent_dim
            W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
            if verbose:
                print('wa.s>',W_a_dot_s.shape)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>',U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
            if verbose:
                print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """
            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]

        def create_inital_state(inputs, hidden_size):
            # We are not using initial states, but need to pass something to K.rnn funciton
            fake_state = K.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim
            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
            fake_state = K.tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim
            return fake_state

        fake_state_c = create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])
        fake_state_e = create_inital_state(encoder_out_seq, encoder_out_seq.shape[1])  # <= (batch_size, enc_seq_len, latent_dim

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]
    
    def get_config(self):
        base_config = super(AttentionLayer, self).get_config()
        return base_config
        
 
 def clean(string):
    string = string.lstrip().rstrip()
    string = string.strip()  # remove leading whitespaces
    string = string.lower()  # for lowercase
    string = re.sub(r"\\n", "", string)
    string = re.sub(r"\n", "", string)
    string = re.sub(r"[?]+", "", string)
    string = re.sub(r"[?]+", " ? ", string)  # replace more than one ? with a single ?
    string = re.sub(r"won't", "will not", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"n't", " not ", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'s", " is", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r"\'t", " not", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"\'m", " am", string)
    string = re.sub(r"\'d've", " would have", string)
    string = re.sub(r"\'d'y", " do you", string)
    string = re.sub('[!"#$%&\'()*+,-./:;<=>[\\]^_`{|}~]+', " ", string)
    string = re.sub('[\\\\]+', ' ', string)
    string = re.sub(r"[?]+", " ? ", string)
    string = re.sub(r"[ ]+", " ", string)  # replace more than one spaces with a single space
    string = string.strip()
    return string
    

def prepare_text(text,char2id):
    text = clean(text)
    text = list(text)
    text = [char2id[char] for char in text]
    return text
 
f = open('char2id', 'rb')
char2id = pkl.load(f)
f.close()

f = open('vocab', 'rb')
vocab = pkl.load(f)
f.close()


for i in range(len(texts)):
    texts[i] = prepare_text(texts[i], char2id)
    
    
padded_data=[]
for i in range(len(texts)):
    t = []
    for j in range(65):
        if(len(texts[i])>j):
            t.append(texts[i][j])
        else:
            t.append(0)
    padded_data.append(t)
    
def plot_stroke(stroke, save_name=None):
    # Plot a single example.
    f, ax = pyplot.subplots()

    x = numpy.cumsum(stroke[:, 1])
    y = numpy.cumsum(stroke[:, 2])

    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = numpy.where(stroke[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=3)
        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        pyplot.show()
    else:
        try:
            pyplot.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print("Error building image!: " + save_name)

    pyplot.close()



def addbias(a):
    return a+1
    
    

def build_model():
    stroke_inp = keras.layers.Input(shape=(None, 3), dtype='float32')
    lstm1 = keras.layers.LSTM(128, return_sequences=True)(stroke_inp)

    attention = keras.layers.Dense(1, activation='tanh')(lstm1)
    attention = keras.layers.Flatten()(attention)
    attention = keras.layers.Activation('softmax')(attention)
    attention = keras.layers.Lambda(addbias)(attention)
    attention = keras.layers.RepeatVector(128)(attention)
    attention = keras.layers.Permute([2, 1])(attention)
    alstm1 = keras.layers.multiply([lstm1, attention])
    alstm1,h,c = keras.layers.LSTM(128, return_sequences=True, return_state=True)(alstm1)
    final_state=[h,c]

    text_inp = keras.layers.Input(shape=(65,), dtype='int32')
    embedding_layer = keras.layers.Embedding(len(vocab)+1, 32, input_length=65)
    text_seq = embedding_layer(text_inp)
    text_lstm1,_,_ = keras.layers.LSTM(128, return_sequences=True, return_state=True)(text_seq, initial_state=final_state)
    text_lstm2 = keras.layers.LSTM(128, return_sequences=True)(text_lstm1)
    text_attention = keras.layers.Dense(1, activation='tanh')(text_lstm1)
    text_attention = keras.layers.Flatten()(text_attention)
    text_attention = keras.layers.Activation('softmax')(text_attention)
    text_attention = keras.layers.Lambda(addbias)(text_attention)
    text_attention = keras.layers.RepeatVector(128)(text_attention)
    text_attention = keras.layers.Permute([2, 1])(text_attention)
    text_alstm = keras.layers.multiply([text_lstm2, text_attention])
    text_prediction = keras.layers.TimeDistributed(keras.layers.Dense(len(vocab)+1, activation='softmax'))(text_alstm)
    
    
    model = keras.models.Model([stroke_inp, text_inp], text_prediction)
    return model
    

x = []
for i in strokes:
    x.append(i[0:-1].reshape( -1, 3))
    
    

with tf.Session(graph=tf.Graph()) as sess:
        K.set_session(sess)
        with tf.device('/device:GPU:3'):
            model = build_model()
            adam = tf.keras.optimizers.Adam(lr=0.06)
            model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')
            for batch in range(10):
                print("-------"+str(batch)+"-------- \n \n")

                for i in range(len(x)-1):
                    verbose = False
                    if(i%50==0):
                        keras.models.save_model(model, 'stroke2text.h5')
                        verbose =True
                    model.fit([x[i:i+1], padded_data[i:i+1]], [padded_data[i:i+1]], batch_size=1, epochs=1, verbose=verbose)

