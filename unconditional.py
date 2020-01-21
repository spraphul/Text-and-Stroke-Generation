import numpy as np
import numpy
import tensorflow as tf
from matplotlib import pyplot
from random import shuffle
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model


strokes = np.load('strokes-py3.npy', allow_pickle=True)
with open('../data/sentences.txt') as f:
    texts = f.readlines()

def plot_stroke(stroke, save_name=None):
    # Plot a single example.
    f, ax = pyplot.subplots()

    x = numpy.cumsum(stroke[:, 1])
    y = numpy.cumsum(stroke[:, 2])

    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = numpy.where(stroke[:, 0] == 1)[0]
    if(len(cuts)==0):
        cuts = np.asarray([10, 20, 50, 80])
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
    

def seqloss():
    def pdf(x1, x2, mu1, mu2, s1, s2, rho):
        norm1 = tf.subtract(x1, mu1)
        norm2 = tf.subtract(x2, mu2)
        s1s2 = tf.multiply(s1, s2)
        z = tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) - \
            2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2)
        negRho = 1 - tf.square(rho)
        result = tf.exp(tf.div(-z, 2 * negRho))
        denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(negRho))
        result = tf.div(result, denom)
        return result
    
    def loss(y_true, pred):
        mean1 = pred[0][:,0]; mean2 = pred[0][:,1]; sigma1= pred[0][:,2]
        sigma2 = pred[0][:,3]; rho = pred[0][:,4]; prob = pred[0][:,5]
        x = y_true[0][:,1]; y = y_true[0][:,2]; penlifts = y_true[0][:,0]
        loss1 = tf.math.reduce_mean(-tf.log(tf.maximum(tf.multiply(pdf(x, y, mean1, mean2, sigma1, sigma2, rho), K.epsilon()), prob)))
        pos = tf.multiply(prob, penlifts)
        neg = tf.multiply(1-prob, 1-penlifts)
        loss2 = tf.math.reduce_mean(-tf.log(pos+neg))
        final_loss = loss1+loss2
        
        return final_loss
    
    
    return loss     
    

def build_model():
    ninp = 3
    ndist = 3
    nout = 6

    inp = keras.layers.Input(shape=(None,ninp), dtype='float32')
    lstm1 = keras.layers.LSTM(32, return_sequences=True)(inp)

    attention = keras.layers.Dense(1, activation='tanh')(lstm1)
    attention = keras.layers.Flatten()(attention)
    attention = keras.layers.Activation('softmax')(attention)
    attention = keras.layers.Lambda(addbias)(attention)
    attention = keras.layers.RepeatVector(32)(attention)
    attention = keras.layers.Permute([2, 1])(attention)
    alstm1 = keras.layers.multiply([lstm1, attention])

    lstm2 = keras.layers.LSTM(64, return_sequences=True)(alstm1)

    mean1 = keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid'), name='mean1')(lstm2)
    mean2 = keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid'), name='mean2')(lstm2)
    sigma1 = keras.layers.TimeDistributed(keras.layers.Dense(1, activation='exponential'), name='sigma1')(lstm2)
    sigma2 = keras.layers.TimeDistributed(keras.layers.Dense(1, activation='exponential'), name='sigma2')(lstm2)
    rho = keras.layers.TimeDistributed(keras.layers.Dense(1, activation='tanh'), name='rho')(lstm2)
    pi = keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid'), name='prob')(lstm2)
    output = keras.layers.concatenate([mean1, mean2, sigma1, sigma2, rho, pi])
    model = Model(inp, output)
    return model 
    

x = []
for i in strokes:
    x.append(i[0:-1].reshape(1, -1, 3))
    
y = []
for i in strokes:
    y.append(i[1:].reshape(1,-1,3))
    


with tf.Session(graph=tf.Graph()) as sess:
        K.set_session(sess)
        with tf.device('/device:GPU:2'):
            model = build_model()
            adam = tf.keras.optimizers.Adam(lr=0.06)
            model.compile(loss=seqloss(),optimizer='adam')
            for epoch in range(10):
                print("-------"+str(epoch)+"--------")
                
                for i in range(len(x)-1):
                    verbose = False
                    if(i%50==0):
                        keras.models.save_model(model, 'unconditional.h5')
                        verbose=True
                    
                    model.fit(x[i:i+1], y[i], batch_size=1, epochs=1,verbose=verbose)
                    
def predprob(x):
    if(x>=0.5):
        return 0
    else:
        return 1


def gen_strokes(model, size):
    stroke = np.asarray([[[0,0,0]]])
    for i in range(size):
        print(i)
        pred = model.predict(stroke)[0][0]
        cov = [[pred[2] * pred[2], pred[4] * pred[2] * pred[3]], [pred[4] * pred[2] * pred[3], pred[3] * pred[3]]]
        mean = [pred[0], pred[1]]
        sample = s = np.random.multivariate_normal(mean, cov, 1)
        x,y = sample[0][0], sample[0][1]
        z = predprob(pred[5])
        stroke = np.vstack((stroke[0], [z,x,y]))
        stroke = stroke.reshape(1,-1,3)
    
    return stroke[0]
