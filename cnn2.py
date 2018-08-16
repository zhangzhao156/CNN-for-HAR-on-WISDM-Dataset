import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn import metrics
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
columns = ['user','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
df = pd.read_csv('WISDM_ar_v1.1_raw.txt', header = None, names = columns)
df = df.dropna()
df.head()
df.info()
N_TIME_STEPS = 90
N_FEATURES = 3
step = 20
segments = []
labels = []
for i in range(0, len(df) - N_TIME_STEPS, step):
    xs = df['x-axis'].values[i: i + N_TIME_STEPS]
    ys = df['y-axis'].values[i: i + N_TIME_STEPS]
    zs = df['z-axis'].values[i: i + N_TIME_STEPS]
    label = stats.mode(df['activity'][i: i + N_TIME_STEPS])[0][0]
    segments.append([xs, ys, zs])
    labels.append(label)

reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)

X_train, X_test, y_train, y_test = train_test_split(
        reshaped_segments, labels, test_size=0.2, random_state=RANDOM_SEED)

num_labels = 6
BATCH_SIZE = 64
LEARNING_RATE = 0.0025
N_EPOCHS = 50

X = tf.placeholder(tf.float32, shape=[None,N_TIME_STEPS,N_FEATURES], name="input")
Y = tf.placeholder(tf.float32, shape=[None,num_labels])
conv1 = tf.layers.conv1d(inputs=X, filters=64, kernel_size=5, strides=1, padding='same', activation = tf.nn.relu)
pool1=tf.layers.max_pooling1d(inputs=conv1,pool_size=5,strides=2,padding='same')
conv2=tf.layers.conv1d(inputs=pool1, filters=32, kernel_size=5, strides=1, padding='same', activation = tf.nn.relu)
flat=tf.layers.flatten(inputs=conv2)
logits=tf.layers.dense(inputs=flat,units=6,activation=tf.nn.relu,name="y_")
#logits = tf.nn.softmax(pred,name="y_")
#loss = -tf.reduce_sum(Y * tf.log(y_))
L2_LOSS = 0.0015

l2 = L2_LOSS * \
    sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y)) + l2
#loss = -tf.reduce_sum(Y * tf.log(logits)) + l2
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE).minimize(loss)

correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cost_history = np.empty(shape=[1],dtype=float)

saver = tf.train.Saver()

history = dict(train_loss=[], 
                     train_acc=[], 
                     test_loss=[], 
                     test_acc=[])

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
#saver.restore(sess, "./checkpoint/har.ckpt")
train_count = len(X_train)

for i in range(1, N_EPOCHS + 1):
    for start, end in zip(range(0, train_count, BATCH_SIZE),
                          range(BATCH_SIZE, train_count + 1,BATCH_SIZE)):
        sess.run(optimizer, feed_dict={X: X_train[start:end],
                                       Y: y_train[start:end]})

    _, acc_train, loss_train = sess.run([logits, accuracy, loss], feed_dict={
                                            X: X_train, Y: y_train})

    _, acc_test, loss_test = sess.run([logits, accuracy, loss], feed_dict={
                                            X: X_test, Y: y_test})

    history['train_loss'].append(loss_train)
    history['train_acc'].append(acc_train)
    history['test_loss'].append(loss_test)
    history['test_acc'].append(acc_test)

    # if i != 1 and i % 10 != 0:
        # continue
    #print(f'epoch: {i} train accuracy: {acc_train} loss: {loss_train}')
    print(f'epoch: {i} test accuracy: {acc_test} loss: {loss_test}')
    
predictions, acc_final, loss_final = sess.run([logits, accuracy, loss], feed_dict={X: X_test, Y: y_test})

print()
print(f'final results: accuracy: {acc_final} loss: {loss_final}')
pickle.dump(predictions, open("predictions2.p", "wb"))
pickle.dump(history, open("history2.p", "wb"))
#tf.train.write_graph(sess.graph_def, '.', './checkpoint2/har.pbtxt')  
#saver.save(sess, save_path = "./checkpoint2/har.ckpt")
sess.close()