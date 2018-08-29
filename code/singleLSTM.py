import tensorflow as tf
from tensorflow.contrib import rnn
from dataset import loadingDB
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import pickle
import pandas as pd

setnum = 1#1 is opp79, 2 is pamap2, 3 is skoda
if setnum == 1:
	train_x, valid_x, test_x, train_y, valid_y, test_y = loadingDB('../', 79)
	n_classes = 18
	DB = 79
if setnum == 2:
	train_x, valid_x, test_x, train_y, valid_y, test_y = loadingDB('../', 52)
	n_classes = 12
	DB = 52
if setnum == 3:
	train_x, valid_x, test_x, train_y, valid_y, test_y = loadingDB('../', 60)
	n_classes = 11
	DB = 60

# set hyperparameters of structure, dim and DB are same
nm_epochs = 100
rnn_size = 256
number_of_layers = 2
keep_rate = 0.5
dim = len(train_x[0])
valid_len = len(valid_x)
test_len = len(test_x)

#####################################################
# We can change valid and test batch size to increase the speed of training process.
# In test and valid process, they has same window length.
test_win = 5000
test_bt = 1
test_se = test_len//test_bt
test_x = test_x[:test_se*test_bt,]
test_y = np.array(test_y)
test_y = test_y[:test_se*test_bt,]
test_x = np.reshape(test_x, (test_bt,test_se,-1))
test_y = np.reshape(test_y, (test_bt,test_se,-1))
print('test shape' ,test_x.shape)
print('test shape' ,test_y.shape)

valid_bt = 1
valid_se = valid_len//valid_bt
valid_x = valid_x[:valid_se*valid_bt,]
valid_y = np.array(valid_y)
valid_y = valid_y[:valid_se*valid_bt,]
valid_x = np.reshape(valid_x, (valid_bt,valid_se,-1))
valid_y = np.reshape(valid_y, (valid_bt,valid_se,-1))
print('valid shape' ,valid_x.shape)
print('valid shape' ,valid_y.shape)
#####################################################

# set fix shape of input(x), target(y), dropout_keep_prob and state
# shape of x and y: [batch_size,window_len,dim]
x = tf.placeholder('float', [None, None, dim])
y = tf.placeholder('float', [None, None, n_classes])
dropout_keep_prob = tf.placeholder(tf.float32, [])
# states shape: [number_of_layers,2,batch_size,rnn_size], number 2 means 'c' and 'h' in LSTM.
# motivated by: https://stackoverflow.com/questions/39112622/how-do-i-set-tensorflow-rnn-state-when-state-is-tuple-true?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
states = tf.placeholder('float', [number_of_layers, 2, None, rnn_size])
l = tf.unstack(states, axis=0)
rnn_tuple_state = tuple(
		 [tf.nn.rnn_cell.LSTMStateTuple(l[idx][0],l[idx][1])
		  for idx in range(number_of_layers)]
)

######################################################
# The structure of lstm base learners
def lstm_cell(dropout_keep_prob):

	return tf.contrib.rnn.DropoutWrapper(rnn.BasicLSTMCell(rnn_size), input_keep_prob=1, output_keep_prob=dropout_keep_prob, state_keep_prob=dropout_keep_prob)

def recurrent_neural_network (x,rnn_tuple_state,dropout_keep_prob):

	stacked_lstm = rnn.MultiRNNCell([lstm_cell(dropout_keep_prob) for _ in range(number_of_layers)])
	x = tf.transpose(x, [1,0,2])
	outputs, states = tf.nn.dynamic_rnn(stacked_lstm, x, initial_state=rnn_tuple_state, dtype=tf.float32, time_major=True)
	output = tf.layers.dense(inputs=outputs, units=n_classes)# use this to solve random window length problem
	output = tf.transpose(output, [1, 0, 2])

	return output, states
######################################################

# motivated by ensemble of deep LSTM learners.
def making_training_set(train_x, train_y, batch_size=100):
	print('Batch size is: ', batch_size)
	seqence_len = len(train_x)//batch_size
	
	# the default one
	indices_start = np.random.randint(low=0, high=len(train_x)-seqence_len, size=(batch_size,))
	
	## calculate  ## coverage 
	indices_all_2d = np.zeros((batch_size, seqence_len))
	for i in range(batch_size):
		indices_all_2d[i,:] = np.arange(indices_start[i],indices_start[i]+seqence_len)
	indices_all = np.reshape(indices_all_2d, (-1))

	coverage = 100*len(np.unique(indices_all))/len(indices_all)#https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html

	X_train3D = np.zeros((batch_size, seqence_len, dim), dtype=np.float32)
	y_train3D = np.zeros((batch_size, seqence_len, n_classes), dtype=np.uint8) 
	for i in range(batch_size):
		idx_start = indices_start[i]
		idx_end = idx_start+seqence_len
		X_train3D[i,:,:] = train_x[idx_start:idx_end, :]
		# y_train3D[i,:,:] = train_y[idx_start:idx_end, :]# if train_y is not a dataframe
		y_train3D[i,:,:] = train_y.iloc[idx_start:idx_end, :]# if train_y is a pandas dataframe

	return X_train3D, y_train3D, coverage

# evaluate on validation and test set
def validation(epoch,sess,cost, accuracy_train, state, predicted, actual, saver):
	print("Testing model performance on validation set:")
	num_val_process = valid_se//test_win + 1
	val_losses     = np.zeros(num_val_process, dtype=np.float32)
	val_accuracies = np.zeros(num_val_process, dtype=np.float32)
	predictvalue = []
	actualvalue = []
	valid_ini = np.zeros((number_of_layers, 2, valid_bt, rnn_size))
	cost_total = 0

	for j in range(num_val_process):

		start = j*test_win
		end = np.min((valid_se, start+test_win))

		feed_dict = {
			x: valid_x[:,start:end,:],
			y: valid_y[:,start:end,:],
			states: valid_ini,
			dropout_keep_prob: 1.0
		}

		# t.eval() is a shortcut for calling in the evaluate and test stage
		val_loss, val_acc, valid_ini, valueP, valueA = sess.run([cost, accuracy_train, state, predicted, actual], feed_dict=feed_dict)

		valueP = np.reshape(np.array(valueP), (-1))
		valueA = np.reshape(np.array(valueA), (-1))
		predictvalue.extend(valueP)
		actualvalue.extend(valueA)

		cost_total += val_loss

		val_losses[j]     = val_loss
		val_accuracies[j] = val_acc

	eval_cost = cost_total/num_val_process#same with np.mean(val_losses)
	predictvalue = np.reshape(predictvalue, (-1))
	actualvalue = np.reshape(actualvalue, (-1))
	f1_value = f1_score(actualvalue, predictvalue, average='macro')
	confusion_value = confusion_matrix(actualvalue, predictvalue)
	print("  Validation F1: %.3f" % f1_value)
	print("  Validation Confusion Matrix: %.3f")
	print(confusion_value)

	print("  Average Validation Accuracy: %.3f" % np.mean(val_accuracies))
	print("  Average Validation Loss:     %.3f" % np.mean(val_losses))

	print("Run in test data:")
	num_test_process = test_se//test_win + 1
	test_losses     = np.zeros(num_test_process, dtype=np.float32)
	test_accuracies = np.zeros(num_test_process, dtype=np.float32)
	test_predictvalue = []
	test_actualvalue = []
	test_ini = np.zeros((number_of_layers, 2, test_bt, rnn_size))
	j = 0
	testcost_total = 0

	for j in range(num_test_process):

		start = j*test_win
		end = np.min((test_se, start+test_win))

		feed_dict = {
			x: test_x[:,start:end,:],
			y: test_y[:,start:end,:],
			states: test_ini,
			dropout_keep_prob: 1.0
		}

		test_loss, test_acc, test_ini, test_valueP, test_valueA = sess.run([cost, accuracy_train, state, predicted, actual], feed_dict=feed_dict)

		test_valueP = np.reshape(np.array(test_valueP), (-1))
		test_valueA = np.reshape(np.array(test_valueA), (-1))
		test_predictvalue.extend(test_valueP)
		test_actualvalue.extend(test_valueA)

		testcost_total += test_loss

		test_losses[j]     = test_loss
		test_accuracies[j] = test_acc

	test_cost = testcost_total/num_test_process# same with np.mean(test_losses)
	test_predictvalue = np.reshape(test_predictvalue, (-1))
	test_actualvalue = np.reshape(test_actualvalue, (-1))
	test_f1_value = f1_score(test_actualvalue, test_predictvalue, average='macro')
	test_confusion_value = confusion_matrix(test_actualvalue, test_predictvalue)
	print("  Test F1: %.3f" % test_f1_value)
	print("  Test Confusion Matrix: %.3f")
	print(test_confusion_value)
	print("  Average Test Accuracy: %.3f" % np.mean(test_accuracies))
	print("  Average Test Loss:     %.3f" % np.mean(test_losses))

	return f1_value, eval_cost, test_f1_value, test_cost


prediction, state = recurrent_neural_network(x,rnn_tuple_state,dropout_keep_prob)# put there as the structure only can defined once
def train_recurrent_neural_network (x, range_mb=[128, 256], range_win=[16, 32], model_name=''):

	print('The range of batch size is: ', range_mb)
	print('The range of window length is: ', range_win)

	# cross entropy loss function need one-hot
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y) )#need specific
	#							default learning_rate is 0.001
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)#https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer

	# 2 means select from second axis as the predict shape in here is: [batch_size, window_len, dim]
	predicted = tf.argmax(prediction,2)
	actual = tf.argmax(y,2)
	correct = tf.equal(predicted, actual)
	accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
	# another way to calculate accuracy
	accuracy_train = tf.contrib.metrics.accuracy(predicted, actual)

	test_ini = np.zeros((number_of_layers, 2, valid_bt, rnn_size))

	# to save tensorflow model 
	tf_vars_to_save = tf.trainable_variables()
	saver = tf.train.Saver(tf_vars_to_save, max_to_keep=99999)

	results = np.empty([0, 5], dtype=np.float32) # train_err, valid_cost, test_cost, val_f1, test_f1

	#######################################################
	# if using GPU, add these lines
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
    #######################################################

	# with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())#tf.initialize_all_variables() not use anymore

		for epoch in range(nm_epochs):

			epoch_wise_results = np.zeros(5)
			epoch_loss = 0
			train_loss = 0
			train_sz = 0 #the sample size during training process
			i = 0

			# generate a random batch size at the beginning of each epoch
			batch_size = np.random.randint(low=range_mb[0], high=range_mb[1], size=1)[0]
			train_x3D, train_y3D, coverage = making_training_set(train_x, train_y, batch_size)
			train_len = len(train_x)//batch_size # train_x3D shape: [batch_size,train_len,dim]
			print('train_len: ', train_len)

			state_ini = np.zeros((number_of_layers, 2, batch_size, rnn_size))

			pos_start = 0
			pos_end = 0

			while pos_end < train_len:

				pos_start = pos_end
				# generate a random window length in each training process
				curr_win_len = np.random.randint(low=range_win[0], high=range_win[1], size=1)[0]
				pos_end += curr_win_len

				inputs = train_x3D[:,pos_start:pos_end,:]
				targets = train_y3D[:,pos_start:pos_end,:]

				# updata state_ini after each run, so it becomes to sample-wise prediction(stateful)
				_, c, state_ini, train_acc = sess.run([optimizer, cost, state, accuracy_train], feed_dict={x: inputs, y: targets, states: state_ini, dropout_keep_prob: keep_rate})
				epoch_loss += c

				sample_sz = batch_size*curr_win_len
				train_loss += c*sample_sz
				train_sz += sample_sz

				i += 1
				if i % 20 == 0:
					print("Traing %06i, Train Accuracy = %.2f, Train Loss = %.3f" % (pos_end*batch_size, train_acc, c))

			epoch_wise_results[0] = train_loss/train_sz
			valid_f1, valid_loss, test_f1, test_loss = validation(epoch+1,sess,cost, accuracy_train, state, predicted, actual, saver)
			epoch_wise_results[1] = valid_loss
			epoch_wise_results[2] = test_loss        
			epoch_wise_results[3] = valid_f1
			epoch_wise_results[4] = test_f1

			results = np.float32(np.vstack((results, epoch_wise_results)))

			print('saving results ...')
		
			np.save('results/'+model_name+'_'+str(DB)+'.npy', results)

			print('Epoch', epoch+1, 'completed out of',nm_epochs,'loss:',epoch_loss)
			print('Epoch', epoch+1, 'completed out of',nm_epochs,'coverage:',coverage)

			# only save models after 10 epoch
			if epoch>=10: 
				path = saver.save(sess, './model/'+model_name+'_'+str(DB)+'_'+str(epoch))
				print("Model saved to %s" % path)

for t in range(30):
	trial = t 
	model_name = 'T_'+str(trial)+'_CE'
	print(model_name)
	train_recurrent_neural_network(x, range_mb=[128, 256], range_win=[16, 32], model_name=model_name)
print('Traing completed') 


