# motivated by ensemble of deep lstm learners
import tensorflow as tf
from tensorflow.contrib import rnn
from dataset import loadingDB
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

setnum = 1#1 is opp, 2 is pam, 3 is skoda
if setnum == 1:
	train_x, valid_x, test_x, train_y, valid_y, test_y = loadingDB('../', 79)
	n_classes = 18
	DB = 79
	csv = open('79.csv','a')
if setnum == 2:
	train_x, valid_x, test_x, train_y, valid_y, test_y = loadingDB('../', 52)
	n_classes = 12
	DB = 52
	csv = open('52.csv','a')
if setnum == 3:
	train_x, valid_x, test_x, train_y, valid_y, test_y = loadingDB('../', 60)
	n_classes = 11
	DB =60
	csv = open('60.csv','a')

print('test shape', np.shape(test_x))
print('test shape', np.shape(test_y))
dim = len(test_x[0])

#####################################################
# can change valid and test batch size to increase the speed of training process, but it will effect the results
# especially for balanced dataset.
test_win = 5000
test_bt = 1
test_se = len(test_x)//test_bt
test_x = test_x[:test_se*test_bt,]
lable_len = test_len = len(test_x)
test_y = np.array(test_y)
test_y = test_y[:test_se*test_bt,]
test_x = np.reshape(test_x, (test_bt,test_se,-1))
test_y = np.reshape(test_y, (test_bt,test_se,-1))
print('test new shape' ,test_x.shape)
print('test new shape' ,test_y.shape)
#####################################################

rnn_size = 256
number_of_layers = 2

# set fix shape
x = tf.placeholder('float', [None, None, dim])
y = tf.placeholder('float', [None, None, n_classes])
dropout_keep_prob = tf.placeholder(tf.float32, [])
soft_max = tf.placeholder('float',[None, n_classes])

states = tf.placeholder('float', [number_of_layers, 2, None, rnn_size])
l = tf.unstack(states, axis=0)
rnn_tuple_state = tuple(
		 [tf.nn.rnn_cell.LSTMStateTuple(l[idx][0],l[idx][1])
		  for idx in range(number_of_layers)]
)

def lstm_cell(dropout_keep_prob):

	return tf.contrib.rnn.DropoutWrapper(rnn.BasicLSTMCell(rnn_size), input_keep_prob=1, output_keep_prob=dropout_keep_prob, state_keep_prob=dropout_keep_prob)#https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell

def recurrent_neural_network (x,rnn_tuple_state,dropout_keep_prob):
	stacked_lstm = rnn.MultiRNNCell([lstm_cell(dropout_keep_prob) for _ in range(number_of_layers)])

	x = tf.transpose(x, [1,0,2])
	outputs, states = tf.nn.dynamic_rnn(stacked_lstm, x, initial_state=rnn_tuple_state, dtype=tf.float32, time_major=True)
	output = tf.layers.dense(inputs=outputs, units=n_classes)
	output = tf.transpose(output, [1, 0, 2])

	return output, states

def f1_Ranking(results, shown_TopN = 30, valid_col=3):
   
	print('valid_col', valid_col)
	test_col = valid_col+1
	idx_set = np.argsort(results[:,valid_col])[::-1]#https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html

	valid_set = np.zeros(shown_TopN)
	test_set = np.zeros(shown_TopN)
	for i in range(shown_TopN):
		
		valid_f1 = results[idx_set[i], valid_col]# as epoch_wise_results[3] = f1_valid
		test_f1 = results[idx_set[i],test_col]# epoch_wise_results[4] = f1_test
		idx = idx_set[i]
		
		valid_set[i] = valid_f1
		test_set[i] = test_f1
		
		if idx>=10:
			print('idx {}, f1_valid: {:.3f}, f1_test: {:.3f}'.format(idx, valid_f1, test_f1))  

	return idx_set, valid_set, test_set  
	
def exp_setting(dataset, exp_id, shown_TopN=30):
	# shown_TopN=30 when doesn't delever the parameter
	a = np.load('results/'+exp_id+'.npy')
	idx_set, valid_set_a, test_set_a = f1_Ranking(a, shown_TopN)
	# select the location of best model, idx_set>=10 as only saved models with epoch>=10
	Best_idx = idx_set[idx_set>=10]

	exp = []
	exp.append(Best_idx[:bestM])
	exp.append(exp_id)
	
	return exp

def score_fusion(exp, test_x, test_y, trial, M=20):
	   
	prob_M = np.zeros((M,lable_len, n_classes))
	f1_list = []
	for i in range(M):

		idx = exp[0][i]
		model = './model/'+exp[1]+'_'+str(idx)
		print(model)
		
		# reload tensorflow model
		saver.restore(sess, model)

		num_test_process = test_se//test_win + 1
		test_losses     = np.zeros(num_test_process, dtype=np.float32)
		test_accuracies = np.zeros(num_test_process, dtype=np.float32)
		predictvalue = []
		actualvalue = []

		test_ini = np.zeros((number_of_layers, 2, test_bt, rnn_size))
		prob_2d = np.zeros((lable_len, n_classes))

		for j in range(num_test_process):
			start = j*test_win
			end = np.min((test_se, start+test_win))

			feed_dict = {
				x: test_x[:,start:end,:],
				y: test_y[:,start:end,:],
				states: test_ini,
				dropout_keep_prob: 1.0 #test no drop 
			}

			test_loss, test_acc, test_ini, valueP, valueA, prob_3d = sess.run([cost, accuracy_train, state, predicted, actual, prediction], feed_dict=feed_dict)

			valueP = np.reshape(np.array(valueP), (-1))
			valueA = np.reshape(np.array(valueA), (-1))
			predictvalue.extend(valueP)
			actualvalue.extend(valueA)
			prob_2d[start*test_bt:end*test_bt,:] = np.reshape(np.array(prob_3d), ((end-start)*test_bt, n_classes))

			test_losses[j]     = test_loss
			test_accuracies[j] = test_acc

			####################### add softmax
			# the prediction value haven't use softmax fuction, in training process, it has no effect as loss function will automatically use it.
			# however, with fusion, it may lead to slightly difference. So, we add it before fusion step. 
			soft = sess.run([softmax],feed_dict={soft_max: prob_2d[start*test_bt:end*test_bt,:]})
			soft_np = np.array(soft)
			prob_2d[start*test_bt:end*test_bt,:] = np.reshape(soft_np, (-1, n_classes))
			#######################

		predictvalue = np.reshape(predictvalue, (-1))
		actualvalue = np.reshape(actualvalue, (-1))
		f1_value = f1_score(actualvalue, predictvalue, average='macro')
		print("  Test F1: %.3f" % f1_value)
		confusion = confusion_matrix(actualvalue, predictvalue)
		print("Tset Confusion matrix:\n%s" % confusion)

		print("  Average Test Accuracy: %.3f" % np.mean(test_accuracies))
		print("  Average Test Loss:     %.3f" % np.mean(test_losses))
		
		prob_M[i,:,:] = prob_2d
		curr_prob_avg = np.mean(prob_M[:i+1,:,:], axis=0)#extract from 0-1, 0-2 ... 0-M, so, the first f1_fused will same with f1_value.
		fused_pred = np.argmax(curr_prob_avg, axis=1)

		f1_fused = f1_score(actualvalue, fused_pred, average='macro')#actualvalue means test_y with shape (lable_len, n_classes)

		print('curr_f1 {:.3f} and sz {} fused_f1 {:.3f}'.format(f1_value, i+1, f1_fused))

		if i == 0 or i == 4 or i == 9 or i == 19:
			f1_list.append(f1_fused)
			if setnum == 1:
				with open('f1_fused_Opp.txt', 'a') as the_file:
					context = "In top " + str(i+1) + " models, "
					context += "the f1_fused of Trial "
					context += str(trial) + " is: " + str(f1_fused)
					context += "\n"
					the_file.write(context)
				name = "Opptunity_" + str(trial)

			if setnum == 2:
				with open('f1_fused_Pamap2.txt', 'a') as the_file:
					context = "In top " + str(i+1) + ", "
					context += "the f1_fused of Trial "
					context += str(trial) + " is: " + str(f1_fused)
					context += "\n"
					the_file.write(context)
				name = "Pamap2_" + str(trial)

			if setnum == 3:
				with open('f1_fused_skoda.txt', 'a') as the_file:
					context = "In top " + str(i+1) + ", "
					context += "the f1_fused of Trial "
					context += str(trial) + " is: " + str(f1_fused)
					context += "\n"
					the_file.write(context)
				name = "Skoda_" + str(trial)
	
	np.savetxt(csv, f1_list, fmt='%.3f', delimiter=' ', header=name)				   
	return prob_M

prediction, state = recurrent_neural_network(x,rnn_tuple_state,dropout_keep_prob)# prediction not do the softmax function
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y) )# need specific
predicted = tf.argmax(prediction,2)
actual = tf.argmax(y,2)
accuracy_train = tf.contrib.metrics.accuracy(predicted, actual)

tf_vars_to_save = tf.trainable_variables()
saver = tf.train.Saver(tf_vars_to_save)

softmax = tf.nn.softmax(soft_max)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) # is use gpu
# sess = tf.Session() # if only use cpu
sess.run(tf.initialize_all_variables())

loss = 'CE'

if DB==79:
	dataset='OPP'
if DB==52:
	dataset='PAMAP2'
if DB==60:
	dataset='Skoda'

for t in range(30):
	Probs_ensemble = []
	trial = t
	exp_id1 = 'T_'+str(trial)+'_'+loss+'_'+str(DB)
	print(dataset+'--'+exp_id1)


	bestM=20
	exp1 = exp_setting(dataset, exp_id1, bestM)
	prob_M1 =  score_fusion(exp1, test_x, test_y, trial, bestM)
	
	prob_avg_1 = np.mean(prob_M1[:1,:,:], axis=0)
	prob_avg_5 = np.mean(prob_M1[:5,:,:], axis=0)
	prob_avg_10 = np.mean(prob_M1[:10,:,:], axis=0)
	prob_avg_20 = np.mean(prob_M1[:20,:,:], axis=0)
	
	Probs_ensemble.append(prob_avg_1)
	Probs_ensemble.append(prob_avg_5)
	Probs_ensemble.append(prob_avg_10)
	Probs_ensemble.append(prob_avg_20)
	
	np.save('Trial/'+exp_id1+'.npy', Probs_ensemble)




