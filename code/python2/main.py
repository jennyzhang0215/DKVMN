import sys
import os
import logging
import argparse
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np

from load_data import DATA
from model import MODEL
from run import train, test

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

def find_file(dir_name, best_epoch):
    for dir, subdir, files in os.walk(dir_name):
        for sub in subdir:
            if sub[0:len(best_epoch)] == best_epoch and sub[len(best_epoch)]=="_":
                return sub

def load_params(prefix, epoch):
    save_dict = nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params

def train_one_dataset(params, file_name, train_q_data, train_qa_data, valid_q_data, valid_qa_data):
    ### ================================== model initialization ==================================
    g_model = MODEL(n_question=params.n_question,
                    seqlen=params.seqlen,
                    batch_size=params.batch_size,
                    q_embed_dim=params.q_embed_dim,
                    qa_embed_dim=params.qa_embed_dim,
                    memory_size=params.memory_size,
                    memory_key_state_dim=params.memory_key_state_dim,
                    memory_value_state_dim=params.memory_value_state_dim,
                    final_fc_dim = params.final_fc_dim)
    # create a module by given a Symbol
    net = mx.mod.Module(symbol=g_model.sym_gen(),
                        data_names = ['q_data', 'qa_data'],
                        label_names = ['target'],
                        context=params.ctx)
    # create memory by given input shapes
    net.bind(data_shapes=[mx.io.DataDesc(name='q_data', shape=(params.seqlen, params.batch_size), layout='SN'),
                          mx.io.DataDesc(name='qa_data', shape=(params.seqlen, params.batch_size), layout='SN')],
             label_shapes=[mx.io.DataDesc(name='target', shape=(params.seqlen, params.batch_size), layout='SN')])
    # initial parameters with the default random initializer
    net.init_params(initializer=mx.init.Normal(sigma=params.init_std))
    # decay learning rate in the lr_scheduler
    lr_scheduler = mx.lr_scheduler.FactorScheduler(step=20*(train_q_data.shape[0]/params.batch_size), factor=0.667, stop_factor_lr=1e-5)

    net.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate': params.lr, 'momentum':params.momentum,'lr_scheduler': lr_scheduler})

    for parameters in net.get_params()[0]:
        print parameters, net.get_params()[0][parameters].asnumpy().shape
    print "\n"

    ### ================================== start training ==================================
    all_train_loss = {}
    all_train_accuracy = {}
    all_train_auc = {}
    all_valid_loss = {}
    all_valid_accuracy = {}
    all_valid_auc = {}
    best_valid_auc = 0

    for idx in xrange(params.max_iter):
        train_loss, train_accuracy, train_auc = train(net, params, train_q_data, train_qa_data, label='Train')
        valid_loss, valid_accuracy, valid_auc = test(net, params, valid_q_data, valid_qa_data, label='Valid')

        print 'epoch', idx + 1
        print "valid_auc\t", valid_auc, "\ttrain_auc\t", train_auc
        print "valid_accuracy\t", valid_accuracy, "\ttrain_accuracy\t", train_accuracy
        print "valid_loss\t", valid_loss, "\ttrain_loss\t", train_loss

        if not os.path.isdir('model'):
            os.makedirs('model')
        if not os.path.isdir(os.path.join('model', params.save)):
            os.makedirs(os.path.join('model', params.save))
        net.save_checkpoint(prefix=os.path.join('model', params.save, file_name), epoch=idx+1)
        # output the epoch with the best validation auc
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_epoch = idx+1

    if not os.path.isdir('result'):
        os.makedirs('result')
    if not os.path.isdir(os.path.join('result', params.save)):
        os.makedirs(os.path.join('result', params.save))
    f_save_log = open(os.path.join('result', params.save, file_name), 'w')
    f_save_log.write("valid_auc:\n" + str(all_valid_auc) + "\n\n")
    f_save_log.write("train_auc:\n" + str(all_train_auc) + "\n\n")
    f_save_log.write("valid_loss:\n" + str(all_valid_loss) + "\n\n")
    f_save_log.write("train_loss:\n" + str(all_train_loss) + "\n\n")
    f_save_log.write("valid_accuracy:\n" + str(all_valid_accuracy) + "\n\n")
    f_save_log.write("train_accuracy:\n" + str(all_train_accuracy) + "\n\n")
    f_save_log.close()
    return best_epoch

def test_one_dataset(params, file_name, test_q_data, test_qa_data, best_epoch):
    print "\n\nStart testing ......................\n Best epoch:", best_epoch
    g_model = MODEL(n_question=params.n_question,
                    seqlen=params.seqlen,
                    batch_size=params.batch_size,
                    q_embed_dim=params.q_embed_dim,
                    qa_embed_dim=params.qa_embed_dim,
                    memory_size=params.memory_size,
                    memory_key_state_dim=params.memory_key_state_dim,
                    memory_value_state_dim=params.memory_value_state_dim,
                    final_fc_dim=params.final_fc_dim)
    # create a module by given a Symbol
    test_net = mx.mod.Module(symbol=g_model.sym_gen(),
                             data_names=['q_data', 'qa_data'],
                             label_names=['target'],
                             context=params.ctx)
    # cresate memory by given input shapes
    test_net.bind(data_shapes=[
        mx.io.DataDesc(name='q_data', shape=(params.seqlen, params.batch_size), layout='SN'),
        mx.io.DataDesc(name='qa_data', shape=(params.seqlen, params.batch_size), layout='SN')],
        label_shapes=[mx.io.DataDesc(name='target', shape=(params.seqlen, params.batch_size), layout='SN')])
    arg_params, aux_params = load_params(prefix=os.path.join('model', params.load, file_name),
                                         epoch=best_epoch)
    test_net.init_params(arg_params=arg_params, aux_params=aux_params,
                         allow_missing=False)
    test_loss, test_accuracy, test_auc = test(test_net, params, test_q_data, test_qa_data, label='Test')
    print "\ntest_auc\t", test_auc
    print "test_accuracy\t", test_accuracy
    print "test_loss\t", test_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to test KVMN.')
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--max_iter', type=int, default=50, help='number of iterations')
    parser.add_argument('--test', type=bool, default=False, help='enable testing')
    parser.add_argument('--train_test', type=bool, default=True, help='enable testing after training')
    parser.add_argument('--show', type=bool, default=True, help='print progress')

    dataset = "STATICS"  # synthetic / assist2009_updated / assist2015 / STATICS

    if dataset == "synthetic":
        parser.add_argument('--batch_size', type=int, default=32, help='the batch size')
        parser.add_argument('--q_embed_dim', type=int, default=10, help='question embedding dimensions')
        parser.add_argument('--qa_embed_dim', type=int, default=10, help='answer and question embedding dimensions')
        parser.add_argument('--memory_size', type=int, default=5, help='memory size')

        parser.add_argument('--init_std', type=float, default=0.1, help='weight initialization std')
        parser.add_argument('--init_lr', type=float, default=0.05, help='initial learning rate')
        parser.add_argument('--final_lr', type=float, default=1E-5, help='learning rate will not decrease after hitting this threshold')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
        parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')
        parser.add_argument('--final_fc_dim', type=float, default=50, help='hidden state dim for final fc layer')

        parser.add_argument('--n_question', type=int, default=50, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=50, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='../../data/synthetic', help='data directory')
        parser.add_argument('--data_name', type=str, default='naive_c5_q50_s4000_v1', help='data set name')
        parser.add_argument('--load', type=str, default='synthetic/v1', help='model file to load')
        parser.add_argument('--save', type=str, default='synthetic/v1', help='path to save model')
    if dataset == "assist2009_updated":
        parser.add_argument('--batch_size', type=int, default=32, help='the batch size')
        parser.add_argument('--q_embed_dim', type=int, default=50, help='question embedding dimensions')
        parser.add_argument('--qa_embed_dim', type=int, default=200, help='answer and question embedding dimensions')
        parser.add_argument('--memory_size', type=int, default=20, help='memory size')

        parser.add_argument('--init_std', type=float, default=0.1, help='weight initialization std')
        parser.add_argument('--init_lr', type=float, default=0.05, help='initial learning rate')
        parser.add_argument('--final_lr', type=float, default=1E-5, help='learning rate will not decrease after hitting this threshold')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
        parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')
        parser.add_argument('--final_fc_dim', type=float, default=50, help='hidden state dim for final fc layer')

        parser.add_argument('--n_question', type=int, default=110, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='../../data/assist2009_updated', help='data directory')
        parser.add_argument('--data_name', type=str, default='assist2009_updated', help='data set name')
        parser.add_argument('--load', type=str, default='assist2009_updated', help='model file to load')
        parser.add_argument('--save', type=str, default='assist2009_updated', help='path to save model')
    elif dataset == "assist2015":
        parser.add_argument('--batch_size', type=int, default=50, help='the batch size')
        parser.add_argument('--q_embed_dim', type=int, default=50, help='question embedding dimensions')
        parser.add_argument('--qa_embed_dim', type=int, default=100, help='answer and question embedding dimensions')
        parser.add_argument('--memory_size', type=int, default=20, help='memory size')

        parser.add_argument('--init_std', type=float, default=0.1, help='weight initialization std')
        parser.add_argument('--init_lr', type=float, default=0.1, help='initial learning rate')
        parser.add_argument('--final_lr', type=float, default=1E-5, help='learning rate will not decrease after hitting this threshold')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
        parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')
        parser.add_argument('--final_fc_dim', type=float, default=50, help='hidden state dim for final fc layer')

        parser.add_argument('--n_question', type=int, default=100, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='../../data/assist2015', help='data directory')
        parser.add_argument('--data_name', type=str, default='assist2015', help='data set name')
        parser.add_argument('--load', type=str, default='assist2015', help='model file to load')
        parser.add_argument('--save', type=str, default='assist2015', help='path to save model')
    elif dataset == "STATICS":
        parser.add_argument('--batch_size', type=int, default=10, help='the batch size')
        parser.add_argument('--q_embed_dim', type=int, default=50, help='question embedding dimensions')
        parser.add_argument('--qa_embed_dim', type=int, default=100, help='answer and question embedding dimensions')
        parser.add_argument('--memory_size', type=int, default=50, help='memory size')

        parser.add_argument('--init_std', type=float, default=0.1, help='weight initialization std')
        parser.add_argument('--init_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--final_lr', type=float, default=1E-5,
                            help='learning rate will not decrease after hitting this threshold')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
        parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')
        parser.add_argument('--final_fc_dim', type=float, default=50, help='hidden state dim for final fc layer')

        parser.add_argument('--n_question', type=int, default=1223, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='../../data/STATICS', help='data directory')
        parser.add_argument('--data_name', type=str, default='STATICS', help='data set name')
        parser.add_argument('--load', type=str, default='STATICS', help='model file to load')
        parser.add_argument('--save', type=str, default='STATICS', help='path to save model')

    params = parser.parse_args()
    params.lr = params.init_lr
    params.memory_key_state_dim = params.q_embed_dim
    params.memory_value_state_dim = params.qa_embed_dim

    params.dataset = dataset
    if params.gpus == None:
        ctx = mx.cpu()
        print "Training with cpu ..."
    else:
        ctx = mx.gpu(int(params.gpus))
        print "Training with gpu(" + params.gpus + ") ..."
    params.ctx = ctx

    # Read data
    dat = DATA(n_question=params.n_question, seqlen=params.seqlen, separate_char=',')

    seedNum =224
    np.random.seed(seedNum)
    if not params.test:
        params.memory_key_state_dim = params.q_embed_dim
        params.memory_value_state_dim = params.qa_embed_dim
        d = vars(params)
        for key in d:
            print '\t', key, '\t', d[key]
        file_name = 'b' + str(params.batch_size) + \
                    '_q' + str(params.q_embed_dim) + '_qa' + str(params.qa_embed_dim) + \
                    '_m' + str(params.memory_size) + '_std' + str(params.init_std) + \
                    '_lr' + str(params.init_lr) + '_gn' + str(params.maxgradnorm) + \
                    '_f' + str(params.final_fc_dim)+'_s'+str(seedNum)
        train_data_path = params.data_dir + "/" + params.data_name + "_train1.csv"
        valid_data_path = params.data_dir + "/" + params.data_name + "_valid1.csv"
        train_q_data, train_qa_data = dat.load_data(train_data_path)
        valid_q_data, valid_qa_data = dat.load_data(valid_data_path)
        print "\n"
        print "train_q_data.shape", train_q_data.shape  ###(3633, 200) = (#sample, seqlen)
        print "train_qa_data.shape", train_qa_data.shape  ###(3633, 200) = (#sample, seqlen)
        print "valid_q_data.shape", valid_q_data.shape  ###(1566, 200)
        print "valid_qa_data.shape", valid_qa_data.shape  ###(1566, 200)
        print "\n"
        best_epoch = train_one_dataset(params, file_name, train_q_data, train_qa_data, valid_q_data, valid_qa_data)
        if params.train_test:
            test_data_path = params.data_dir + "/" + params.data_name + "_test.csv"
            test_q_data, test_qa_data = dat.load_data(test_data_path)
            test_one_dataset(params, file_name, test_q_data, test_qa_data, best_epoch)
    else:
        params.memory_key_state_dim = params.q_embed_dim
        params.memory_value_state_dim = params.qa_embed_dim
        test_data_path = params.data_dir + "/" + params.data_name  +"_test.csv"
        test_q_data, test_qa_data = dat.load_data(test_data_path)
        best_epoch = 30
        file_name = 'b' + str(params.batch_size) + \
                    '_q' + str(params.q_embed_dim) + '_qa' + str(params.qa_embed_dim) + \
                    '_m' + str(params.memory_size) + '_std' + str(params.init_std) + \
                    '_lr' + str(params.init_lr) + '_gn' + str(params.maxgradnorm) + \
                    '_f' + str(params.final_fc_dim) + '_s' + str(seedNum)
        test_one_dataset(params, file_name, test_q_data, test_qa_data, best_epoch)
