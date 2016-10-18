from paddle.trainer_config_helpers import *

data_dir='./data/'
define_py_data_sources2(train_list= data_dir + 'train.list',
                        test_list= data_dir + 'test.list',
                        module='provider',
                        obj='process')
settings(
    batch_size = 128,
    learning_rate = 1e-3,
    learning_method = AdaGradOptimizer()
)

img = data_layer(name='pixel', size=28*28)

hidden1 = fc_layer(input=img, size=200, act=TanhActivation())
hidden2 = fc_layer(input=hidden1, size=200, act=TanhActivation())
predict = fc_layer(input=hidden2, size=10, act=SoftmaxActivation())

outputs(classification_cost(input=predict, label=data_layer(name='label', size=10)))
