from paddle.trainer_config_helpers import *

data_dir = './data/'
define_py_data_sources2(train_list=data_dir + 'train.list',
                        test_list=data_dir + 'test.list',
                        module='provider',
                        obj='process')
settings(
    batch_size=128,
    learning_rate=1e-3,
    learning_method=AdamOptimizer(),
    regularization=L2Regularization(0.5)
)

img = data_layer(name='pixel', size=28 * 28)

hidden1 = simple_img_conv_pool(input=img, filter_size=3, num_filters=32, pool_size=3,
                               num_channel=1)

hidden2 = fc_layer(input=hidden1, size=200, act=TanhActivation(),
                   layer_attr=ExtraAttr(drop_rate=0.5))
predict = fc_layer(input=hidden2, size=10, act=SoftmaxActivation())

outputs(predict)
