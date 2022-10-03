from __future__ import print_function
import os
import time
import random
import datetime
import imageio
import numpy as np
from PIL import Image
import tf_slim as slim
from datetime import datetime
import tensorflow as tf
from tensorflow_core.python.ops.gen_summary_ops import summary_writer

from my_util.BasicConvLSTMCell import BasicConvLSTMCell
from my_util.util import im2uint8, ResnetBlock, print_link


class DEBLUR(object):
    def __init__(self, args):
        self.args = args
        self.n_levels = args.n_levels
        self.normalize_noise = args.normalize_noise
        self.scale = 0.5
        self.chns = 3 if self.args.model == 'color' else 1  # input / output channels
        self.reblur = args.beta != 0

        if args.phase == 'train':
            self.crop_size = args.crop_size
            self.data_list = open(args.datalist, 'rt').read().splitlines()
        # self.data_list = list(map(lambda x: x.split(' '), self.data_list))
            self.data_list = list(map(lambda x: x.split('\t'), self.data_list))
            random.shuffle(self.data_list)
            self.batch_size = args.batch_size
            self.epoch = args.epoch
            self.data_size = (len(self.data_list)) // self.batch_size
            self.org_ckpt_step = self.args.org_ckpt_step
            already_train_steps = args.step - self.org_ckpt_step
            already_train_epochs = already_train_steps / self.data_size
            remaining_epochs = self.epoch - already_train_epochs
            self.max_steps = int(remaining_epochs * self.data_size + self.args.step)
            self.learning_rate = args.learning_rate

        self.transf_lr = args.transfer_learning

        self.train_dir = os.path.join('../checkpoints', args.model, args.expname)
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        self.kernel_sz = 341
        self.output_chns = 3

        # training attributes
        self.loss_total  = None
        self.global_step = None
        self.lr          = None
        self.sess        = None
        self.saver       = None
        self.saver_to_restore = None
        self.restoring_same = self.args.restoring_same
        self.all_vars    = None
        self.transf_vars = None
        self.g_vars      = None
        self.lstm_vars   = None

        self.upsample_type = self.args.upsample_type

        self.linear_data = self.args.linear_data
        if self.args.linear_data:
            self.gamma = 2.2
        else:
            self.gamma = 1

    def upsample_layer(self, type, input, output_size, kernel_size, upsampling_size, name):
        if type == 'conv':
            return slim.layers.conv2d_transpose(input, output_size, kernel_size, stride=upsampling_size, scope=name)
        if type in ['bilinear', 'nearest']:
            layer_1 = tf.compat.v1.keras.layers.UpSampling2D(size=upsampling_size, interpolation=type)(input)
            layer_2 = slim.layers.conv2d(layer_1, output_size, kernel_size, stride=1, scope=name)
            return layer_2
        raise ValueError('no such layer exist')

    def input_producer(self, batch_size=10):
        def read_data(data_queue):
            img_a = tf.image.decode_image(tf.io.read_file(data_queue[0]))
            img_b = tf.image.decode_image(tf.io.read_file(data_queue[1]))
            img_k = tf.image.decode_image(tf.io.read_file(data_queue[2]))
            img_a, img_b, img_k = preprocessing([img_a, img_b, img_k])
            return img_a, img_b, img_k

        def preprocessing(imgs):
            imgs     = [tf.cast(img, tf.float32) / 255.0 for img in imgs]
            if self.args.model != 'color':
                imgs = [tf.image.rgb_to_grayscale(img) for img in imgs]

            imgs_ab  = tf.stack(imgs[:2], axis=0)
            imgs_k   = imgs[-1]
            img_crop = tf.unstack(tf.image.random_crop(imgs_ab, [2, self.crop_size, self.crop_size, self.chns]), axis=0)
            # k_crop = tf.unstack(tf.image.random_crop(imgs_k, [1, self.kernel_sz, self.kernel_sz, self.chns]), axis=0)
            imgs_k.set_shape([self.kernel_sz, self.kernel_sz, 1])
            img_crop_k = img_crop + [imgs_k]
            return img_crop_k

        with tf.compat.v1.variable_scope('input'):
            list_all = tf.convert_to_tensor(self.data_list, dtype=tf.string)
            gt_list = list_all[:, 0]
            in_list = list_all[:, 1]
            k_list  = list_all[:, 2]
            data_queue = tf.compat.v1.train.slice_input_producer( (in_list, gt_list, k_list), capacity=20)

            # data_queue = tf.data.Dataset.from_tensor_slices(tuple([gt_list, in_list])).shuffle(tf.shape(list_all, out_type=tf.int64)[0]).repeat(self.epoch)
            # list(self.data_queue.as_numpy_iterator())
            # !TODO look, there is a squirrel over there
            image_in, image_gt, image_k = read_data(data_queue)
            batch_in, batch_gt, batch_k = tf.compat.v1.train.batch([image_in, image_gt, image_k], batch_size=batch_size, num_threads=8, capacity=20, )
            # batch_in, batch_gt = tf.data.Dataset.batch([image_in, image_gt], batch_size=batch_size, drop_remainder=False) # representing whether the last batch should be dropped in the case it has fewer than batch_size elements; the default behavior is not to drop the smaller batch.

        return batch_in, batch_gt, batch_k

    def generator(self, inputs, reuse=False, scope='g_net'):
        n, h, w, c = inputs.get_shape().as_list()

        if self.args.model == 'lstm':
            with tf.compat.v1.variable_scope('LSTM'):
                cell = BasicConvLSTMCell([h / 4, w / 4], [3, 3], 128)
                rnn_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        x_unwrap = []
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=None,
                                # weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True), # weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                weights_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution=("uniform" if True else "truncated_normal")),
                                biases_initializer=tf.compat.v1.constant_initializer(0.0)):
                                # biases_initializer=tf.constant_initializer(0.0)):

                inp_pred = inputs
                for i in xrange(self.n_levels):
                    scale = self.scale ** (self.n_levels - i - 1)
                    hi = int(round(h * scale))
                    wi = int(round(w * scale))
                    inp_blur = tf.image.resize(inputs, [hi, wi])
                    inp_pred = tf.stop_gradient(tf.image.resize(inp_pred, [hi, wi]))
                    inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')
                    if self.args.model == 'lstm':
                        rnn_state = tf.image.resize(rnn_state, [hi // 4, wi // 4])

                    # encoder
                    conv1_1 = slim.layers.conv2d(inp_all, 32, [5, 5], scope='enc1_1')
                    conv1_2 = ResnetBlock(conv1_1, 32, 5, scope='enc1_2')
                    conv1_3 = ResnetBlock(conv1_2, 32, 5, scope='enc1_3')
                    conv1_4 = ResnetBlock(conv1_3, 32, 5, scope='enc1_4')
                    conv2_1 = slim.layers.conv2d(conv1_4, 64, [5, 5], stride=2, scope='enc2_1')
                    conv2_2 = ResnetBlock(conv2_1, 64, 5, scope='enc2_2')
                    conv2_3 = ResnetBlock(conv2_2, 64, 5, scope='enc2_3')
                    conv2_4 = ResnetBlock(conv2_3, 64, 5, scope='enc2_4')
                    conv3_1 = slim.layers.conv2d(conv2_4, 128, [5, 5], stride=2, scope='enc3_1')
                    conv3_2 = ResnetBlock(conv3_1, 128, 5, scope='enc3_2')
                    conv3_3 = ResnetBlock(conv3_2, 128, 5, scope='enc3_3')
                    conv3_4 = ResnetBlock(conv3_3, 128, 5, scope='enc3_4')

                    if self.args.model == 'lstm':
                        deconv3_4, rnn_state = cell(conv3_4, rnn_state)
                    else:
                        deconv3_4 = conv3_4

                    deconv3_3 = ResnetBlock(deconv3_4, 128, 5, scope='dec3_3')
                    deconv3_2 = ResnetBlock(deconv3_3, 128, 5, scope='dec3_2')
                    deconv3_1 = ResnetBlock(deconv3_2, 128, 5, scope='dec3_1')

                    deconv2_4 = self.upsample_layer(self.upsample_type, deconv3_1, 64, [4, 4], 2, 'dec2_4')

                    cat2 = deconv2_4 + conv2_4
                    deconv2_3 = ResnetBlock(cat2, 64, 5, scope='dec2_3')
                    deconv2_2 = ResnetBlock(deconv2_3, 64, 5, scope='dec2_2')
                    deconv2_1 = ResnetBlock(deconv2_2, 64, 5, scope='dec2_1')

                    deconv1_4 = self.upsample_layer(self.upsample_type, deconv2_1, 32, [4, 4], 2, 'dec1_4')

                    cat1 = deconv1_4 + conv1_4
                    deconv1_3 = ResnetBlock(cat1, 32, 5, scope='dec1_3')
                    deconv1_2 = ResnetBlock(deconv1_3, 32, 5, scope='dec1_2')
                    deconv1_1 = ResnetBlock(deconv1_2, 32, 5, scope='dec1_1')
                    inp_pred = slim.layers.conv2d(deconv1_1, self.chns, [5, 5], activation_fn=None, scope='dec1_0')

                    if self.transf_lr:
                        conv1_1.trainable = False
                        conv1_2.trainable = False
                        conv1_3.trainable = False
                        conv1_4.trainable = False
                        conv2_1.trainable = False
                        conv2_2.trainable = False
                        conv2_3.trainable = False
                        conv2_4.trainable = False
                        conv3_1.trainable = False
                        conv3_2.trainable = False
                        conv3_3.trainable = False
                        conv3_4.trainable = False

                        deconv3_3.trainable = False
                        deconv3_2.trainable = False
                        deconv3_1.trainable = False
                        deconv2_4.trainable = False
                        deconv2_3.trainable = False
                        deconv2_2.trainable = False
                        deconv2_1.trainable = False
                        deconv1_4.trainable = False
                        deconv1_3.trainable = True
                        deconv1_2.trainable = True
                        deconv1_1.trainable = True
                        inp_pred.trainable  = True

                    x_unwrap.append(inp_pred)
                    if i == 0:
                        tf.compat.v1.get_variable_scope().reuse_variables()

            return x_unwrap

    def build_model(self):
        img_in, img_gt, img_k = self.input_producer(self.batch_size)
        img_in = self.add_noise(img_in)
        tf.compat.v1.summary.image('img_in', im2uint8(img_in ** (1 / self.gamma)))
        tf.compat.v1.summary.image('img_gt', im2uint8(img_gt))
        # tf.compat.v1.summary.image('img_k', im2uint8(img_k))
        print('img_in, img_gt, img_k', img_in.get_shape(), img_gt.get_shape(), img_k.get_shape())

        # generator
        x_unwrap = self.generator(img_in, reuse=False, scope='g_net')

        # calculate multi-scale loss + Reblur2Deblur loss
        self.loss_total = 0
        for i in xrange(self.n_levels):
            _, hi, wi, _ = x_unwrap[i].get_shape().as_list()
            gt_i = tf.image.resize(img_gt, [hi, wi])
            gt_i_linear = gt_i ** self.gamma
            # gt_i_linear_mean = tf.reduce_mean(gt_i_linear, axis=[1, 2, 3], keepdims=True)
            # pred_i_linear_mean = tf.reduce_mean(x_unwrap[i], axis=[1, 2, 3], keepdims=True)
            # main_loss = tf.reduce_mean((gt_i_linear - (gt_i_linear_mean / pred_i_linear_mean) * x_unwrap[i]) ** 2)
            main_loss = tf.reduce_mean((gt_i_linear - x_unwrap[i]) ** 2)

            if self.reblur:
                # pred deblur
                pred_i = x_unwrap[i]
                pred_i = tf.transpose(pred_i, [1, 2, 0, 3])
                pred_i = tf.reshape(pred_i, [1, hi, wi, self.batch_size * self.chns])

                # kernels
                _, hk, wk, _ = img_k.get_shape().as_list()
                scale = self.scale ** (self.n_levels - i - 1)
                hki = int(round(hk * scale))
                wki = int(round(wk * scale))
                blur_kernel = tf.image.resize(img_k, [hki, wki])
                blur_kernel = tf.transpose(tf.repeat(blur_kernel, 3, -1), [1, 2, 0, 3])
                blur_kernel = tf.reshape(blur_kernel, [hki, wki, self.batch_size * self.chns, 1])

                # pred reblur
                reblur_i = tf.nn.depthwise_conv2d(pred_i, blur_kernel, padding='SAME', strides=[1, 1, 1, 1])
                reblur_i = tf.reshape(reblur_i, [hi, wi, self.batch_size, self.chns])
                reblur_i = tf.transpose(reblur_i, [2, 0, 1, 3])

                # gt blur
                blur_i = tf.image.resize(img_in, [hi, wi])
                reblur_loss = tf.reduce_mean((blur_i - reblur_i) ** 2)
            else:
                reblur_loss = 0

            overall_loss = main_loss + self.args.beta * reblur_loss
            self.loss_total += overall_loss


            display_photo = tf.clip_by_value(x_unwrap[i]/tf.reduce_max(x_unwrap[i], axis=[1, 2, 3], keep_dims=True), clip_value_min=0, clip_value_max=1)
            tf.compat.v1.summary.image('out_' + str(i), im2uint8(display_photo ** (1 / self.gamma)))
            tf.compat.v1.summary.scalar('main_loss_' + str(i), main_loss)
            # tf.compat.v1.summary.scalar('reblur_loss_' + str(i), reblur_loss)

        # losses
        tf.compat.v1.summary.scalar('loss_total', self.loss_total)

        # training vars
        all_vars = tf.compat.v1.trainable_variables()
        # all_vars = filter(lambda i: 'g_net/dec4_1' not in i.name and 'g_net/enc4_1' not in i.name, all_vars)

        self.all_vars = all_vars
        self.transf_vars = []
        transf_layers = ['dec1_3', 'dec1_2', 'dec1_1', 'dec1_0']
        self.transf_vars = []
        for var in all_vars:
            for index in range(len(transf_layers)):
                if transf_layers[index] in var.name:
                    self.transf_vars.append(var)
        self.g_vars = [var for var in all_vars if 'g_net' in var.name]
        self.lstm_vars = [var for var in all_vars if 'LSTM' in var.name]
        for var in all_vars:
            print(var.name)

    def train(self):
        def get_optimizer(loss, global_step=None, var_list=None, is_gradient_clip=False):
            train_op = tf.compat.v1.train.AdamOptimizer(self.lr)
            if is_gradient_clip:
                grads_and_vars = train_op.compute_gradients(loss, var_list=var_list)
                unchanged_gvs = [(grad, var) for grad, var in grads_and_vars if not 'LSTM' in var.name]
                rnn_grad = [grad for grad, var in grads_and_vars if 'LSTM' in var.name]
                rnn_var = [var for grad, var in grads_and_vars if 'LSTM' in var.name]
                capped_grad, _ = tf.clip_by_global_norm(rnn_grad, clip_norm=3)
                capped_gvs = list(zip(capped_grad, rnn_var))
                train_op_min = train_op.apply_gradients(grads_and_vars=capped_gvs + unchanged_gvs, global_step=global_step)
            else:
                train_op_min = train_op.minimize(loss, global_step, var_list)
            return train_op_min, train_op

        global_step = tf.Variable(initial_value=self.args.step, dtype=tf.int32, trainable=False)

        # build model
        self.build_model()

        # # session and thread
        # gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        # self.sess = sess
        # sess.run(tf.compat.v1.global_variables_initializer())

        # learning rate decay
        self.lr =tf.compat.v1.train.polynomial_decay(self.learning_rate, global_step, self.max_steps, end_learning_rate=0, power=0.3)
        tf.compat.v1.summary.scalar('learning_rate', self.lr)

        # training operators
        # if self.transf_lr == True:
        #     train_gnet = get_optimizer(self.loss_total, global_step, self.transf_vars)
        # else:
        train_gnet, opt = get_optimizer(self.loss_total, global_step, self.all_vars)

        # session and thread
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        sess =tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        self.sess = sess

        sess.run(tf.compat.v1.global_variables_initializer())
        if self.args.upsample_type != 'conv':
            trainable_list = filter(lambda i: i.name not in ['g_net/dec1_4/weights:0', 'g_net/dec2_4/weights:0'], self.all_vars)
            self.saver_to_restore = tf.compat.v1.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1,
                                                             var_list=trainable_list)
        else:
            self.saver_to_restore = tf.compat.v1.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
        self.saver = tf.compat.v1.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
        if self.args.step:    #new for starting from zero
            checkpoint = self.load(sess, os.path.join('../checkpoints', self.args.model, self.args.expname),
                                   step=self.args.step, restoring_same=self.restoring_same) # self.restoring_same
            self.args.checkpoint = checkpoint

        args_cnt = 0
        cont = True
        while cont:
            if not os.path.exists(os.path.join(self.train_dir, 'args{}.txt'.format(args_cnt))):
                break
            args_cnt += 1

        with open(os.path.join(self.train_dir, 'args{}.txt'.format(args_cnt)), 'w') as f:
            for arg in sorted(vars(self.args)):
                attr = getattr(self.args, arg)
                f.write('{:<20} : {}\n'.format(arg, attr))

        # saver = tf.compat.v1.train.import_meta_graph(r'./checkpoints/color/deblur.model-523000.meta')
        # self.saver.restore(sess, tf.train.latest_checkpoint(r'./checkpoints/color/deblur.model-392000.meta'))
        # self.saver = saver
        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

        # training summary
        summary_op = tf.compat.v1.summary.merge_all()
        summary_writer = tf.compat.v1.summary.FileWriter(self.train_dir, sess.graph, flush_secs=30)

        print("[*] Start training")
        print_link("[*] Saving results to ", self.train_dir)

        # for step in xrange(sess.run(global_step), self.max_steps + 1):
        print("start step {}".format(sess.run(global_step)))
        print("end step {}".format(self.max_steps))
        for step in xrange(sess.run(global_step), self.max_steps):
            start_time = time.time()
            curr_opt, loss_total_val = sess.run([train_gnet, self.loss_total])

            duration = time.time() - start_time
            # print loss_value
            assert not np.isnan(loss_total_val), 'Model diverged with loss = NaN'

            if step % 5 == 0:
                num_examples_per_step = self.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = (%.5f; %.5f, %.5f)(%.1f data/s; %.3f s/bch)')
                print(format_str % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step, loss_total_val, 0.0,
                                    0.0, examples_per_sec, sec_per_batch))

            if step % 20 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)

            # Save and validate the model checkpoint periodically.
            if step % 2000 == 0 or step == self.max_steps:
                checkpoint_path = os.path.join(self.train_dir)
                self.save(sess, checkpoint_path, step)
                self.val(step)


    def save(self, sess, checkpoint_dir, step):
        model_name = "deblur.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, step=None, restoring_same=True):
        print(" [*] Reading checkpoints...")
        model_name = "deblur.model"
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if step is not None:
            ckpt_name = model_name + '-' + str(step)
            if restoring_same:
                self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            else:
                self.saver_to_restore.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            checkpoint_path = os.path.join(checkpoint_dir, ckpt_name)
            print("[*] Reading intermediate checkpoints {}... Success".format(checkpoint_path))
            return checkpoint_path
            # print_tensors_in_checkpoint_file(checkpoint_path, all_tensors=False, tensor_name='g_net/dec2_1/conv1/weights')

            return str(step)
        elif ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_iter = ckpt_name.split('-')[1]
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading updated checkpoints... Success")
            return ckpt_iter
        else:
            print(" [*] Reading checkpoints... ERROR")
            return False

    def val(self, step):
        tf.compat.v1.disable_eager_execution()

        height = self.args.height
        width = self.args.width
        input_path = self.args.val_input_path
        output_path = os.path.join(self.train_dir, 'validation', str(step))


        if not os.path.exists(output_path):
            os.makedirs(output_path)
        imgsName = sorted(os.listdir(input_path))

        H, W = height, width
        inp_chns = 3 if self.args.model == 'color' else 1
        self.batch_size = 1 if self.args.model == 'color' else 3
        inputs = tf.compat.v1.placeholder(shape=[self.batch_size, H, W, inp_chns], dtype=tf.float32)
        outputs = self.generator(inputs, reuse=True)

        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))

        # self.saver = tf.compat.v1.train.Saver()
        self.load(sess, self.train_dir, step=step)

        print("[*] Start validation")
        print_link("[*] Reading images from", input_path)
        print_link("[*] Saving results to", output_path)

        for i, imgName in enumerate(imgsName):
            blur = imageio.imread(os.path.join(input_path, imgName))
            h, w, c = blur.shape
            # make sure the width is larger than the height
            rot = False
            if h > w:
                blur = np.transpose(blur, [1, 0, 2])
                rot = True
            h = int(blur.shape[0])
            w = int(blur.shape[1])
            resize = False
            if h > H or w > W:
                scale = min(1.0 * H / h, 1.0 * W / w)
                new_h = int(h * scale)
                new_w = int(w * scale)
                # blur = scipy.misc.imresize(blur, [new_h, new_w], 'bicubic')
                blur = np.array(Image.fromarray(blur).resize([new_w, new_h], Image.BICUBIC))
                resize = True
                blurPad = np.pad(blur, ( (0, H - new_h), (0, W - new_w), (0, 0)), 'edge')
            else:
                blurPad = np.pad(blur, ((0, H - h), (0, W - w), (0, 0)), 'edge')
            blurPad = np.expand_dims(blurPad, 0)
            if self.args.model != 'color':
                blurPad = np.transpose(blurPad, (3, 1, 2, 0))

            start = time.time()
            deblur = sess.run(outputs, feed_dict={inputs: blurPad / 255.0})
            duration = time.time() - start
            print('Saving results {}/{}: {} ... {:4.3f}s'.format(i, len(imgsName), os.path.join(output_path, imgName), duration))
            res = deblur[-1]
            if self.args.model != 'color':
                res = np.transpose(res, (3, 1, 2, 0))

            res = res[0, :, :, :]
            res = im2uint8(res / res.max())
            # crop the image into original size
            if resize:
                res = res[:new_h, :new_w, :]
                res = np.array(Image.fromarray(res).resize([h, w], Image.BICUBIC))
                # res = scipy.misc.imresize(res, [h, w], 'bicubic')
            else:
                res = res[:h, :w, :]

            if rot:
                res = np.transpose(res, [1, 0, 2])
            imageio.imsave(os.path.join(output_path, imgName), res)
    def test(self, height, width, input_path, output_path):
        tf.compat.v1.disable_eager_execution()

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        imgsName = sorted(os.listdir(input_path))

        H, W = height, width
        inp_chns = 3 if self.args.model == 'color' else 1
        self.batch_size = 1 if self.args.model == 'color' else 3
        inputs = tf.compat.v1.placeholder(shape=[self.batch_size, H, W, inp_chns], dtype=tf.float32)
        outputs = self.generator(inputs, reuse=False)

        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))

        self.saver = tf.compat.v1.train.Saver()
        self.load(sess, self.train_dir, step=self.args.step)

        print("[*] Start testing")
        print_link("[*] Reading images from", input_path)
        print_link("[*] Saving results to", output_path)

        for i, imgName in enumerate(imgsName):
            blur = imageio.imread(os.path.join(input_path, imgName))
            h, w, c = blur.shape
            # make sure the width is larger than the height
            rot = False
            if h > w:
                blur = np.transpose(blur, [1, 0, 2])
                rot = True
            h = int(blur.shape[0])
            w = int(blur.shape[1])
            resize = False
            if h > H or w > W:
                scale = min(1.0 * H / h, 1.0 * W / w)
                new_h = int(h * scale)
                new_w = int(w * scale)
                # blur = scipy.misc.imresize(blur, [new_h, new_w], 'bicubic')
                blur = np.array(Image.fromarray(blur).resize([new_w, new_h], Image.BICUBIC))
                resize = True
                blurPad = np.pad(blur, ( (0, H - new_h), (0, W - new_w), (0, 0)), 'edge')
            else:
                blurPad = np.pad(blur, ((0, H - h), (0, W - w), (0, 0)), 'edge')
            blurPad = np.expand_dims(blurPad, 0)
            if self.args.model != 'color':
                blurPad = np.transpose(blurPad, (3, 1, 2, 0))

            start = time.time()
            deblur = sess.run(outputs, feed_dict={inputs: blurPad / 255.0})
            duration = time.time() - start
            print('Saving results {}/{}: {} ... {:4.3f}s'.format(i, len(imgsName), os.path.join(output_path, imgName), duration))
            res = deblur[-1]
            if self.args.model != 'color':
                res = np.transpose(res, (3, 1, 2, 0))
            res = res[0, :, :, :]
            res = im2uint8(res / res.max())
            # res = im2uint8(res)
            # crop the image into original size
            if resize:
                res = res[:new_h, :new_w, :]
                res = np.array(Image.fromarray(res).resize([h, w], Image.BICUBIC))
                # res = scipy.misc.imresize(res, [h, w], 'bicubic')
            else:
                res = res[:h, :w, :]

            if rot:
                res = np.transpose(res, [1, 0, 2])
            imageio.imsave(os.path.join(output_path, imgName), res)

    def convolve2D(image, kernel, padding=0, strides=1):
        # Cross Correlation
        kernel = np.flipud(np.fliplr(kernel))

        # Gather Shapes of Kernel + Image + Padding
        xKernShape = kernel.shape[0]
        yKernShape = kernel.shape[1]
        xImgShape = image.shape[0]
        yImgShape = image.shape[0]

        # Shape of Output Convolution
        xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
        yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
        output = np.zeros((xOutput, yOutput))

        # Apply Equal Padding to All Sides
        if padding != 0:
            imagePadded = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2))
            imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        else:
            imagePadded = image

        # Iterate through image
        for y in range(image.shape[1]):
            # Exit Convolution
            if y > image.shape[1] - yKernShape:
                break
            # Only Convolve if y has gone down by the specified Strides
            if y % strides == 0:
                for x in range(image.shape[0]):
                    # Go to next row once kernel is out of bounds
                    if x > image.shape[0] - xKernShape:
                        break
                    try:
                        # Only Convolve if x has moved by the specified Strides
                        if x % strides == 0:
                            output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                    except:
                        break

        return output

    def add_noise(self, img_in):
        noise_std = self.args.noise_std

        if not self.normalize_noise:
            max_img = 1
        else:
            max_img = tf.math.reduce_max(img_in)

        # white_level = tf.pow(10., tf.random_uniform([1, 1, 1], np.log10(.1), np.log10(1.)))
        # img_in = img_in * white_level
        if self.args.noise == 'bpn_noise':
            sig_read = tf.pow(10., tf.random_uniform([self.batch_size, 1, 1, 1], *noise_std[:2])) #
            sig_shot = tf.pow(10., tf.random_uniform([self.batch_size, 1, 1, 1], *noise_std[2:])) #noise_std(2, 3)
            read = max_img * sig_read * tf.random_normal(tf.shape(img_in))
            shot = max_img * tf.sqrt(img_in) * sig_shot * tf.random_normal(tf.shape(img_in))
            noisy = img_in + shot + read
        elif self.args.noise == 'Uniform_Gaussian_noise':
            sig_read = np.random.choice(noise_std, size=[self.batch_size, 1, 1, 1])
            read = max_img * sig_read * tf.random_normal(tf.shape(img_in))
            noisy = img_in + read
        else:
            raise IOError('args.noise should be bpn_noise or Uniform_Gaussian_noise')
        if self.args.clip:
            noisy = tf.clip_by_value(noisy, clip_value_min=0, clip_value_max=1)
        return noisy


