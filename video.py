import cv2
from numpy.lib.stride_tricks import _broadcast_arrays_dispatcher
from tensorflow.python.framework.tensor_conversion_registry import get

import numpy as np
from numpy.lib.npyio import load
from tensorflow import keras
from tensorflow.keras import models
import tensorflow as tf
from tensorflow.keras.applications import vgg19
from PIL import Image
import matplotlib.pyplot as plt

def load_img(path):
    img = Image.open(path)
    long = max(img.size)
    scale = 512/long
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
    img = keras.preprocessing.image.img_to_array(img)

    img = np.expand_dims(img, axis=0)
    return img

def cv2pil(cv_img):
    color_conv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(color_conv)
    long = max(img.size)
    scale = 512/long
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
    img = keras.preprocessing.image.img_to_array(img)

    img = np.expand_dims(img, axis=0)
    return img

def preprocess(img):
    return vgg19.preprocess_input(img)

def deprocess(img):
    x = img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


def get_model():
    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1"
    ]
    content_layers = ["block5_conv2"]
    vgg = vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layer]
    model_outputs = style_outputs + content_outputs
    return models.Model(vgg.input, model_outputs)

def content_loss(base, target):
    return tf.reduce_mean(tf.square(base, target))

def gram_matrix(input):
    channels = int(input.shape[-1])
    a = tf.reshape(input, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram/tf.cast(n, tf.float32)

def style_loss(base, gram):
    h, w, c = base.get_shape().as_list()
    gram_style = gram_matrix(base)
    return tf.reduce_mean(tf.square(gram_style - gram))

style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
content_layer = ["block5_conv2"]

def get_features(model, content, style, vid=False):
    if vid:
        content_img = preprocess(cv2pil(content))
    else:
        content_img = preprocess(load_img(content))
    style_img = preprocess(load_img(style))
    style_outs = model(style_img)
    content_outs = model(content_img)

    style_feats = [style_layer[0] for style_layer in style_outs[:5]]
    content_feats = [content_layer[0] for content_layer in content_outs[5:]]
    return style_feats, content_feats

def compute_loss(model, loss_weights, init_img, gram_style_feats, content_feats):
    style_weight, content_weight = loss_weights

    model_outs = model(init_img)
    style_out_feats = model_outs[:5]
    content_out_feats = model_outs[5:]
    style_score, content_score = 0, 0
    wt_style_layer = 1/float(5)
    for target_style, comb_style in zip(gram_style_feats, style_out_feats):
        style_score += wt_style_layer * style_loss(comb_style[0], target_style)

    wt_content_layer = 1/float(1)
    for target_cont, comb_cont in zip(content_feats, content_out_feats):
        content_score += wt_content_layer * content_loss(comb_cont[0], target_cont)

    style_score *= style_weight
    content_score *= content_weight

    loss = style_score + content_score
    return loss, style_score, content_score

def compute_gradients(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)

    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_img']), all_loss


def transfer(content, style_path, num_iters=1000, content_wt=1e3, style_wt=1e-2, vid=False):
    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    style_feats, content_feats = get_features(model, content, style_path, vid=vid)
    gram_style_feats = [gram_matrix(style_feat) for style_feat in style_feats]
    if vid:
        init_img = preprocess(cv2pil(content))
    else:
        init_img = preprocess(load_img(content))
    init_img = tf.Variable(init_img, dtype=tf.float32)
    opt = tf.keras.optimizers.Adam(learning_rate=.5, beta_1=0.99, epsilon=1e-1)
    iters = 1
    best_loss, best_img = float('inf'), None
    loss_wts = (style_wt, content_wt)

    cfg = {
        'model': model,
        'loss_weights': loss_wts,
        'init_img': init_img,
        'gram_style_feats': gram_style_feats,
        'content_feats': content_feats
    }

    rows, cols = 2, 5
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    imgs = []
    for i in range(num_iters):
        if i%100==0:
            print('Progress: ', i/num_iters)
        grads, all_loss = compute_gradients(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_img)])
        clipped = tf.clip_by_value(init_img, min_vals, max_vals)
        init_img.assign(clipped)

        if loss < best_loss:
            best_loss = loss
            best_img = deprocess(init_img.numpy())

    return best_img, best_loss

def show_results(best_img, res):
    plt.imshow(best_img)
    # plt.show()
    plt.savefig('{}.png'.format(res), bbox_inches='tight')

def main_vid(vid, style):
    cap = cv2.VideoCapture(vid)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    long = max(frame_width, frame_height)
    scale = 512/long
    frame_width = round(frame_width*scale)
    frame_height = round(frame_height*scale)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_height, frame_width))
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()
        if ret:
            best_img, best_loss = transfer(frame, style, vid=True)
            out.write(best_img)
        else:
            break
    cap.release()
    out.release()

def main_pic(content, style):
    best_img, best_loss = transfer(content, style)
    print(best_img.shape)
    show_results(best_img, 'chistar')

if __name__ == '__main__':
    content = 'chicago.jpg'
    style = 'wave.jpg'
    main_pic(content, style)