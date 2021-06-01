import numpy as np
import cv2
from global_params import*


def get_pics(node, env):
    img_shape = env.img_shape
    pic = np.zeros((img_shape[0], img_shape[1], img_shape[2], num_frames))
    for k in range(num_frames):
        pic[:, :, :, k] = env.pics[node.snapshot]
        if node.parent is None:
            break
        node = node.parent

    return pic


def format_pics(train_pics):
    if len(train_pics.shape) == 4:
        new_train_pics = np.zeros((1, 64, 64, 4))
    else:
        new_train_pics = np.zeros((len(train_pics), state_dim[0], state_dim[1], state_dim[2]))
    for k in (range(len(new_train_pics))):
        for f in range(num_frames):
            if len(train_pics.shape) == 4:
                img = train_pics[:, :, :, f]
            else:
                img = train_pics[k, :, :, :, f]
            crop_image = img[34:-16, :, :]
            gray_image = crop_image.mean(-1, keepdims=True)
            resize_image = cv2.resize(gray_image, (64, 64))
            new_train_pics[k, :, :, f] = np.asarray(resize_image[..., np.newaxis] / 255.0, dtype='float32')[:, :, 0]

    return new_train_pics


def flip_horizontal(train_pics, train_actions, train_value, train_action_value):
    # flipping images to augment the data.
    new_train_pics = np.zeros(train_pics.shape)
    new_train_actions = np.zeros(train_actions.shape)
    new_train_value = np.zeros(train_value.shape)
    new_train_action_value = np.zeros(train_action_value.shape)
    print('augmenting data...')
    for k in range(len(train_pics)):
        new_train_pics[k, :, :, :] = cv2.flip(train_pics[k, :, :, :], 1)
        new_train_actions[k, :] = train_actions[k, :]
        t = new_train_actions[k, 2]
        new_train_actions[k, 2] = new_train_actions[k, 3]
        new_train_actions[k, 3] = t
        new_train_value[k] = train_value[k]
        new_train_action_value[k] = train_action_value[k]
    return new_train_pics, new_train_actions, new_train_value, new_train_action_value