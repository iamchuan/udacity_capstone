import os
import re
import imageio

from torch import nn


def weights_init(m):
    # init weight
    if issubclass(m.__class__, nn.modules.conv._ConvNd):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        # init bias
        if m.bias.data is not None:
            nn.init.constant_(m.bias.data, 0)
    elif issubclass(m.__class__, nn.modules.batchnorm._BatchNorm):
        nn.init.ones_(m.weight.data)


def gif_generator(image_dir, name_list=[], name_pattern=None, output_name='animation.gif', max_frames=-1):
    images = []
    if name_list:
        for img_name in name_list:
            img_path = os.path.join(image_dir, img_name)
            if os.path.isfile(img_path):
                images.append(imageio.imread(img_path))
    elif name_pattern:
        for file in sorted(os.listdir(path=image_dir)):
            if re.match(name_pattern, file):
                images.append(imageio.imread(os.path.join(image_dir, file)))

    if max_frames > 0:
        images = images[:max_frames]
    imageio.mimwrite(os.path.join(image_dir, output_name), images, duration=.1)
