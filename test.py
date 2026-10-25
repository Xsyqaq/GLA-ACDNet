import os

import numpy as np

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import tensor2im,save_image
from PIL import Image


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch),'images')  # define the website directory

    if not os.path.exists(dir):
      os.makedirs(dir)
    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        img_path, houzhui = os.path.splitext(img_path[0])
        img_basename = os.path.basename(img_path)
        for label, image in visuals.items():
            image_numpy = tensor2im(image)
            image_numpy = Image.fromarray(image_numpy)
            image_numpy = np.array(image_numpy, dtype='uint8')

            path = os.path.join(dir, '%s_%s.png' % (img_basename, label))
            save_image(image_numpy, path)

