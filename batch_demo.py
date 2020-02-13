from __future__ import absolute_import
import sys

sys.path.append('./')

import argparse
import os
import os.path as osp
import numpy as np
import math
import time
from PIL import Image, ImageFile
from tqdm import tqdm
import pytesseract

import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from config import get_args
from lib import datasets, evaluation_metrics, models
from lib.models.model_builder import ModelBuilder
from lib.datasets.dataset import LmdbDataset, AlignCollate
from lib.loss import SequenceCrossEntropyLoss
from lib.trainers import Trainer
from lib.evaluators import Evaluator
from lib.utils.logging import Logger, TFLogger
from lib.utils.serialization import load_checkpoint, save_checkpoint
from lib.utils.osutils import make_symlink_if_not_exists
from lib.evaluation_metrics.metrics import get_str_list
from lib.utils.labelmaps import get_vocabulary, labels2strs

global_args = get_args(sys.argv[1:])
global_args.STN_ON = True
global_args.with_lstm = True


def image_process(image_path, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
    img = Image.open(image_path).convert('RGB')

    if keep_ratio:
        w, h = img.size
        ratio = w / float(h)
        imgW = int(np.floor(ratio * imgH))
        imgW = max(imgH * min_ratio, imgW)

    img = img.resize((imgW, imgH), Image.BILINEAR)
    img = transforms.ToTensor()(img)
    img.sub_(0.5).div_(0.5)

    return img

def load_image_in_PIL(path):
    img = Image.open(path)
    img.load()  # Very important for loading large image
    return img

class DataInfo(object):
    """
    Save the info about the dataset.
    This a code snippet from dataset.py
    """

    def __init__(self, voc_type):
        super(DataInfo, self).__init__()
        self.voc_type = voc_type

        assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
        self.EOS = 'EOS'
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))

        self.rec_num_classes = len(self.voc)


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        print('using cuda.')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (32, 100)

    dataset_info = DataInfo(args.voc_type)
    print(dataset_info.char2id)

    # Create model
    model = ModelBuilder(arch=args.arch, rec_num_classes=dataset_info.rec_num_classes,
                         sDim=args.decoder_sdim, attDim=args.attDim, max_len_labels=args.max_len,
                         eos=dataset_info.char2id[dataset_info.EOS], STN_ON=args.STN_ON)

    # Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

    if args.cuda:
        device = torch.device("cuda")
        model = model.to(device)
        model = nn.DataParallel(model)

    # Evaluation
    model.eval()

    try:
        test_list_file = open(os.path.join(args.image_path, 'annotation_test.txt'),  'r')
        test_list = test_list_file.read().splitlines()
        test_list_file.close()
    except IOError:
        test_list = os.listdir(args.image_path)

    # print(test_list)
    data_n = min(100, len(test_list))
    aster_correct_cnt = 0
    tesseract_correct_cnt = 0

    custom_oem_psm_config = '--oem 3 --psm 7'

    for test_name in tqdm(test_list[:data_n]):

        img_path = os.path.join(args.image_path, test_name).split(' ')[0]
        target_str = img_path.split('_')[-2]
        print(img_path, target_str)

        img = image_process(img_path)
        with torch.no_grad():
            img = img.to(device)
        input_dict = {}
        input_dict['images'] = img.unsqueeze(0)
        # TODO: testing should be more clean.
        # to be compatible with the lmdb-based testing, need to construct some meaningless variables.
        rec_targets = torch.IntTensor(1, args.max_len).fill_(1)
        rec_targets[:, args.max_len - 1] = dataset_info.char2id[dataset_info.EOS]
        input_dict['rec_targets'] = rec_targets
        input_dict['rec_lengths'] = [args.max_len]
        output_dict = model(input_dict)
        pred_rec = output_dict['output']['pred_rec']
        # print(pred_rec)
        pred_str, _ = get_str_list(pred_rec, input_dict['rec_targets'], dataset=dataset_info, lower_flag=False)
        if pred_str[0] == target_str:
            aster_correct_cnt += 1

        img = load_image_in_PIL(img_path).convert('RGB')
        detected_str = pytesseract.image_to_string(img, config=custom_oem_psm_config)
        # print(i, detected_str,  dataset_info['id2char'][predicted[i].item()], dataset_info['id2char'][sample['target'][i].item()])
        if detected_str == target_str:
            tesseract_correct_cnt += 1

        print(f'GT: {target_str}, ASTER: {pred_str[0]}, Tesseract: {detected_str}')
        if detected_str == target_str:
            print('===================== correct')

    print(f'Aster acc: {aster_correct_cnt} / {data_n}. {aster_correct_cnt/data_n}')
    print(f'Tesseract acc: {tesseract_correct_cnt} / {data_n}. {tesseract_correct_cnt/data_n}')


if __name__ == '__main__':
    # parse the config
    args = get_args(sys.argv[1:])
    args.STN_ON = True
    args.with_lstm = True

    main(args)