import torch
import numpy as np
from models import spinal_net
import cv2
import decoder
import os
from dataset import BaseDataset
import draw_points
# from boostnet_labeldata.tools import draw_gt_pts3
import argparse

def apply_mask(image, mask, alpha=0.5):
    """Apply the given mask to the image.
    """
    color = np.random.rand(3)
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

class Network(object):
    def __init__(self, args):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        heads = {'hm': args.num_classes,
                 'reg': 2*args.num_classes,
                 'wh': 2*4,}
                 # 'mid_point':2*args.num_classes,}

        self.model = spinal_net.SpineNet(heads=heads,
                                         pretrained=True,
                                         down_ratio=args.down_ratio,
                                         final_kernel=1,
                                         head_conv=256)
        self.num_classes = args.num_classes
        self.decoder = decoder.DecDecoder(K=args.K, conf_thresh=args.conf_thresh)
        self.dataset = {'spinal': BaseDataset}

    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model

    def map_mask_to_image(self, mask, img, color=None):
        if color is None:
            color = np.random.rand(3)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mskd = img * mask
        clmsk = np.ones(mask.shape) * mask
        clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
        clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
        clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
        img = img + 1. * clmsk - 1. * mskd
        return np.uint8(img)


    def test(self, args, save):
        save_path = args.weights_dir  # +args.dataset  #_spinal_12.7(best)
        self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        self.model = self.model.to(self.device)
        self.model.eval()

        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir,
                               phase='test',
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=args.down_ratio)

        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)


        for cnt, data_dict in enumerate(data_loader):
            images = data_dict['images'][0]
            img_id = data_dict['img_id'][0]
            images = images.to('cuda')
            print('processing {}/{} image ... {}'.format(cnt, len(data_loader), img_id))
            with torch.no_grad():
                output = self.model(images)
                # print(output)
                hm = output['hm']  #
                wh = output['wh']
                reg = output['reg']

            torch.cuda.synchronize(self.device)
            pts2 = self.decoder.ctdet_decode(hm, wh, reg)   # 17, 11
            # print(pts2.shape)
            pts0 = pts2.copy()
            pts0[:, :10] *= args.down_ratio

            print('total pts num is {}'.format(len(pts2)))

            ori_image = dsets.load_image(dsets.img_ids.index(img_id))
            # ori_image2 = draw_gt_pts3(img_id,ori_image) #  画原图
            # ori_image2 = cv2.resize(ori_image2, (args.input_w, args.input_h))

            ori_image_regress = cv2.resize(ori_image, (args.input_w, args.input_h))
            ori_image_points = ori_image_regress.copy()
            ori_image_heatmap =ori_image_regress.copy()

            h,w,c = ori_image.shape
            pts0 = np.asarray(pts0, np.float32)
            # pts0[:,0::2] = pts0[:,0::2]/args.input_w*w
            # pts0[:,1::2] = pts0[:,1::2]/args.input_h*h
            sort_ind = np.argsort(pts0[:,1])
            # print(sort_ind)
            pts0 = pts0[sort_ind]
            # print(pts0)

            ori_image_heatmap, ori_image_regress, ori_image_points = \
                draw_points.draw_landmarks_regress_test(pts0,ori_image_heatmap,ori_image_regress,ori_image_points)

            ori_image_heatmap = cv2.resize(ori_image_heatmap, (512, 800))
            ori_image_regress = cv2.resize(ori_image_regress, (512, 800))
            ori_image_points = cv2.resize(ori_image_points, (512, 800))

            cv2.imshow('ori_image_heatmap', ori_image_heatmap)
            cv2.imshow('ori_image_regress', ori_image_regress)
            cv2.imshow('ori_image_points', ori_image_points)
            # cv2.imshow('gt_image_points', ori_image2)
            k = cv2.waitKey(0) & 0xFF
            if k == ord('q'):
                cv2.destroyAllWindows()
                exit()

def parse_args():
    parser = argparse.ArgumentParser(description='CenterNet Modification Implementation')
    parser.add_argument('--transfer', type=bool, default=False, help='transfer train flag')
    parser.add_argument('--num_epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=1.25e-4, help='Init learning rate')  # 1.25e-4
    parser.add_argument('--down_ratio', type=int, default=4, help='down ratio')
    parser.add_argument('--input_h', type=int, default=1024, help='input height')
    parser.add_argument('--input_w', type=int, default=512, help='input width')
    parser.add_argument('--K', type=int, default=17, help='maximum of objects')
    parser.add_argument('--conf_thresh', type=float, default=0.2, help='confidence threshold')
    parser.add_argument('--seg_thresh', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--ngpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--resume', type=str, default='model_last_2020-11-16 15:55:16.pth', help='weights to be resumed')
    parser.add_argument('--weights_dir', type=str, default='weights_spinal', help='weights dir')
    parser.add_argument('--data_dir', type=str, default='Datasets/', help='data directory')
    parser.add_argument('--phase', type=str, default='test', help='data directory')
    parser.add_argument('--dataset', type=str, default='spinal', help='data directory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    is_object = Network(args)
    is_object.test(args, save=False)