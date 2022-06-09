import albumentations as A
import cv2
import numpy as np
import torch
from torchvision import transforms
from albumentations.pytorch import ToTensorV2


CROP_SIZE = 224
DATASET_MEAN = [0.485, 0.456, 0.406]
DATASET_STD = [0.229, 0.224, 0.225]


class ScaleMinSideToSize(object):
    def __init__(self, size, elem_name='image'):
        # self.size = torch.tensor(size, dtype=torch.float)
        self.size = np.asarray(size, dtype=float)
        self.elem_name = elem_name

    def __call__(self, sample):
        h, w, _ = sample[self.elem_name].shape
        if h > w:
            f = self.size[0] / w
        else:
            f = self.size[1] / h

        sample[self.elem_name] = cv2.resize(sample[self.elem_name], None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        sample["scale_coef"] = f

        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2).float()
            landmarks = landmarks * f
            sample['landmarks'] = landmarks.reshape(-1)

        return sample


class CropCenter(object):
    def __init__(self, size=224, elem_name='image'):
        self.size = size
        self.elem_name = elem_name

    def __call__(self, sample):
        img = sample[self.elem_name]
        h, w, _ = img.shape
        margin_h = (h - self.size) // 2
        margin_w = (w - self.size) // 2
        sample[self.elem_name] = img[margin_h:margin_h + self.size, margin_w:margin_w + self.size]
        sample["crop_margin_x"] = margin_w
        sample["crop_margin_y"] = margin_h

        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2)
            landmarks -= torch.tensor((margin_w, margin_h), dtype=landmarks.dtype)[None, :]
            sample['landmarks'] = landmarks.reshape(-1)

        return sample


class TransformByKeys(object):
    def __init__(self, transform, names):
        self.transform = transform
        self.names = set(names)

    def __call__(self, sample):
        for name in self.names:
            if name in sample:
                sample[name] = self.transform(sample[name])

        return sample


class FaceHorizontalFlip(A.HorizontalFlip):
    def apply_to_keypoints(self, keypoints, **params):
        keypoints = np.array(keypoints)
        keypoints[:, 0] = (params['cols'] - 1) - keypoints[:, 0]
        lm = keypoints
        nm = np.zeros_like(lm)
        nm[:64,:]     = lm[64:128,:]     # [  0, 63]  -> [ 64, 127]:  i --> i + 64
        nm[64:128,:]  = lm[:64,:]        # [ 64, 127] -> [  0, 63]:   i --> i - 64
        nm[128:273,:] = lm[272:127:-1,:] # [128, 272] -> [128, 272]:  i --> 400 - i
        nm[273:337,:] = lm[337:401,:]    # [273, 336] -> [337, 400]:  i --> i + 64
        nm[337:401,:] = lm[273:337,:]    # [337, 400] -> [273, 336]:  i --> i - 64
        nm[401:464,:] = lm[464:527,:]    # [401, 463] -> [464, 526]:  i --> i + 64
        nm[464:527,:] = lm[401:464,:]    # [464, 526] -> [401, 463]:  i --> i - 64
        nm[527:587,:] = lm[527:587,:]    # [527, 586] -> [527, 586]:  i --> i
        nm[587:714,:] = lm[714:841,:]    # [587, 713] -> [714, 840]:  i --> i + 127
        nm[714:841,:] = lm[587:714,:]    # [714, 840] -> [587, 713]:  i --> i - 127
        nm[841:873,:] = lm[872:840:-1,:] # [841, 872] -> [841, 872]:  i --> 1713 - i
        nm[873:905,:] = lm[904:872:-1,:] # [873, 904] -> [873, 904]:  i --> 1777 - i
        nm[905:937,:] = lm[936:904:-1,:] # [905, 936] -> [905, 936]:  i --> 1841 - i
        nm[937:969,:] = lm[968:936:-1,:] # [937, 968] -> [937, 968]:  i --> 1905 - i
        nm[969:971,:] = lm[970:968:-1,:] # [969, 970] -> [969, 970]:  i --> 1939 - i
        return nm


test_transforms = transforms.Compose([
        ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
        CropCenter(CROP_SIZE),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD), ("image",)),
    ])
image_augmentations = A.Compose([
    A.RandomBrightnessContrast(p=0.33),
    A.RandomGamma(p=0.33),
    A.CoarseDropout(p=0.3, min_holes=4, max_holes=10, min_width=10, max_width=30, min_height=10, max_height=30),
    A.ToGray(p=0.1),
])
key_points_augmentations = A.Compose([
    FaceHorizontalFlip(p=0.5),
    A.Rotate([-20, 20], p=0.5),
],  keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)
