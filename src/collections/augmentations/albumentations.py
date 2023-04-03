import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms(p=0.5):
    imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    augms = [
        A.Cutout(p=p),
        # A.RandomRotate90(p=p),
        A.Flip(p=p),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2,
                                       ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=50,
                val_shift_limit=50)
        ], p=p),
        # A.OneOf([
        #     A.IAAAdditiveGaussianNoise(),
        #     A.GaussNoise(),
        # ], p=p),
        # A.OneOf([
        #     A.MotionBlur(p=0.2),
        #     A.MedianBlur(blur_limit=3, p=0.1),
        #     A.Blur(blur_limit=3, p=0.1),
        # ], p=p),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=p),
        # A.OneOf([
        #     A.OpticalDistortion(p=0.3),
        #     A.GridDistortion(p=0.1),
        #     A.IAAPiecewiseAffine(p=0.3),
        # ], p=p), 
        A.Normalize(**imagenet_stats),
        # ToTensorV2()
    ]
    return A.Compose(augms)


def get_val_transforms():
    imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    augms = [
        A.Normalize(**imagenet_stats),
        # ToTensorV2()
    ]
    return A.Compose(augms)
