import albumentations as A

def get_train_transforms(img_size=640):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.MotionBlur(p=0.2),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_val_transforms(img_size=640):
    return A.Compose([
        A.Resize(img_size, img_size),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
