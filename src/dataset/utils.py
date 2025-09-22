import cv2
import numpy as np
import random

def augment_image(img):
    # Случайное отражение
    if random.random() > 0.5:
        img = cv2.flip(img, 1)

    # Яркость/контраст
    if random.random() > 0.5:
        alpha = 1.0 + (0.2 * (random.random() - 0.5))
        beta = 10 * (random.random() - 0.5)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Шум
    if random.random() > 0.5:
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)

    return img
