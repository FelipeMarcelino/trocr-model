import random

from PIL import Image, ImageEnhance, ImageFilter


def augment_image(image):
    """Aplica aumentos de dados aleatórios na imagem"""
    aug_type = random.choice(["blur", "contrast", "brightness", "rotate", "none"])

    if aug_type == "blur":
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    elif aug_type == "contrast":
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
    elif aug_type == "brightness":
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
    elif aug_type == "rotate":
        angle = random.uniform(-5, 5)
        image = image.rotate(angle, resample=Image.BICUBIC, expand=False)
