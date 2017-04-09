from PIL import Image, ImageFilter


def read_image(image_path):
    im = Image.open(image_path).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    new_image = Image.new('L', (28, 28), 255)

    # Width 가 더 크면
    if width > height:
        # 가로 세로 비율에 따라 Resize --> 20 x 20
        height = int(round((20.0 / width * height), 0))

        if height == 0:
            height = 1

        img = im.resize((20, height), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - height) / 2), 0))

        new_image.paste(img, (4, wtop))
    else:
        width = int(round((20.0 / height * width), 0))

        if width == 0:
            width = 1

        img = im.resize((width, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - width) / 2), 0))
        new_image.paste(img, (wleft, 4))

    tv = list(new_image.getdata())

    # Pixel 값을 Normalize (0 ~ 1)
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva
