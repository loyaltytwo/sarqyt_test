import re
import string
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image


def getNumber(image_path):
    image = Image.open(image_path).convert('LA')

    img = np.asarray(image)[:, :, 0]

    img = cv2.medianBlur(img, 3)

    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    image = Image.fromarray(img)
    # image.show()

    str_data = pytesseract.image_to_string(image,
                                           config=f' -c tessedit_char_whitelist={string.digits + string.ascii_letters}.,:;#-')
    # print(str_data)

    pattern = re.compile(r'(no|#)\.?[\s:]+([\d\w-]+)', re.IGNORECASE)

    match = pattern.search(str_data)

    results = []

    while match is not None:
        results.append([match.group(0), match.group(2)])
        match = pattern.search(str_data, match.span(0)[1])

    return results


if __name__ == '__main__':
    folder = Path(
            '/Users/contactone/Desktop/sarqyt_test/SROIE2019/0325updated.task2train(626p)')


    for img in folder.glob('*.jpg'):
        numbers = getNumber(img)
        print(numbers)
        if len(numbers) == 0:
            Image.open(img).show()