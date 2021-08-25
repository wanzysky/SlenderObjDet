import cv2


def encode_image(img):
    return cv2.imencode('.jpg', img)[1].tostring()
