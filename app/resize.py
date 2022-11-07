import cv2
import numpy as np
import math
import os

def resize(src_file, width, height):
    """
    画像ファイルを与えられたサイズにサイズ変換する関数
    アス比は固定、左上に寄せる、余った部分はゼロで埋める

    参照URL: https://skattun.hatenablog.jp/entry/2019/04/14/182411

    Returns
    -------
    ret_scale : float
    拡大/縮小した倍率
    """

    # src_img = cv2.imread(src_file)
    src_img = cv2.imdecode(src_file, cv2.IMREAD_COLOR)
    h, w, c = src_img.shape

    ret_scale = height / width
    img_scale = h / w
    if  ret_scale < img_scale:
        dst_img = np.zeros((h, math.ceil(h / ret_scale), 3), dtype = np.uint8)
    else:
        dst_img = np.zeros((math.ceil(w * ret_scale), w, 3), dtype = np.uint8)

    resize_img = src_img

    # dst_imgにresize_imgを合成
    h, w, c = resize_img.shape
    dst_img[0:h, 0:w] = resize_img

    print(resize_img.shape)
    print(dst_img.shape)

    return dst_img, img_scale, h, w #ret_scale


def scale_to_height(img, height, h0 = 1, w0 = 1):
    """
    高さが指定した値になるように、アスペクト比を固定して、リサイズする。
    h0, w0: オリジナル画像の縦横サイズ
    h, w: resize後画像の縦横サイズ
    """
    h, w = img.shape[:2]
    # print(h,w)
    width = round(w * (height / h))
    dst = cv2.resize(img, dsize=(width, height))
    
    # オリジナル画像の縦横をスケールする
    h_out = round(h0 * (height / h))
    w_out = round(w0 * (width / w))

    # print(h_out, w_out)

    return dst, h_out, w_out


def resize_test(src_file, dst_file, width, height):
    """
    テスト出力用
    """
    # dst_img, ret_scale, h, w = resize(src_file, width, height)
    dst_img = cv2.imdecode(src_file, cv2.IMREAD_COLOR)

    dst_img, h_out, w_out = scale_to_height(dst_img, 640)

    cv2.imwrite(dst_file, dst_img)

    return dst_img.shape



if __name__ == '__main__':
    DIR = '../lib/pose_sample'
    jpg_num = sum(os.path.isfile(os.path.join(DIR, name)) for name in os.listdir(DIR))

    for i in range(0,jpg_num):
        scale = resize_test('../lib/pose_sample/pose{}.jpg'.format(i), '../lib/resize_sample/resize_pose{}.jpg'.format(i), 640, 480)
        print("No.{}: {}".format(i,scale))