'''
基于模板匹配的方法获取到 大图中 缺口位置的对应坐标px
需要：原图（大图），缺口（小图），返回的缺口起始坐标y

该方法主要是利用请求返回的缺口图片和原图中进行相似度的模板匹配，提供两种方法来完成①run：对原图和缺口图都进行canny边缘检测，然后对检测后的边缘图
进行模板匹配；②run2：对原图和缺口图直接进行阈值过滤二值化，因为原图中缺口位置的颜色都比较连续，所以可以较好的过滤出缺口位置，进行匹配。

自己测试的效果是run2好一些。因为验证码数量有限，可能结论不准确，大家可以再自行测试下。

请求滑块的json例子信息如下：其中initx，inity表示用户滑动缺口时的初始位置
cdnPic1: "/hycdn?index=1&image=936943957294983168"
cdnPic2: "/hycdn?index=2&image=936943957294983168"
initx: "68"
inity: "67"
ret: "0"
sess: ""
'''

import cv2
import numpy as np

def alpha2white(img):
    '''
    将透明的缺口图转换为黑色背景的缺口图
    :param img:
    :return:
    '''
    sp = img.shape
    width = sp[0]
    height = sp[1]
    for yh in range(height):
        for xw in range(width):
            color_d = img[xw, yh]
            if (color_d[3] != 255):  # 找到alpha通道不為255的像素
                img[xw, yh] = [0, 0, 0, 0]  # 改變這個像素
    return img


def show_img(img):
    # 摁下 任意键 可以继续执行程序
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyWindow('img')

def get_roi(img, init_y, ksize=136):
    '''
    从原始的大图中截取出 可能存在小图的区域
    :param img: 原始大图
    :param init_y: 请求缺口图时返回的缺口图初始y坐标
    :param ksize: 缺口图的大小
    '''
    (h, w, c) = img.shape
    roi_img_rbg = img[init_y:init_y+ksize, int(w/2):w]
    # show_img(roi_img)
    roi_img_gray = cv2.cvtColor(roi_img_rbg, cv2.COLOR_BGR2GRAY)
    return roi_img_gray, roi_img_rbg


def preprocess_roi(roi_img):
    '''
    对提取的roi区域灰度图片进行进一步的处理，主要是进行边缘检测，以便获得较好的轮廓结果
    :param roi_img:
    :return:
    '''
    # 二值化
    ret, threshold = cv2.threshold(src=roi_img, thresh=230, maxval=255, type=cv2.THRESH_BINARY)
    # 边缘检测
    canny_img = cv2.Canny(threshold, 25, 75)
    # 膨胀
    kernal = np.ones((3, 3), dtype=np.uint8)
    dilation = cv2.dilate(src=canny_img, kernel=kernal, iterations=2)
    erod = cv2.erode(src=dilation, kernel=kernal, iterations=1)
    # show_img(erod)
    return erod

def run(bing_img_path, small_img_path):
    '''
    模板匹配canny边缘检测后的两个图片---匹配线条
    :param bing_img_path:
    :param small_img_path:
    :return:
    '''
    # 读取原始大图，并获得roi区域的灰度结果用于匹配
    big_img = cv2.imread(bing_img_path)
    roi_img_gray, roi_img_rbg = get_roi(big_img, init_y)
    big_canny = cv2.Canny(roi_img_gray, 25, 75)
    show_img(big_canny)

    # 读取匹配的缺口图
    small_img = cv2.imread(small_img_path, -1)
    small_img = cv2.cvtColor(alpha2white(small_img), cv2.COLOR_BGR2GRAY)
    height, width = small_img.shape
    # 边缘检测
    small_canny = cv2.Canny(small_img, 25, 75)
    show_img(small_canny)
    res = cv2.matchTemplate(image=big_canny, templ=small_canny, method=cv2.TM_CCOEFF_NORMED)
    print(np.max(res))
    loc = np.where(res == np.max(res))
    # print(loc)
    # 可能匹配多个目标的固定写法，如果是确定只有一个目标可以用 minMaxLoc 函数处理，从loc中遍历得到的左上坐标 pt -> (10, 10)
    for pt in zip(*loc[::-1]):
        # 指定左上和右下坐标 画矩形
        cv2.rectangle(roi_img_rbg, pt, (pt[0] + width, pt[1] + height), color=(0, 0, 255), thickness=2)
    show_img(roi_img_rbg)


def run2(bing_img_path, small_img_path):
    '''
    模板匹配二值化后的两个图片---匹配图案
    :param bing_img_path:
    :param small_img_path:
    :return:
    '''
    # 读取原始大图，并获得roi区域的灰度结果用于匹配
    big_img = cv2.imread(bing_img_path)
    roi_img_gray, roi_img_rbg = get_roi(big_img, init_y)
    threshold1 = cv2.adaptiveThreshold(src=roi_img_gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      thresholdType=cv2.THRESH_BINARY_INV, blockSize=115, C=1)
    show_img(threshold1)

    # 读取匹配的缺口图
    small_img = cv2.imread(small_img_path, -1)
    small_img = cv2.cvtColor(alpha2white(small_img), cv2.COLOR_BGR2GRAY)
    height, width = small_img.shape
    ret, threshold = cv2.threshold(src=small_img, thresh=1, maxval=255, type=cv2.THRESH_BINARY)
    show_img(threshold)

    res = cv2.matchTemplate(image=threshold1, templ=threshold, method=cv2.TM_CCOEFF_NORMED)
    # print(res)
    loc = np.where(res == np.max(res))
    # 可能匹配多个目标的固定写法，如果是确定只有一个目标可以用 minMaxLoc 函数处理，从loc中遍历得到的左上坐标 pt -> (10, 10)
    for pt in zip(*loc[::-1]):
        # 指定左上和右下坐标 画矩形
        cv2.rectangle(roi_img_rbg, pt, (pt[0] + width, pt[1] + height), color=(0, 0, 255), thickness=2)
    show_img(roi_img_rbg)

if __name__ == '__main__':
    # 验证码图片的宽度
    captcha_width = 680
    # 缺口（小图）起始的坐标 y，请求验证码时返回的json文件中可以获取到
    # 以下为测试文件中的10个缺口起始 y 坐标
    init_y = 30
    run2('./10.jpg', '10_1.png')