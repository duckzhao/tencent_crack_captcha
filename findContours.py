'''
基于轮廓提取的方法获取到 大图中 缺口位置的对应坐标px
需要：原图（大图），返回的缺口起始坐标y

相较 threshold.py 中的方法，该方法不依赖于滑块缺口边缘的白色区域，通用性较强，但没有经过过滤的原图可能会有许多待筛选轮廓，
因此会耗费大量计算在求轮廓面积和最小外接矩形面积上，结果准确率也还可以。

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
from copy import deepcopy

def show_img(img):
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
    预处理，以便更好的进行轮廓提取
    :param roi_img: 经过在原图处理拿到的 含有缺口图的 目标区域
    '''
    # 高斯模糊，去掉一些尖锐噪音
    gauss_blur = cv2.GaussianBlur(roi_img, ksize=(5, 5), sigmaX=0)
    # show_img(gauss_blur)
    # 边缘提取
    canny_img = cv2.Canny(gauss_blur, 25, 75)
    # 进行膨胀，当缺口边缘线条不连续时可用
    # kernal = np.ones((3, 3), dtype=np.uint8)
    # dilation = cv2.dilate(src=canny_img, kernel=kernal, iterations=1)

    # show_img(canny_img)
    return canny_img

def find_contours(proi_img, roi_img_rbg):
    contours, hierarchy = cv2.findContours(proi_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(roi_img_rbg, contours, -1, (0, 255, 255), 1)
    # show_img(roi_img_rbg)
    draw_img = deepcopy(roi_img_rbg)
    # 存放所有可能的px坐标结果
    px_list = []
    for i in range(len(contours)):
        # 计算当前轮廓的面积
        area = cv2.contourArea(contours[i])
        # print(area)
        # 过滤面积过大和过小的轮廓
        if area > 5000 and area < 9000:
            cv2.drawContours(draw_img, contours, i, (0, 0, 0), 1)
            # 求最大外接矩形
            x, y, w, h = cv2.boundingRect(contours[i])
            # 过滤最小外接矩形面积过大或过小的轮廓
            if not ((w * h) > 12000 or (w * h) < 7000):
                cv2.rectangle(draw_img, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
                # print(w*h)
                show_img(draw_img)
                draw_img = deepcopy(roi_img_rbg)
                px_list.append(x)
    return px_list

def get_px(px_list):
    '''
    将检测到的px结果进行均方求和，获得平均结果作为最终检测结果
    :param px_list:
    :return:
    '''
    if px_list == []:
        print('检测失败')
        return -1
    else:
        answer = sum(px_list) // len(px_list) + int(captcha_width/2)
        print(answer)
        return answer

def run(img_path):
    img = cv2.imread(img_path)
    roi_img_gray, roi_img_rbg = get_roi(img, init_y_list[i])
    proi_img = preprocess_roi(roi_img_gray)
    px_list = find_contours(proi_img, roi_img_rbg)
    px = get_px(px_list)
    return px


if __name__ == '__main__':
    # 验证码图片的宽度
    captcha_width = 680
    # 缺口（小图）起始的坐标 y，请求验证码时返回的json文件中可以获取到
    # 以下为测试文件中的7个缺口起始 y 坐标
    init_y_list = [30, 86, 82, 158, 32, 30, 50, 56, 106, 30]
    for i in range(10):
        run('{}.jpg'.format(i + 1))

'''
answer:
514
476
498
517
489
509
500
'''