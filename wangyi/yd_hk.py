'''
基于模板匹配的方法获取到 大图中 缺口位置的对应坐标px
需要：原图（大图），缺口（小图）

该方法主要是利用请求返回的缺口图片和原图中进行相似度的模板匹配，对原图和缺口图都进行canny边缘检测，然后对检测后的边缘图进行模板匹配。
为了降低模板匹配的错误率，利用小图中的滑块颜色变化，计算出了滑块对应的起始y坐标，然后在大图中提取出了相应的roi区域，缩小决策空间。

易盾相对腾讯滑块麻烦一点，滑块图片背景的干扰项也更多，有多个缺口位置，因此过滤y坐标还是比较重要的。其次易盾背景图的线条、颜色也比较复杂，
计算轮廓面积，提取轮廓的方法都不是很好直接套用。

当然以上麻烦对于深度学习模型都可以很简单解决。
'''

import cv2
import numpy as np

def show_img(img):
    # 摁下 任意键 可以继续执行程序
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyWindow('img')

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

def find_part_about_hk(img, safe_space=5):
    '''
    找到缺口在原图上对应的 y坐标 的取值范围
    :param img: 处理过透明度后的缺口图
    :return:
    '''
    # h158 * w60
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    part_of_index = np.where(gray_img.sum(axis=1) > 0)
    # print(part_of_index)
    # 得到滑块对应的区域大小
    min_part = list(part_of_index)[0][0]
    max_part = list(part_of_index)[0][-1]
    # print(min_part, max_part)

    hk_img = img[min_part-safe_space:max_part+safe_space, :]
    # show_img(hk_img)

    return min_part, max_part, hk_img

def get_roi(img, init_min_y, init_max_y, safe_space=5):
    '''
    从原始的大图中截取出 可能存在小图的区域
    :param img: 原始大图
    :param init_y: 请求缺口图时返回的缺口图初始y坐标
    :param ksize: 缺口图的大小
    '''
    roi_img_rbg = img[init_min_y-safe_space+2:init_max_y+safe_space+2, :]
    # show_img(roi_img_rbg)
    roi_img_gray = cv2.cvtColor(roi_img_rbg, cv2.COLOR_BGR2GRAY)
    return roi_img_gray, roi_img_rbg


def run(img_path):
    small_img = cv2.imread('./{}.png'.format(img_path), -1)
    small_img = alpha2white(small_img)
    # show_img(small_img)
    min_part, max_part, hk_img = find_part_about_hk(small_img)
    h, w, c = hk_img.shape

    big_img = cv2.imread('./{}.jpg'.format(img_path))
    roi_img_gray, roi_img_rbg = get_roi(big_img, min_part, max_part)

    # 进行边缘检测
    roi_img_canny = cv2.Canny(roi_img_gray, 300, 300)
    hk_img_canny = cv2.Canny(cv2.cvtColor(hk_img, cv2.COLOR_BGR2GRAY), 400, 400)

    cv2.imshow('1', roi_img_canny)
    cv2.imshow('2', hk_img_canny)

    res = cv2.matchTemplate(image=roi_img_canny, templ=hk_img_canny, method=cv2.TM_CCOEFF_NORMED)
    # print(res)
    print(np.max(res))  # 匹配的最大概率
    loc = np.where(res == np.max(res))

    # 可能匹配多个目标的固定写法，如果是确定只有一个目标可以用 minMaxLoc 函数处理，从loc中遍历得到的左上坐标 pt -> (10, 10)
    for pt in zip(*loc[::-1]):
        # 指定左上和右下坐标 画矩形
        cv2.rectangle(roi_img_rbg, pt, (pt[0] + w, pt[1] + h), color=(0, 0, 255), thickness=2)
        print(pt[0])
    show_img(roi_img_rbg)


if __name__ == '__main__':
    for i in range(11):
        run(i+1)