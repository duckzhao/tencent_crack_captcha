# tencent_crack_captcha
腾讯滑块验证码缺口坐标识别，基于opencv-python

## 腾讯滑块
以下三个文件均为识别腾讯滑块缺口距离的方案，都基于opencv-python完成，使用模板匹配或者轮廓检测方案。具体的效果没有测试过，不过比很多网上的代码应该好一些。欢迎大家测试后给我反馈。
1. findContours.py
2. threshold.py
3. matchTemplate.py

https://007.qq.com/captcha

## 网易滑块
以下文件为网易易盾滑块的缺口识别方案，同样欢迎大家测试效果。
1. wangyi/yd_hk.py

https://dun.163.com/trial/jigsaw

## 环境
python 3.x
opencv-python 4.x.x
numpy 1.20.x

## 最后
有空的话我会准备一下汉字点选的数据集，然后使用yolo训练一个检测模型，使用分类网络完成汉字识别，如果各位带佬有数据集，欢迎大家分享给我帮大家训练，一起讨论模型优化。邮箱：1161678627@qq.com