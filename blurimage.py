from PIL import Image, ImageFilter
import os
# MediaList = ['carbonate2D','coal2D','sandstone2D']
# MediaList = ['shuffled2D']
import cv2
DataDivide=['_test_HR','_train_HR','_valid_HR']
PATH = '/data/DeepRockSR-2D_copy/DeepRockSR-2D'



for d in range(len(DataDivide)):
    print(d)
    longPath=PATH+'/'+'shuffled2D'+'/'+'shuffled2D'+DataDivide[d]
    # try:
    #     os.mkdir(longPath+'B_G_BB9')
    # except:
    #     continue
    MediaPathList = os.listdir(longPath)
    # longPath = DeepRockSR-2D/DeepRockSR-2D/carbonate2D/carbonate2D_test_HR
    #  MediaPathList 400 ['3797.png', '3814.png', '3765.png', '4000.png', '3918.png', '3731.png', '3829.png', '3905.png', '3723.png', '3947.png']
    for i in range(len(MediaPathList)):
        # 1 打开文件, 返回一个文件对象;
        im = cv2.imread(longPath+'/'+MediaPathList[i])

        # 2. 对图片进行模糊效果
        # im2 = im.filter(ImageFilter.BLUR)

        # clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(4, 4))
        # im2 = clahe.apply(im)
        im2 = cv2.GaussianBlur(im, (5, 5), 1.5)

        # im2 = im2.filter(ImageFilter.BoxBlur(radius=9))

        cv2.imwrite(longPath+'B_G_BB9'+'/'+'B_G_BB9'+MediaPathList[i], im2)
        # print(i)

