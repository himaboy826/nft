import cv2
import numpy as np
import random

###mix body and head
img1 = cv2.imread('head.png')
img2 = cv2.imread('body.png')

rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols]

img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

img1_bg = cv2.bitwise_and(roi,roi,mask = mask)
img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)

layer1 = cv2.add(img1_bg,img2_fg)
#img1[0:rows, 0:cols] = dst

###mix background
'''
rows, cols, channels = layer1.shape
layer1_gray = cv2.cvtColor(layer1, cv2.COLOR_BGR2HSV)
low_value = np.array([0, 0, 221])
high_value = np.array([180, 30, 255])
binary_img = cv2.inRange(layer1_gray, low_value, high_value)
erode = cv2.erode(binary_img, None, iterations=1)
dilate = cv2.dilate(erode, None, iterations=1)
for i in range(rows):
    for j in range(cols):
        if dilate[i, j] == 255:
            layer1[i, j] = (0, 0, 255)
cv2.imwrite('test.png', layer1)
'''
#mix eyes
eyes_num = ['1', '2', '3', '4']
for i in eyes_num:
    print(i)
    filename = 'eyes'+ i +'.png'
    print(filename)
    img_eyes = cv2.imread(filename)
    rows,cols,channels = img_eyes.shape
    roi = layer1[0:rows, 0:cols]
    img_eyes_gray = cv2.cvtColor(img_eyes,cv2.COLOR_BGR2GRAY)
    ret, eyes_mask = cv2.threshold(img_eyes_gray, 254, 255, cv2.THRESH_BINARY)
    eyes_mask_inv = cv2.bitwise_not(eyes_mask)
    img_end = cv2.bitwise_and(roi,roi,mask = eyes_mask)
    img_top = cv2.bitwise_and(img_eyes,img_eyes,mask = eyes_mask_inv)
    layer2 = cv2.add(img_end,img_top)
    #dst = cv2.add(img_end,img_top)
    #cv2.imwrite('output/nft'+i+'.png', dst)
    
    #mix mouth
    mouth_num = ['1', '2', '3', '4']
    for j in mouth_num:
        print(j)
        filename = 'mouth'+ j +'.png'
        print(filename)
        img_mouth = cv2.imread(filename)
        rows,cols,channels = img_mouth.shape
        roi = layer2[0:rows, 0:cols]
        img_mouth_gray = cv2.cvtColor(img_mouth,cv2.COLOR_BGR2GRAY)
        ret, mouth_mask = cv2.threshold(img_mouth_gray, 254, 255, cv2.THRESH_BINARY)
        mouth_mask_inv = cv2.bitwise_not(mouth_mask)
        img_end = cv2.bitwise_and(roi,roi,mask = mouth_mask)
        img_top = cv2.bitwise_and(img_mouth,img_mouth,mask = mouth_mask_inv)
        layer3 = cv2.add(img_end,img_top)
        #dst = cv2.add(img_end,img_top)
        #cv2.imwrite('output/nft'+i+j+'.png', dst)
        #mix camera
        camera_num = ['1', '2', '3', '4']
        for k in camera_num:
            print(k)
            filename = 'camera'+ k +'.png'
            print(filename)
            img_camera = cv2.imread(filename)
            rows,cols,channels = img_camera.shape
            roi = layer3[0:rows, 0:cols]
            img_camera_gray = cv2.cvtColor(img_camera,cv2.COLOR_BGR2GRAY)
            ret, camera_mask = cv2.threshold(img_camera_gray, 254, 255, cv2.THRESH_BINARY)
            camera_mask_inv = cv2.bitwise_not(camera_mask)
            img_end = cv2.bitwise_and(roi,roi,mask = camera_mask)
            img_top = cv2.bitwise_and(img_camera,img_camera,mask = camera_mask_inv)
            layer4 = cv2.add(img_end,img_top)
            #dst = cv2.add(img_end,img_top)
            #cv2.imwrite('output/nft'+i+j+'.png', dst)
            #mix acc
            acc_num = ['1', '2', '3']
            for z in acc_num:
                print(z)
                filename = 'acc'+ z +'.png'
                print(filename)
                img_acc = cv2.imread(filename)
                rows,cols,channels = img_acc.shape
                roi = layer4[0:rows, 0:cols]
                img_acc_gray = cv2.cvtColor(img_acc,cv2.COLOR_BGR2GRAY)
                ret, acc_mask = cv2.threshold(img_acc_gray, 254, 255, cv2.THRESH_BINARY)
                acc_mask_inv = cv2.bitwise_not(acc_mask)
                img_end = cv2.bitwise_and(roi,roi,mask = acc_mask)
                img_top = cv2.bitwise_and(img_acc,img_acc,mask = acc_mask_inv)
                #layer5 = cv2.add(img_end,img_top)
                dst = cv2.add(img_end,img_top)
                #cv2.imwrite('output/nft'+i+j+k+z+'.png', dst)


                ###mix background
                rows, cols, channels = dst.shape
                dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
                low_value = np.array([0, 0, 221])
                high_value = np.array([180, 30, 255])
                binary_img = cv2.inRange(dst_gray, low_value, high_value)
                erode = cv2.erode(binary_img, None, iterations=1)
                dilate = cv2.dilate(erode, None, iterations=1)
                color1=random.randrange(0, 255)
                color2=random.randrange(0, 255)
                color3=random.randrange(0, 255)
                for x in range(rows):
                    for y in range(cols):
                        if dilate[x, y] == 255:
                            print(color1,color2,color3)
                            #dst[x, y] = (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255))
                            dst[x, y] = (color1, color2, color3)
                cv2.imwrite('output/nft'+i+j+k+z+'.png', dst)
                #cv2.imwrite('test.png', dst)


'''
img3 = cv2.imread('eyes1.png')
rows,cols,channels = img3.shape
roi = dst[0:rows, 0:cols]

img3gray = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
ret, mask3 = cv2.threshold(img3gray, 254, 255, cv2.THRESH_BINARY)
mask3_inv = cv2.bitwise_not(mask3)

img3_end = cv2.bitwise_and(roi,roi,mask = mask3)
img3_top = cv2.bitwise_and(img3,img3,mask = mask3_inv)
dst = cv2.add(img3_end,img3_top)
'''
#cv2.imshow('res',dst)
#cv2.imwrite('image.png', dst)
#cv2.waitKey(0)
#cv2.destroyAllWindows()