#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2

# for hinge feature
LEG_LENGTH = 25
N_ANGLE_BINS = 12
BIN_SIZE = 360 // N_ANGLE_BINS

#N_ANGLE_BINS라디안의 등장확률
def get_hinge_features(bw_img):

    
   #윤곽선 찾아줌 (윤곽선 개수 , 픽셀 수 , 2(x,y좌표))
    contours, _ = cv2.findContours(
        bw_img, cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]

    hist = np.zeros((N_ANGLE_BINS, N_ANGLE_BINS))

    #cnt : (픽셀수 , (x,y좌표))
    for cnt in contours:
        n_pixels = len(cnt)
        if n_pixels <= LEG_LENGTH:
            continue
        
        #points : 각 픽셀의 좌표들((x1,y1) , (x2,y2) ,,,,,)
        points = np.array([point[0] for point in cnt])
 
        xs, ys = points[:, 0], points[:, 1]
        
        #각 윤곽선을 이루는 좌표들 시계방향 , 반시계 방향
        #무작위 성을 주기 위한 듯
        point_1s = np.array([cnt[(i + LEG_LENGTH) % n_pixels][0]
                            for i in range(n_pixels)])
        point_2s = np.array([cnt[(i - LEG_LENGTH) % n_pixels][0]
                            for i in range(n_pixels)])
        x1s, y1s = point_1s[:, 0], point_1s[:, 1]
        x2s, y2s = point_2s[:, 0], point_2s[:, 1]
        
        #한 점과 윤곽선을 이루는 모든 점의 각도
        #phi은 길이가 픽셀의 개수 , 각도 저장
        phi_1s = np.degrees(np.arctan2(y1s - ys, x1s - xs) + np.pi)
        phi_2s = np.degrees(np.arctan2(y2s - ys, x2s - xs) + np.pi)
            
        #모든점을 다 보기 때문에 하나만 선택하는 듯
        indices = np.where(phi_2s > phi_1s)[0]
        
        #각도 정보 저장 후 
        for i in indices:
            phi1 = int(phi_1s[i] // BIN_SIZE) % N_ANGLE_BINS
            phi2 = int(phi_2s[i] // BIN_SIZE) % N_ANGLE_BINS
            hist[phi1, phi2] += 1
            #hist는 시계 방향 점의 각도 와 시계 반대방향(시계 반대인지아닌지는 중요x 반대방향 중요 o) 의 빈도 수
            
            
    #전체 등장 수로 나눠 주기 때문에 확률이 됨
    normalised_hist = hist / np.sum(hist)
    
    # 결국 한쪽의 각도일 때 반대쪽 각도의 확률이 나옴
    feature_vector = normalised_hist[np.triu_indices_from(
        normalised_hist, k=1)]

    #12*12 = 144 중 12는 대각 밑에 부분 66개 반환
    return feature_vector


def get_chain_code_features(img):
    
     #윤곽선 찾아줌 (윤곽선 개수 , 픽셀 수 , 2(x,y좌표))
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    dir = [0, 0, 0, 0, 0, 0, 0, 0]

    chain_code = ""

    chain_code_pair = np.zeros((8, 8))

    for i in range(len(contours)):
        chain_code = ""
        for j in range(1, len(contours[i]), 1):
            if (contours[i][j]-contours[i][j-1] == np.array([[-1, -1]])).all():
                dir[0] += 1
                chain_code = chain_code+"0"
            elif (contours[i][j]-contours[i][j-1] == np.array([[0, -1]])).all():
                dir[1] += 1
                chain_code = chain_code+"1"
            elif (contours[i][j]-contours[i][j-1] == np.array([[1, -1]])).all():
                dir[2] += 1
                chain_code = chain_code+"2"
            elif (contours[i][j]-contours[i][j-1] == np.array([[1, 0]])).all():
                dir[3] += 1
                chain_code = chain_code+"3"
            elif (contours[i][j]-contours[i][j-1] == np.array([[1, 1]])).all():
                dir[4] += 1
                chain_code = chain_code+"4"
            elif (contours[i][j]-contours[i][j-1] == np.array([[0, 1]])).all():
                dir[5] += 1
                chain_code = chain_code+"5"
            elif (contours[i][j]-contours[i][j-1] == np.array([[-1, 1]])).all():
                dir[6] += 1
                chain_code = chain_code+"6"
            elif (contours[i][j]-contours[i][j-1] == np.array([[-1, 0]])).all():
                dir[7] += 1
                chain_code = chain_code+"7"

        for k in range(1, len(chain_code), 1):
            chain_code_pair[int(chain_code[k-1])][int(chain_code[k])] += 1

    # normalization
    rangeo = np.max(chain_code_pair)-np.min(chain_code_pair)
    chain_code_pair -= np.min(chain_code_pair)
    chain_code_pair = chain_code_pair/rangeo

    dir = np.array(dir).reshape(1, 8)
    rangeo = np.max(dir)-np.min(dir)
    dir -= np.min(dir)
    dir = dir/rangeo

    feature = np.concatenate(
        (chain_code_pair.flatten().reshape([1, 64]), dir), axis=1)

    return feature















#preprocessing과정
#원본 이미지 -> 흑백변환 -> 이진화 -> hinge , chain vector추출
def preprocessing(img,color=False):
    
    if color:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 199, 5)
     
    feature = np.concatenate([get_chain_code_features(img)[0],get_hinge_features(img)], axis=0)
    
    return feature

