import math
from matplotlib import pyplot as plt
import random
import numpy as np
from random import randint
from numpy import asmatrix
import torch


def getRandomPointInCircle(num, radius, centerx, centery): #在圆内取随机点
    samplePoint = []
    for i in range(num):
        while True:
            x = random.uniform(-radius, radius)
            y = random.uniform(-radius, radius)
            if (x ** 2) + (y ** 2) <= (radius ** 2):
                samplePoint.append((round(x,3) + centerx, round(y,3) + centery, 0))
                break

        plt.plot(x + centerx, y + centery, '*', color="blue")

    return samplePoint

def distance3(dot1, dot2): #三维坐标距离
    return math.sqrt((dot1[0] - dot2[0])**2 + (dot1[1] - dot2[1])**2 +(dot1[2] - dot2[2])**2)

def distance2(dot1, dot2): #二维坐标距离
    return math.sqrt((dot1[0] - dot2[0])**2 + (dot1[1] - dot2[1])**2)

def generate(num):
    K = randint(0, 2) * 2 + 8 #随机用户数8 10 12
    N_T = 16
    N = 10
    bsLocation = [0, 0, 0]
    uaeLocation = [random.uniform(0, 200), random.uniform(0,20), 200]

    alpha = 2.8 #衰减指数参数
    rol = 20 #衰减倍数

    userLocation = getRandomPointInCircle(K, 15, 200, 30)

    data = []
    for i in range(0, num):

        h = np.zeros((N, 1))
        #g = np.zeros((N_T, 1))
        for user in userLocation:
            # average_power_loss = 1e-3  # Average power loss (10^(-3))
            # num_samples = N_T  # Number of fading samples to generate
            # sigma = np.sqrt(average_power_loss / 2)
            # rayleigh_samples = sigma * np.random.randn(num_samples) + 1j * sigma * np.random.randn(num_samples)
            # g_k = rayleigh_samples * math.sqrt(rol * (distance3(user, bsLocation)**(-alpha)))
            # g_k = g_k.reshape(N_T, 1)
            # g = np.concatenate((g, g_k),1)


            disUK = distance3(uaeLocation, user)
            sin_fi_k = distance2(uaeLocation, user) / disUK
            a_k = np.array([])
            for count in range(0, N):
                a_k = np.append(a_k, math.cos(- 1 * math.pi * sin_fi_k * count) + 1j * math.sin(- 1 * math.pi * sin_fi_k * count))
            h_k = a_k * math.sqrt(rol * (disUK**(-alpha))) #1 * N
            h_k = h_k.reshape(N, 1)
            h = np.concatenate((h, h_k), 1)

        h = np.delete(h, 0, 1)
        #g = np.delete(g, 0, 1)

        distanceBU = distance3(bsLocation, uaeLocation)
        sin_fi_R = distance2(bsLocation, uaeLocation) / distanceBU
        sin_fi_B = uaeLocation[2] / distanceBU
        a_B = np.array([])
        a_R = np.array([])
        for count in range(0, N):
            a_R = np.append(a_R, math.cos(- 1 * math.pi * sin_fi_R * count) + 1j * math.sin(- 1 * math.pi * sin_fi_R * count))
        for count in range(0, N_T):
            a_B = np.append(a_B, math.cos(- 1 * math.pi * sin_fi_B * count) + 1j * math.sin(- 1 * math.pi * sin_fi_B * count))

        H = asmatrix(a_R.reshape(N, 1)) * asmatrix(a_B.reshape(N_T, 1)).T.conjugate()
        H = H * math.sqrt(rol * (distanceBU**(-alpha)))
        print(H.shape)
        H = np.concat((H,h), axis = 1) # N * K ++ N * N_T
        data.append(H)
    path = "data/demo_K" + str(K) + "_" + str(num) +".npy"
    #np.save(path, data)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    generate(100)
    datas = np.load("data/demo_K10_10.npy")
    tensor_data = torch.from_numpy(datas)
    print(datas.shape)
