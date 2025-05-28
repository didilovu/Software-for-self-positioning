import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import cv2
import math
import heapq
from ultralytics import YOLO
from PIL import Image, ImageDraw

def wheremypoint(StX,StY,EnX,EnY):
    X = int((EnX-StX)/2 + StX)
    Y = int((EnY-StY)/2 + StY)
    return X,Y
    
def far(h_cam):
    f=4.15
    h_real=5
    D=(f*h_real)/h_cam
    return D
    
def point_proc(file_path):
    arr_point = []
    
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            nums = line.split()
            if len(nums) < 3:
                print(f"Некорректный формат строки {line_number}. Ожидается минимум 3 значения.")
                continue
            try:
                x = float(nums[1].replace(',', '.'))
                y = float(nums[2].replace(',', '.'))
                arr_point.append([x, y])
            except ValueError:
                print(f"Некорректные данные в строке {line_number}.")
    return arr_point

def map_to_mat(arr_point, cam_point, size): #представление карты в матричном виде с препятствиями
    if (cam_point[0]<0):
        cam_point[0] = cam_point[0] - cam_point[0]
        for i in range(len(arr_point)):
            arr_point[i][0] = arr_point[i][0] - cam_point[0]
    if (cam_point[1]<0):
        cam_point[1] = cam_point[1] - cam_point[1]
        for i in range(len(arr_point)):
            arr_point[i][1] = arr_point[i][1] - cam_point[1]
    arr_point = [[int(x) for x in y] for y in arr_point]
    mat = np.zeros((size, size))
    for point in arr_point:
        mat[point[0],point[1]] = 1
    #mat[cam_point[0], cam_point[1]] = 2
    df = pd.DataFrame(mat)
    return df


def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

def path(df, start, finish):
    start = (int(round(start[0])), int(round(start[1])))
    finish = (int(round(finish[0])), int(round(finish[1])))
    if (start[0]<0):
        start[0] = start[0] - start[0]
    if (start[1]<0):
        start[1] = start[1] - start[1]
    rows = len(df)
    if rows == 0:
        return []
    cols = len(df[0])
    
    if (start[0] < 0 or start[0] >= rows or start[1] < 0 or start[1] >= cols or
        finish[0] < 0 or finish[0] >= rows or finish[1] < 0 or finish[1] >= cols):
        return []
    if df[start[0]][start[1]] == 1 or df[finish[0]][finish[1]] == 1:
        return []

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    heap = []
    heapq.heappush(heap, (0 + heuristic(start, finish), 0, start[0], start[1], [start]))
    g_scores = { (start[0], start[1]): 0 }
    
    while heap:
        _, g, x, y, current_path = heapq.heappop(heap)
        
        if (x, y) == finish:
            return current_path
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and df[nx][ny] == 0:
                new_g = g + 1
                if (nx, ny) not in g_scores or new_g < g_scores[(nx, ny)]:
                    g_scores[(nx, ny)] = new_g
                    new_f = new_g + heuristic((nx, ny), finish)
                    heapq.heappush(heap, (new_f, new_g, nx, ny, current_path + [(nx, ny)])) 
    return []


def camera_position(obj0, obj1, d0, d1):
    A = np.array(obj0)
    B = np.array(obj1)
    
    AB = B - A
    AB_length = np.linalg.norm(AB)
    
    if AB_length > d0 + d1 or AB_length < abs(d0 - d1):
        print("Невозможно определить положение носителя")
    else:
        x = (d0**2 - d1**2 + AB_length**2) / (2 * AB_length)
        y = np.sqrt(d0**2 - x**2)
        camera_pos = A + (x * AB / AB_length) + (y * np.array([-AB[1], AB[0]]) / AB_length)
    return camera_pos.flatten().tolist()

def metr():
    model = YOLO('best.pt')
    results = model.val(data="C:/Users/79132/last_dataset/dataset.yaml", split="test")
    print(f"Precision: {results.results_dict['metrics/precision']}")
    print(f"Recall: {results.results_dict['metrics/recall']}")
    print(f"mAP50-95: {results.results_dict['metrics/mAP50-95']}")