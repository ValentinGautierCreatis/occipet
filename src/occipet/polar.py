#!/usr/bin/env python3

import numpy as np


def radial_subsampling(data:np.ndarray, factor:float)->np.ndarray:
    tab = np.zeros(data.shape, dtype=data.dtype)
    circumference = 2 * np.sum(data.shape)
    number_angles = int((circumference - 4)/factor)
    thetas = np.linspace(0, 2*np.pi, number_angles)
    yc, xc = data.shape[0]//2, data.shape[1]//2
    for theta in thetas:
        x, y = compute_stop_point(xc, yc, data.shape[1] - 1, data.shape[0] - 1, theta)
        dx = abs(x - xc)
        dy = abs(y - yc)

        if dx > dy:
            plot_pixel(xc, yc, x, y, dx, dy, 0, data, tab)
        else:
            plot_pixel(yc, xc, y, x, dy, dx, 1, data, tab)
    return tab


def put_data(x, y, data, tab):
    tab[y, x] = data[y, x]


def plot_pixel(x1, y1, x2, y2, dx, dy, decide, data, tab):
    pk = 2 * dy - dx
    put_data(x1, y1, data, tab)
    for _ in range(dx+1):
        x1 = (x1 + 1) if x1<x2 else (x1 - 1)
        if pk < 0:
            if decide == 0:
                put_data(x1, y1, data, tab)
                pk = pk + 2 * dy
            else:
                put_data(y1, x1, data, tab)
                pk = pk + 2 * dy
        else:
            y1 = (y1 + 1) if y1<y2 else (y1 - 1)
            if (decide == 0):
                put_data(x1, y1, data, tab)
            else:
                put_data(y1, x1, data, tab)
            pk = pk + 2 * dy - 2 * dx
    return tab


def compute_stop_point(xc, yc, x_max, y_max, theta):
    if np.pi/4 <= theta <= 3 * (np.pi/4):
        y = y_max
        x = xc + np.tan(np.pi/2 - theta) * (y - yc)

    elif 5 * np.pi/4 <= theta <= 7 * np.pi/4:
        y = 0
        x = xc + np.tan(np.pi/2 - theta) * (y - yc)

    elif 3 * np.pi/4 < theta < 5 * np.pi/4:
        x = 0
        y = yc + np.tan(theta) * (x -xc)

    else:
        x = x_max
        y = yc + np.tan(theta) * (x -xc)

    return int(x), int(y)
