# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 20:09:22 2018

@author: Mohammad Wasil Saleem
"""

import numpy as np
import math
import matplotlib.pyplot as plot
from matplotlib import colors

def DisplayData(sel):
    [m, n] = sel.shape
    
    example_width = int(np.round(np.sqrt(n)))
    example_height = int(n / example_width)
    
    # Compute number of items to display
    display_rows = math.floor(np.sqrt(m))
    display_cols = math.ceil(m / display_rows)

    # Between images padding
    #pad = 1

    display_array = np.ones((  display_rows * (example_height ),   display_cols * (example_width ) ))
    #current_image = 1
    for i in range(display_rows):
        for j in range(display_cols):            
            current_image = i * display_cols + j

            image_part = sel[current_image, :].reshape((example_height, example_width))
            display_array[
                (j * example_height):((j + 1) * example_height),
                (i * example_width):((i + 1) * example_width)
            ] = image_part
    
    fig, ax = plot.subplots()
    ax.imshow(display_array.T)
    plot.show()

def hsv(n=63):
       return colors.hsv_to_rgb( np.column_stack([ np.linspace(0, 1, n+1), np.ones( ((n+1), 2) )]) )
