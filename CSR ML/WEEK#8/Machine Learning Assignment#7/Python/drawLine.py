# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 17:41:22 2018

@author: Mohammad Wasil Saleem
"""

#import matplotlib.pyplot as plot

def DrawLine(p1, p2,ax, **kwargs):
    
    ax.plot( [p1[0],p2[0]], [p1[1],p2[1]], **kwargs)
