# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:16:59 2025

@author: pnapi
"""

#-------------------------------------------------------------#

#constants:
    
    #GRS80 ellipsoid:
a = 6378137. #bigger semi-axis of grs80 [meters]
e2 = 0.00669438002290 #eccentricity of grs80 squared 

    #Krasowski ellipsoid:
akr = 6378245. #bigger semi-axis of Krasowski ellipsoid [meters]
e2kr = 0.00669342162296 #eccentricity of Krasowski squared 

rho = 206265.0 #rho
ro = rho #just to avoid stupidity

    #Projection scales
m0_1992 = 0.9993 #projection scale PL-1992 <-> GK
m0_2000 = 0.999923 #projection scale PL-2000 <-> GK

#-------------------------------------------------------------#
