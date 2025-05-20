# -*- coding: utf-8 -*-
"""

This module contains definitions of classes allowing for quick conversions
of coordinates between coordinate systems.
And angles.
Don't expect it to be working too well, or you might get dissapointed.

As much dissapointed as I, 
when I saw this -------╮
                       v
Created on Sun Jan 26 01:55:53 2025

What am I doing with my life...

@author: pnapi

"""

from . import const
from . import funcs as fn
from . import geod
import numpy as np

# real stuff below


class Angle:
    """
    Args:
        value (float or tuple):
            If giving the value in radians, just enter the value. \n
            If giving the value in degrees:
                insert as a tuple of format:
                    (degrees, minutes, seconds) 
                    where mins & secs are optional.
                    e.g. angle of value 12°34'56.789" is defined like this::
                        geod.Angle((12, 34, 56.789)) 
                        # -> 'deg' as ang_type is not necessary, 
                        # it will be taken care of automatically.
                However you can also input degrees as a single value,
                since 12°34'56.789" ≈ 12.582441°
                you can simply input the value as a float, not tuple 
                (although it will still work just fine (at least it should))
            If giving the value in grads, just enter the value.
                e.g. 12g34c56cc is defined::
                    geod.Angle(12.3456, 'grad')
                    # do not input as a tuple, analogy to 'deg'
        ang_type (str): 
            Specify how the angle is given. Possible options are:
                'rad' when given in radians. This is the default input.
                'deg' when given in degrees.
                'grad' when given in grads.
                
    #--------------------------#     
            
    This class allows for quick angle conversions between systems ->
    -> radians, degrees (and mins secs) and grads.
    """
    def __init__(self, value, ang_type='rad'):
        """

    Args:
        value (float or tuple):
            If giving the value in radians, just enter the value.
            If giving the value in degrees:
                insert as a tuple of format:
                    (degrees, minutes, seconds) 
                    where mins & secs are optional.
                    e.g. angle of value 12°34'56.789" is defined like this::
                        geod.Angle((12, 34, 56.789)) 
                        # -> 'deg' as ang_type is not necessary, 
                        # it will be taken care of automatically.
                However you can also input degrees as a single value,
                since 12°34'56.789" ≈ 12.582441°
                you can simply input the value as a float, not tuple,
                just specify the ang_type then or it will be treated as rad.
            If giving the value in grads, just enter the value.
                e.g. 12g34c56cc is defined::
                    geod.Angle(12.3456, 'grad')
        ang_type (str): 
            Specify how the angle is given. Possible options are:
                'rad' when given in radians. This is the default input.
                'deg' or 'dms' when given in degrees.
                'grad' when given in grads.

        """
        # if giving a tuple assume automatically it's dms
        if type(value) is tuple or type(value) is list:
            ang_type = 'dms'
        # error filter
        if ang_type in ['rad', 'grad'] and type(value) == tuple:
            raise TypeError(f'Angle values in {
                            ang_type} should be given as float, not tuple')
            
        elif ang_type == 'rad':
            self.rad = value
        
        elif ang_type == 'grad':
            self.rad = value/200 * np.pi
            
        elif ang_type == 'dms' or ang_type == 'deg':
            dms = [0, 0, 0]
            if type(value) == tuple or type(value) == list:
                for i in range(len(value)):
                    try:
                        dms[i] = value[i]
                    except IndexError:
                        dms[i] = 0
                d, m, s = dms
            if type(value) not in (tuple, list):
                d = value
            # print(m)
            self.rad = (d + m/60 + s/3600) * np.pi/180
        else:
            raise TypeError(f"ang_type should be 'rad', 'grad' or 'dms'/'deg', not '{ang_type}'")
            
        # default print as dms
        self.grad_called = False
        
        # str params
        self.rounding = 5
        self.negatives_ok = False
        self.pad = False
            
            
    def format_deg(self, rounding=5, negatives_ok=False, pad=False):
        """
        Returns a formatted string and creates an attribute str_deg.

        Args:
            rounding (int): 
                Amount of decimal spaces behind the comma. 
                Defaults to 5.
            negatives_ok (bool): 
                Are negative values ok or should they all be converted to range
                between 0 and 2π. 
                Defaults to False.
            pad (bool or '+'): 
                Add blank space padding, if '+' then place a + sign instead. 
                Defaults to False.

        Returns:
            String that looks somewhat like this:
                ±10° 02' 34.56789"

        """
        self.rounding = rounding
        self.negatives_ok = negatives_ok
        self.pad = pad
        
        rad = self.rad
        
        if negatives_ok == False:
            if rad >= 0:
                pass
            else:
                rad += 2*np.pi
        else:
            pass
        
        try:
            sign = rad/abs(rad)
        except ZeroDivisionError:
            sign = 1
        deg = rad*180/np.pi
        int_deg = int(deg)
        int_min = int((deg - int_deg)*60)
        sec = (((deg - int_deg)*60)-int_min)*60
        sgn = ''
        if sign == -1:
            sgn = '-'
        d, m ,s = abs(int_deg), abs(int_min), round(abs(sec), rounding)        
        
        if sgn != '-':
            if pad == True:
                sgn = ' '
            elif pad == '+':
                sgn = '+'
                
        if round(s, rounding) == 60:
            s = 0 
            m += 1 
            
        if round(m, rounding) == 60:
            m = 0 
            d += 1
        self.str_deg = f'{sgn}{d:02}\u00b0 {m:02}\' {(round(s, rounding)):0{rounding+3}.{rounding}f}\"'
        return self.str_deg
    
    
    def format_grad(self, rounding=5, negatives_ok=False, pad=False):
        """
        Returns a formatted string and creates an attribute str_grad.

        Args:
            rounding (int): 
                Amount of decimal spaces behind the comma. 
                Defaults to 5.
            negatives_ok (bool): 
                Are negative values ok or should they all be converted to range
                between 0 and 2π. 
                Defaults to False.
            pad (bool or '+'): 
                Add blank space padding, if '+' then place a + sign instead. 
                Defaults to False.

        Returns:
            String that looks somewhat like this:
                ±10g 02c 34.56789cc

        """
        
        self.rounding = rounding
        self.negatives_ok = negatives_ok
        self.pad = pad
        
        self.grad_called = True
        
        radi = self.rad
        
        if negatives_ok == False:
            if radi >= 0:
                pass
            else:
                radi += 2*np.pi
        else:
            pass
        
        sgn = str(round(radi/abs(radi)))[0]
        
        g = abs(radi)/np.pi * 200 
        c = g - np.floor(g)
        g = round(g-c)
        c *= 100
        cc = c - np.floor(c)
        c = round(c-cc)
        cc *= 100
        cc = round(cc, rounding)
        
        if sgn != '-':
            if pad == True:
                sgn = ' '
            elif pad == '+':
                sgn = '+'
            else:
                sgn = ''
                
        self.str_grad = f'{sgn}{g:02}g {c}c {cc}cc'
        return self.str_grad
    
    def dec(self):
        return f'{round(self.rad*180/np.pi, 5)}'
    
    def __repr__(self):
        """
        
        Returns:
            String containing info about the value of the angle.
            By defalut, displayed in deg format like this:
                ±10° 02' 34.56789"
            unless format has been modified using format_deg or format_grad.
            If format_grad was called, __str__ will return a value in grads.

        """
        if self.grad_called:
            return self.format_grad(self.rounding, self.negatives_ok, self.pad)
        else:
            return self.format_deg(self.rounding, self.negatives_ok, self.pad)

    def __str__(self):
        """
        
        Returns:
            String containing info about the value of the angle.
            By defalut, displayed in deg format like this:
                ±10° 02' 34.56789"
            unless format has been modified using format_deg or format_grad.
            If format_grad was called, __str__ will return a value in grads.

        """
        if self.grad_called:
            return self.format_grad(self.rounding, self.negatives_ok, self.pad)
        else:
            return self.format_deg(self.rounding, self.negatives_ok, self.pad)
        
    def __sub__(self, other):
        return Angle(self.rad - other.rad)
    
    def __add__(self, other):
        return Angle(self.rad + other.rad)
    
    def __mul__(self, other):
        return Angle(self.rad * other)

    def __rmul__(self, other):
        return Angle(self.rad * other)
    
    def __truediv__(self, other):
        return Angle(self.rad / other)
        
    def __eq__(self, other):
        return str(self) == str(other) 
        # Stringified - why? -> avoids FPU ambiguity,
        # checks the proximity of angles to given (def. 5) number 
        #       of decimal places specified by the .format attribute 
    
    def __neg__(self):
        return Angle(-self.rad)
    
    def __gt__(self, other):
        return self.rad > other.rad
    
    def __lt__(self, other):
        return self.rad < other.rad
    
    def __pow__(self, power):
        'Why the @@@@ would you need this but here you go'
        return Angle(self.rad**power)
    
    def sin(self, times=1):
        return np.sin(self.rad * times)
    
    def cos(self, times=1):
        return np.cos(self.rad * times)
    
    def tan(self, times=1):
        return np.tan(self.rad * times)
    
    def cot(self, times=1):
        return 1 / np.tan(self.rad * times)
    
    
    
class Coordinates3d:
    """
    Arguments for class initialization: 
        crd_type (str): 
            Type of coordinates given on entry.
            Acceptable values are:
                'xyz' for when XYZ coordinates are given,
                'flh' for when φλh coordinates are given.
        c1 (float): 
            First element of cooridantes set. Either X or φ.
        c2 (float): 
            Second element of cooridantes set. Either Y or λ.
        c3 (float): 
            Third element of cooridantes set. Either Z or h.
        a (float):
            Major semi-axis of the ellipsoid. Defaults to GRS80 value.
            This value might not be needed when only operating on XYZ,
            though it's still safer to have it defined
        e2 (float):
            Eccentricity of the ellipsoid. Defaults to GRS80 value.
            Same as with a, the value of e2 might be unnecessary
            but it's better to have it.
    
    #--------------------------#
    
    This class creates a set of 3D coordinates of a point in a given space.
    Input can be given as ortho XYZ or geodetic φλh,
    the coordinates are automatically converted to XYZ by default,
    to allow easier calculations.
    """

    def __init__(self, crd_type, c1, c2, c3, a=geod.a, e2=geod.e2):
        """

        Args:
            crd_type (str): 
                Type of coordinates given on entry.
                Acceptable values are:
                    'xyz' for when XYZ coordinates are given,
                    'flh' for when φλh coordinates are given.
            c1 (float or tuple (for deg-min-sec vals)): 
                First element of cooridantes set. Either X or φ.
            c2 (float or tuple (for deg-min-sec vals)): 
                Second element of cooridantes set. Either Y or λ.
            c3 (float): 
                Third element of cooridantes set. Either Z or h.
            a (float):
                Major semi-axis of the ellipsoid. Defaults to GRS80 value.
                This value might not be needed when only operating on XYZ,
                though it's still safer to have it defined
            e2 (float):
                Eccentricity of the ellipsoid. Defaults to GRS80 value.
                Same as with a, the value of e2 might be unnecessary
                but it's better to have it.

        """
        # check if the given type is acceptable
        acc = ['xyz', 'flh', 
               '00', '2000', 'pl00', 'pl2000', 'PL2000',
               '92', '1992', 'pl92', 'pl1992', 'PL1992']
        if crd_type not in acc:
            raise geod.CoordinatesTypeError(
                    'Input a correct coordinate type.')
            
        # get the right type
        if type(c1) == Angle:
            c1 = c1.rad
        elif type(c1) in (tuple, list):
            c1 = Angle(c1, 'dms').rad
        if type(c2) == Angle:
            c2 = c2.rad
        elif type(c2) in (tuple, list):
            c2 = Angle(c2, 'dms').rad
            
        
        # assign ellipsoid params
        self.a, self.e2 = a, e2

        # if XYZ given, do nothing
        match crd_type:
            case 'xyz':
                self.x, self.y, self.z = c1, c2, c3
    
            # if φλh given, convert it to XYZ
            case 'flh':
                c1 = Angle(c1).rad
                c2 = Angle(c2).rad
                self.x, self.y, self.z = fn.flh2XYZ(c1, c2, c3)
                
            case 'PL2000' | 'pl2000' | '2000' | '00' | 'pl00' | 'PL00':
                c1, c2 = fn.PL2000_2fl(c1, c2)
                self.x, self.y, self.z = fn.flh2XYZ(c1, c2, c3)
                
            case 'PL1992' | 'pl1992' | '1992' | '92' | 'pl92' | 'PL92':
                c1, c2 = fn.PL1992_2fl(c1, c2)
                self.x, self.y, self.z = fn.flh2XYZ(c1, c2, c3)

    # list of options of output coords system types for conversions
    possible_conversions = {
        'xyz': 'Returns XYZ coords',
        'flh': 'Returns φλh coords',
        'PL2000': 'Returns coords in flat PL-2000 system',
        'PL1992': 'Returns coords in flat PL-1992 system',
        'gk': 'Returns coords in GaussKrüger system'
        }
    
    def convert(self, output_type, return_as_str=False, la_0=None):
        """
        Converts given coordinates to o different system (either 3D or 2D)
        based on the users desire.
        
        #--------------------------#
        
        Args:
            output_type (str): 
                Specify the type of coordinate system the output should be in.
                Check {coordinates3d.possible_conversions} for help on options.
            return_as_str (bool):
                Specify whether to return coords as a list or string.
                Defaults to False -> returns as list.
            la_0 (float):
                Value of the 0 longitude, in radians.
                Used only for Gauss-Krüger transformations,
                in PL-2000 and PL-1992 the value is calculated automatically 
                based on the value of the longitude.
                Defaults to None, since it's only used for this single purpose.
        
        #--------------------------#
        
        Returns:
            3- or 2-element list of coordinates in the given coordinate system,
            or a string containing info about them. 

        """
        # raise an error if output_type is invalid
        if output_type not in self.possible_conversions:
            raise TypeError('Invalid output_type.')

        # get φλh for conversions
        phi, la, height = fn.XYZ2flh(
            self.x, self.y, self.z, self.a, self.e2)
        self.phi = Angle(phi)
        self.la = Angle(la)
        self.height = height

        # convert to {output_type}
        if output_type == 'xyz':
            if return_as_str:
                return (
                    f"""
    X: {round(self.x, 3):,}m
    Y: {round(self.y, 3):,}m
    Z: {round(self.z, 3):,}m
                    """)
            else:
                return [self.x, self.y, self.z]

        if output_type == 'flh':
            if return_as_str:
                return (
                    f"""
    φ: {self.phi.format_deg(negatives_ok=True)}
    λ: {self.la.format_deg(negatives_ok=True)}
    h: {round(self.height, 3):,}m
                    """)
            else:
                return [self.phi, self.la, self.height]

        if output_type == 'PL2000':
            x00, y00 = fn.fl2_PL2000(self.phi, self.la)
            if return_as_str:
                return (
                    f"""
    X: {round(x00, 3):,}m
    Y: {round(y00, 3):,}m
                    """)
            else:
                return [x00, y00]

        if output_type == 'PL1992':
            x92, y92 = fn.fl2_PL1992(self.phi, self.la)
            if return_as_str:
                return (
                    f"""
    X: {round(x92, 3):,}m
    Y: {round(y92, 3):,}m
                       """)
            else:
                return [x92, y92]

        if output_type == 'gk':
            if la_0 is None:
                raise geod.MissingArgError(
                    'For GK transformations, λ must be given.')
            xgk, ygk = fn.fl2XY_GaussKruger(self.phi.rad, self.la.rad, la_0)
            if return_as_str:
                return (
                    f"""
    X: {round(xgk, 3):,}m
    Y: {round(ygk, 3):,}m
                    """)
            else:
                return [xgk, ygk]
            
    def __repr__(self):
        return self.convert('flh', True)
    
    def __sub__(self, other):
        return self.x-other.x, self.y-other.y, self.z-other.z
    

class Coordinates2d:
    """
    Arguments for class initialization: 
        crd_type (str): 
            Type of coordinates given on entry.
            Acceptable values are:
                'gk' for when Gauss-Krüger XY coordinates 
                and λ0 longitude are given, 
                entering the value of 0-longitude is mandatory in 'gk' case.
                'fl' for when φλ coordinates are given.
                'PL2000' for when XY coordinates in PL2000 system are given
                'PL1992' for when XY coordinates in PL1992 system are given
        c1 (float): 
            First element of cooridantes set. 
            Either X (in whichever system) or φ.
        c2 (float): 
            Second element of cooridantes set. 
            Either Y (in whichever system) or λ.
        a (float):
            Major semi-axis of the ellipsoid. Defaults to GRS80 value.
            This value as opposed to 3D coords is necessary. 
            This is to avoid ambiguity between different ellipsoids,
            where φλ refer to points on their surfaces, 
            while each ellipsoid is defined ever so slightly differently.
        e2 (float):
            Eccentricity of the ellipsoid. Defaults to GRS80 value.
            Same as with a, the value of e2 is necessary here.
        la_0 (float):
            Value of 0-longitude in radians for Gauss-Krüger XY cooridinates
            
    #--------------------------#
    
    This class creates a set of 2D coordinates of a point in a given space.
    Input can be given as flat XY in any of available systems or geodetic φλ.
    The coordinates are automatically converted to φλ by default,
    to allow easier calculations.
    """

    acceptable_input_types = [
        'gk',
        'fl',
        'PL2000',
        'PL1992']

    possible_conversions = {
        'fl': 'Returns φλ coords',
        'PL2000': 'Returns coords in PL-2000 system',
        'PL1992': 'Returns coords in PL-1992 system',
        'gk': 'Returns coords in GaussKrüger system'}

    def __init__(self, crd_type, c1, c2, a=geod.a, e2=geod.e2, la_0=None):
        """

        Args:
            crd_type (str): 
                Type of coordinates given on entry.
                Acceptable values are:
                    'gk' for when Gauss-Krüger XY coordinates 
                    and λ0 longitude are given, entering the value of 0-longitude
                    is mandatory in 'gk' case.
                    'fl' for when φλ coordinates are given.
                    'PL2000' for when XY coordinates in PL2000 system are given
                    'PL1992' for when XY coordinates in PL1992 system are given
            c1 (float): 
                First element of cooridantes set. 
                Either X (in whichever system) or φ.
            c2 (float): 
                Second element of cooridantes set. 
                Either Y (in whichever system) or λ.
            a (float):
                Major semi-axis of the ellipsoid. Defaults to GRS80 value.
                This value as opposed to 3D coords is necessary. 
                This is to avoid ambiguity between different ellipsoids,
                where φλ refer to points on their surfaces, 
                while each ellipsoid is defined ever so slightly differently.
            e2 (float):
                Eccentricity of the ellipsoid. Defaults to GRS80 value.
                Same as with a, the value of e2 is necessary here.
            la_0 (float):
                Value of 0-longitude in radians for Gauss-Krüger XY cooridinates

        """
        # assign ellipsoid params
        self.a, self.e2 = a, e2
        
        # get the right type
        if type(c1) == Angle:
            c1 = c1.rad
        if type(c2) == Angle:
            c2 = c2.rad

        # check if the given type is acceptable
        if crd_type not in self.acceptable_input_types:
            raise TypeError('Input a correct coordinate type.')

        # assert the input of la_0 for GK coords
        if crd_type == 'gk' and la_0 is None:
            raise geod.MissingArgError(
                'Value of 0-longitude is mandatory for GK coordinates')

        # if φλ given, do nothing
        if crd_type == 'fl':
            self.phi, self.la = Angle(c1), Angle(c2)

        # if xy any other given, convert it to φλ
        if crd_type == 'gk':
            self.phi, self.la = fn.XY_GaussKruger_2fl(c1, c2, la_0)
            self.phi = Angle(self.phi)
            self.la = Angle(self.la)

        if crd_type == 'PL2000':
            self.phi, self.la = geod.PL2000_2fl(c1, c2)
            self.phi = Angle(self.phi)
            self.la = Angle(self.la)

        if crd_type == 'PL1992':
            self.phi, self.la = geod.PL1992_2fl(c1, c2)
            self.phi = Angle(self.phi)
            self.la = Angle(self.la)

    def convert(self, output_type, return_as_str=False, la_0=None):
        """
        Converts given coordinates to o different system (either 3D or 2D)
        based on the users desire.
        
        #--------------------------#
        
        Args:
            output_type (str): 
                Specify the type of coordinate system the output should be in.
                Check {coordinates2d.possible_conversions} for help on options.
            return_as_str (bool):
                Specify whether to return coords as a list or string.
                Defaults to False -> returns as list.
            la_0 (float):
                Value of the 0 longitude, in radians.
                Used only for Gauss-Krüger transformations,
                in PL-2000 and PL-1992 the value is calculated automatically 
                based on the value of the longitude.
                Defaults to None, since it's only used for this single purpose.
        
        #--------------------------#
        
        Returns:
            2-element list of coordinates in the given coordinate system,
            or a string containing info about them. 

        """
        # raise an error if output_type is invalid
        if output_type not in self.possible_conversions:
            raise TypeError('Invalid output_type.')

        # convert to {output_type}
        if output_type == 'fl':
            if return_as_str:
                return (
                    f"""
    φ: {self.phi.format_deg(negatives_ok=True)}
    λ: {self.la.format_deg(negatives_ok=True)}
                    """)
            else:
                return [self.phi, self.la]

        if output_type == 'gk':
            if la_0 is None:
                raise geod.MissingArgError(
                    'Value of 0-longitude is mandatory for GK conversions')
            else:
                xgk, ygk = fn.fl2XY_GaussKruger(self.phi.rad, self.la.rad, la_0)
            if return_as_str:
                return (
                    f"""
    X: {round(xgk, 3):,}
    Y: {round(ygk, 3):,}
                    """)
            else:
                return [xgk, ygk]

        if output_type == 'PL2000':
            x00, y00 = fn.fl2_PL2000(
                self.phi.rad, self.la.rad, int(fn.lambda_0(self.la.rad)).rad)
            if return_as_str:
                return (
                    f"""
    X: {round(x00, 3):,}
    Y: {round(y00, 3):,}
                    """)
            else:
                return [x00, y00]

        if output_type == 'PL1992':
            x92, y92 = fn.fl2_PL1992(self.phi.rad, self.la.rad)
            if return_as_str:
                return (
                    f"""
    X: {round(x92, 3):,}
    Y: {round(y92, 3):,}
                    """)
            else:
                return [x92, y92]
            
    def __repr__(self):
        output_type = input('Type? -> \n' 
                            + str(self.possible_conversions).replace(',','\n') 
                            + '\n: ')
        return self.convert(output_type, True)
    
    
class Time:
    def __init__(self, value, input_type='s'):
        self.repr_type = 's'
        match input_type:
            case 's':
                self.value = value
            case 'm':
                self.value = value*60
            case 'h':
                self.value = value*3600
            case 'd':
                self.value = value*86400
            case 'w':
                self.value = value*604800
            case _:
                raise TypeError(f'Invalid Time.input_type (\'{input_type}\' is not s/m/h/d/w)')
            
    def format_as(self, repr_type):
        repr_types = ('w', 'd', 'h', 'm', 's',
                        'milli', 'micro', 'nano', 'pico', 'femto')
        assert repr_type in repr_types, f"repr_type must be from {repr_types}, not '{repr_type}'"
        self.repr_type = repr_type
        s = self.value
        
        if s > 4.5e17:
            return 'basically forever'
        else:        
            match repr_type:
                case 'w':
                    w = s - s%604800
                    d = s - w - s%86400
                    h = s - w - d - s%3600
                    m = s - w - d - h - s%60
                    sc= s - w - d - h - m 
                    w /= 604800
                    d /= 86400
                    h /= 3600
                    m /= 60
                    w, d, h, m = int(w), int(d), int(h), int(m)
                    return f'{w}w {d}d {h}:{m:02}:{sc:02.2f}'
                case 'd':
                    d = s - s%86400
                    h = s - d - s%3600
                    m = s - d - h - s%60
                    sc= s - d - h - m 
                    d /= 86400
                    h /= 3600
                    m /= 60
                    d, h, m = int(d), int(h), int(m)
                    return f'{d}d {h}:{m:02}:{sc:02.2f}'
                case 'h':
                    h = s - s%3600
                    m = s - h - s%60
                    sc= s - h - m 
                    h, m = int(h), int(m)
                    return f'{h}:{m:02}:{sc:02.2f}'
                case 'm':
                    m = s - s%60
                    sc = s - m
                    m = int(m)
                    return f'{m:02}:{sc:02.2f}'
                case 's':
                    return f'{s:.2f}'
                case 'milli':
                    return f'{s * 1e3}ms'
                case 'micro':
                    return f'{s * 1e6}\u03bcs'
                case 'nano':
                    return f'{s * 1e9}ns'
                case 'pico':
                    return f'{s * 1e12}ps'
                case 'femto':
                    return f'{s * 1e15}fs'
        
    def __str__(self):
        return self.format_as(self.repr_type)
    
    def __repr__(self):
        return f'{self.value}s'
    
    def __sub__(self, other):
        return Time(self.value - other.value)
    
    def __add__(self, other):
        return Time(self.value + other.value)
    
    def __mul__(self, other):
        return Time(self.value * other)

    def __rmul__(self, other):
        return Time(self.value * other)
    
    def __truediv__(self, other):
        return Time(self.value / other)
        
    def __eq__(self, other):
        return str(self) == str(other) 
    
    def __neg__(self):
        return Time(-self.value)
    
    def __gt__(self, other):
        return self.value > other.value
    
    def __lt__(self, other):
        return self.value < other.value
    
    def __pow__(self, power):
        'Why the @@@@ would you need this but here you go'
        return Time(self.value**power)
    
        
