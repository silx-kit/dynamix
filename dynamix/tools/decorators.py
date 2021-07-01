"""Bunch of useful decorators"""

__authors__ = ["Jerome Kieffer", "H. Payno", "P. Knobel", "V. Valls"]
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "01/07/2021"
__status__ = "development"
__docformat__ = 'restructuredtext'

import time
import logging

timelog = logging.getLogger("timeit")

def timeit(func):

    def wrapper(*arg, **kw):
        '''This is the docstring of timeit:
        a decorator that logs the execution time'''
        t1 = time.perf_counter()
        res = func(*arg, **kw)
        t2 = time.perf_counter()
        name = func.func_name
        timelog.warning("%s took %.3fs", name, t2 - t1)
        return res

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper
