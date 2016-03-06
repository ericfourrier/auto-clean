#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: efourrier

Purpose : File with all custom exceptions
"""

class NotNumericColumn(Exception):
    """ The column should be numeric  """
    pass

class NumericError(Exception):
    """ The column should not be numeric  """
    pass

# class NotFactor
