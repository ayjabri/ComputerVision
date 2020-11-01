#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:25:22 2020

@author: aymanjabri
"""


import json
from json import JSONEncoder
import numpy


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def write_json(X):

    with open("test.json", "w") as write_file:
        json.dump(X, write_file, cls=NumpyArrayEncoder)