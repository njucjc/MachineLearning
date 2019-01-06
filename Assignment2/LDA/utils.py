#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Jin Chen

def load_data(path):
    with open(path, 'r') as fp:
        content = [line.strip() for line in fp.readlines()]
    return content

def save_data(content, path, mode='w'):
    with open(path, mode) as fp:
        for line in content:
            fp.write(line+'\n')