#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 09:50:17 2022

@author: mulapket
"""
from mpi4py import MPI

p = MPI.COMM_WORLD.rank

p = MPI.COMM_WORLD.rank
if p!=0 :
    message = MPI.COMM_WORLD.recv(source=p-1)
    print(message)
if p < MPI.COMM_WORLD.size:
    message = p+100
    print(message)