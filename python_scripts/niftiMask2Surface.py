# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:18:16 2017

@author: lindseykitchell

This is a function that takes in a binary nifti image and outputs an
.stl surface mesh.

inputs:
img_path: path string to nifti image
smooth_iter: number of smoothing iterations, default = 10

Example:

import os

img_path = os.path.join('/Users/lindseykitchell/Box Sync/fiberVolumes/',
                        'HCP_105115_STREAM_Lmax8_conn6_boolVol_R_Arc.nii.gz')

niftiMask2surface(img_path, 15)

"""

import vtk


def niftiMask2Surface(img_path, smooth_iter=10):
    # import the binary nifti image
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(img_path)
    reader.Update()

    # do marching cubes to create a surface
    surface = vtk.vtkDiscreteMarchingCubes()
    surface.SetInputConnection(reader.GetOutputPort())
    # GenerateValues(number of surfaces, label range start, label range end)
    surface.GenerateValues(1, 1, 1)
    surface.Update()

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(surface.GetOutputPort())
    smoother.SetNumberOfIterations(smooth_iter)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputConnection(smoother.GetOutputPort())
    connectivityFilter.SetExtractionModeToLargestRegion()
    connectivityFilter.Update()

    # doesn't work, but may need in future
    # close_holes = vtk.vtkFillHolesFilter()
    # close_holes.SetInputConnection(smoother.GetOutputPort())
    # close_holes.SetHoleSize(10)
    # close_holes.Update()

    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(connectivityFilter.GetOutputPort())
    writer.SetFileTypeToBinary()
    writer.SetFileName("arc_smooth.stl")
    writer.Write()
