# Write files to visualize in Paraview

#import toParaview as toPara
#toPara.StructuredScalar(InputArray, OutFileName, lim1, lim2)



# Unstructured scalar -- Points1D      
#####           Unstructured vector
# Structured scalar  -vti
# Structured vector  - vtk
# TENSOR?

#http://urania.udea.edu.co/sitios/astronomia-2.0/pages/descargas.rs/files/descargasdt5vi/Cursos/CursosElectivos/FisicaAstrofisicaComputacional/2009-2/Documentacion/mayavi/data.html
#http://docs.enthought.com/mayavi/mayavi/data.html

from tvtk.api import tvtk, write_data
import numpy as np
from tvtk.array_handler import get_vtk_array_type
from evtk.hl import pointsToVTK

def StructuredVector(InputArray, OutFileName, lim1, lim2):
    nGr = np.shape(InputArray)[:,:,:,0][0]

    nx1IC = ny1 = nz1 = int(lim1)
    nx2IC = ny2 = nz2 = int(lim2)
    im = jm = km = nGr*1j
    nf0 = nf1 = nf2 = nGr

    Xg, Yg, Zg = np.mgrid[nx1IC:nx2IC:im,ny1:ny2:jm,nz1:nz2:km]
# Make the data.
    dims = np.array((nf0, nf1, nf2))
    vol = np.array((lim1,lim2,lim1,lim2,lim1,lim2))
    origin = vol[::2]
    spacing = (vol[1::2] - origin)/(dims -1)
    xmin, xmax, ymin, ymax, zmin, zmax = vol                
    x, y, z = np.ogrid[xmin:xmax:dims[0]*1j,
                    ymin:ymax:dims[1]*1j,
                    zmin:zmax:dims[2]*1j]
    x, y, z = [t.astype('f') for t in (x, y, z)]
    
    
    i = tvtk.ImageData(origin=origin, spacing=spacing, dimensions=dims)
    i.point_data.scalars = InputArray[:,:,:,0].ravel()
    i.point_data.scalars.name = 'v0'
    i.dimensions = InputArray[:,:,:,0].shape
    # add second point data field
    i.point_data.add_array(InputArray[:,:,:,1].ravel())
    i.point_data.get_array(1).name = 'v1'
    i.point_data.update()
    
    i.point_data.add_array(InputArray[:,:,:,2].ravel())
    i.point_data.get_array(2).name = 'v2'
    i.point_data.update()
    
    fileOut = OutFileName+'.vtk'
    print fileOut
    write_data(i, fileOut)
    
    
def StructuredScalar(InputArray, OutFileName, lim1, lim2):
    

    nGr = np.shape(InputArray)[0]
    
    # Make the data.
    nx1IC = ny1 = nz1 = int(lim1)
    nx2IC = ny2 = nz2 = int(lim2)
    im = jm = km = nGr*1j
    nf0 = nf1 = nf2 = nGr
    
    Xg, Yg, Zg = np.mgrid[nx1IC:nx2IC:im,ny1:ny2:jm,nz1:nz2:km]
    # Make the data.
    dims = np.array((nf0, nf1, nf2))
    
    vol = np.array((lim1,lim2,lim1,lim2,lim1,lim2))
    origin = vol[::2]
    spacing = (vol[1::2] - origin)/(dims -1)
    xmin, xmax, ymin, ymax, zmin, zmax = vol
    x, y, z = np.ogrid[xmin:xmax:dims[0]*1j,
                    ymin:ymax:dims[1]*1j,
                    zmin:zmax:dims[2]*1j]
    x, y, z = [t.astype('f') for t in (x, y, z)]
    scalars = InputArray
    
    # Make the tvtk dataset.
    spoints = tvtk.StructuredPoints(origin=origin, spacing=spacing,
                                    dimensions=dims)
    s = scalars.transpose().copy()
    spoints.point_data.scalars = np.ravel(s)
    spoints.point_data.scalars.name = 'scalars'
    
    spoints.scalar_type = get_vtk_array_type(s.dtype)
    
    fileOut = OutFileName+'.vti'
    print fileOut
    w = tvtk.XMLImageDataWriter(input=spoints, file_name=fileOut)
    w.write()   
        
        
def UnstructuredScalar(InputCoords, OutFileName):
    #mlab.options.backend = 'envisage'
#------------ Writing  .vti files ----------------------
# Make the data.
    dims = np.array((ngr, ngr, ngr))
    vol = np.array((nx1, nx2, ny1, ny2, nz1, nz2))
    origin = vol[::2]
    spacing = (vol[1::2] - origin)/(dims -1)
    xmin, xmax, ymin, ymax, zmin, zmax = vol                
    x, y, z = np.ogrid[xmin:xmax:dims[0]*1j,
	                ymin:ymax:dims[1]*1j,
        	        zmin:zmax:dims[2]*1j]
    x, y, z = [t.astype('f') for t in (x, y, z)]
	
    scalars = r


    # Make the tvtk dataset.
    spoints = tvtk.StructuredPoints(origin=origin, spacing=spacing,
                                    dimensions=dims)
                                    
    s = scalars.transpose().copy()
    spoints.point_data.scalars = np.ravel(s) 
    spoints.point_data.scalars.name = 'scalars'

    fileOut =OutFileName+'.vti'
    w = tvtk.XMLImageDataWriter(input=spoints, file_name=fileOut)
    w.write()
    

def UnstructuredGrid(x, y, z, data, OutFileName): 
   #mlab.options.backend = 'envisage'
#------------ Writing  .vti files ----------------------
# Make the data.
    points = np.array([x,y,z]).T
    scalars = np.array(data) 
# Now create the UG.    
    ug = tvtk.UnstructuredGrid(points=points)
    
    ug.point_data.scalars = scalars
    ug.point_data.scalars.name = 'Data'


    fileOut = OutFileName+'.vtu'
    print fileOut
    w = tvtk.XMLUnstructuredGridWriter(input=ug, file_name=fileOut)
    w.write()

      
    
    
    
def Points1d(x, y, z, data, OutFileName):     # make general *args, kwargs type
 
    #points = np.array([x,y,z]).T

    #data = np.array(data)    
    #
    #mesh = tvtk.PolyData(points=points)
    #mesh.point_data.scalars = data
    #mesh.point_data.scalars.name = 'Data'
    #
    #
    #fileOut = OutFileName+'.vtp'
    #print fileOut
    #w = tvtk.XMLPolyDataWriter(input=mesh, file_name=fileOut)
    #w.write()
    #
 
    #pressure = np.random.rand(npoints)  
    #temp = np.random.rand(npoints) 
    fileOut =  OutFileName
    print OutFileName+'.vtu'
    #pointsToVTK(fileOut, x, y, z, data = {"r" : data, "pressure" : pressure})
    pointsToVTK(fileOut, x , y , z , data = {"data" : data})