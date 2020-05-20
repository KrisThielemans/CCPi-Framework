import tomophantom
from tomophantom import TomoP3D
from ccpi.optimisation.algorithms import CGLS, RCGLS
from ccpi.framework import BlockDataContainer, ImageGeometry, AcquisitionGeometry
from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.astra.operators import AstraProjector3DSimple
from ccpi.utilities.display import plotter2D
import os
import numpy as np

# Set up image geometry
num_vx = 128
num_vy = 128
num_vz = 128

ig = ImageGeometry(voxel_num_x = num_vx, voxel_num_y = num_vy, voxel_num_z = num_vz)

print(ig)

# Load Shepp-Logan phantom 
model = 13
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "Phantom3DLibrary.dat")

#tomophantom takes angular input in degrees
phantom_3D = TomoP3D.Model(model, (num_vx, num_vy, num_vz), path_library3D)
#set up acquisition geometry
number_pixels_x = 128
number_projections = 180
angles = np.linspace(0, np.pi, number_projections, dtype=np.float32)
ag = AcquisitionGeometry(geom_type='parallel', angles=angles, 
                         pixel_num_h=number_pixels_x, 
                         pixel_num_v=num_vz, 
                         dimension_labels=['vertical', 'angle', 'horizontal'])
print(ag)


phantom_sino = TomoP3D.ModelSino(model, num_vx, number_pixels_x, num_vz, angles*180./np.pi, path_library3D)
# Rescale the tomophantom data, set the max absorbtion to 25%
set_ratio_absorption = 0.25
new_max_value = -np.log(set_ratio_absorption)
sino_max = np.amax(phantom_sino)
scale = new_max_value/sino_max

# Allocate the image data container and copy the dataset in.
# This is only used as a reference to the ground truth
model = ig.allocate(0)
model.fill(phantom_3D*scale)

# Allocate the acquisition data container and copy the sinogram in
sinogram = ag.allocate(0)
sinogram.fill(phantom_sino*scale)

#plotter2D([model.subset(vertical=64), sinogram.subset(vertical=64)])
background_counts = 1000 #lower counts will increase the noise

counts = float(background_counts) * np.exp(-sinogram.as_array())
noisy_counts = np.random.poisson(counts)
sino_out = -np.log(noisy_counts/background_counts)

sinogram_noisy = ag.allocate()
sinogram_noisy.fill(sino_out)


device = "gpu"
operator = AstraProjector3DSimple(ig, ag)



print ("calculating norm of A")
normA = operator.norm()
print ("calculating norm of Gradient")
L = Gradient(ig)
normL = L.norm()
ratio = normA/normL
gamma = 0.3
alpha = gamma * ratio

print (alpha)

# gamma selects the weighing between the regularisation and the fitting term:
# 1 means equal weight
# 

cgls_simple = CGLS(operator=operator, data=sinogram_noisy, max_iteration = 1000, 
                   update_objective_interval=20)
cgls_simple.run(40)

operator_block = BlockOperator( operator, alpha * L)
zero_data = L.range_geometry().allocate(0)
data_block = BlockDataContainer(sinogram_noisy, zero_data)
  
cgls_regularised = CGLS(operator=operator_block, data=data_block, 
                        update_objective_interval = 10)
cgls_regularised.max_iteration = 10000
cgls_regularised.run(10)

rcgls = RCGLS(operator=operator, data=sinogram_noisy, alpha=alpha, 
              max_iteration = 1000, update_objective_interval=20, tolerance=1e-2)
rcgls.run(10)



plotter2D([model.subset(vertical=64), cgls_simple.get_output().subset(vertical=64),
           rcgls.get_output().subset(vertical=64),
           cgls_regularised.get_output().subset(vertical=64)])