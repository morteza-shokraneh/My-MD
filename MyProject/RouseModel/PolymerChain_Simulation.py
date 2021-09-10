#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 06:23:46 2021

@author: morteza
"""

import sys
import hoomd
import hoomd.md
import numpy as np
import math
import os

#initialize the system
hoomd.context.initialize("")

#parameters
Monomers = 50
Bonds = Monomers - 1
k = 1 #spring constant
passive_particles = 0
active_particles = 0
gamma = 1 #drag coefficinet
r0 = 0 #rest length
kT = 1 #temperature in units of kT
dt = 0.01 #step time
integration_steps = 100000
number_of_points = 1000 #for graphical representation
return_period = integration_steps/number_of_points #for analayzing of quantities

#lennard jones parameters
epsilon_AA = 0
sigma_AA = 0
epsilon_AB = 0
sigma_AB = 0
epsilon_BB = 0
sigma_BB = 0

r_cut = 2.5 #cut-off radius

number_of_simulation = 1000

#Kuhn length b
Kuhn_length_b = math.sqrt(3*kT/k)

#Rouse time
tau_rouse_theory = gamma*Monomers**2*Kuhn_length_b**2 / (3*np.pi**2*kT)

quantities_file = open('quantities_final.dat', 'w')

for step in range (0, int(integration_steps)):
    quantities_file.write('0	0	0	0\n')
quantities_file.close()


#create box with particles (large eonough!)
Lx = 10 * Monomers
Ly = 10 * Monomers
Lz = 10 * Monomers
#define the system as snapshot
snapshot = hoomd.data.make_snapshot(N = Monomers + passive_particles,
                                    box=hoomd.data.boxdim(Lx=Lx, Ly=Ly, Lz=Lz),
                                    particle_types=['A','B'],
                                    bond_types=['polymer'])

#Set ID's positions and bonds for monomers
initial_IDs = []
initial_positions = []
initial_bonds= []
initial_position_of_cm_x = 0
initial_position_of_cm_y = 0
initial_position_of_cm_z = 0

for monomer in range(Monomers):
    initial_IDs.append(0)
    initial_positions.append([monomer - (Monomers/2) + 0.5, 0, 0])
    
    #initial distance of center of mass from origin
    initial_position_of_cm_x += (initial_positions[monomer][0] / Monomers)
    initial_position_of_cm_y += (initial_positions[monomer][1] / Monomers)
    initial_position_of_cm_z += (initial_positions[monomer][2] / Monomers)
    
initial_position_of_cm = [initial_position_of_cm_x,
                          initial_position_of_cm_y,
                          initial_position_of_cm_z]

initial_end_to_end_vector = [initial_positions[Monomers-1][0]-initial_positions[0][0],
                             initial_positions[Monomers-1][1]-initial_positions[0][1],
                             initial_positions[Monomers-1][2]-initial_positions[0][2]]
    
for bond in range(Monomers-1):
    initial_bonds.append([bond, bond+1])
    
#resize the bonds list to actual number of bonds
snapshot.bonds.resize(Monomers-1)

#set initial passive particle positions and IDs
for passive_particle in range(passive_particles):
    initial_IDs.append(1)
    initial_positions.append([passive_particle - (passive_particles/2) + 0.5, 1, 0])
    
    
snapshot.particles.typeid[:] = initial_IDs
snapshot.particles.position[:] = initial_positions
snapshot.bonds.group[:] = initial_bonds



#Initialization of integration parameters
system = hoomd.init.read_snapshot(snapshot)
#bond type and strength between monomers
harmonic = hoomd.md.bond.harmonic()
harmonic.bond_coeff.set('polymer', k=k, r0=r0)

all = hoomd.group.all()  #every partcle

#for visualisation
hoomd.dump.gsd("trajectory.gsd",
               period=return_period,
               group=all, overwrite=True)


#set integrator and drag coefficient
hoomd.md.integrate.mode_standard(dt=dt)
integrator = hoomd.md.integrate.brownian(group=all, kT=kT, dscale=False,
                                         seed=np.random.randint(0,9999))
integrator.set_gamma('A', gamma=gamma)


#Load quantities file
quantities_file = open('quantities_final.dat','r')
all_lines = quantities_file.readlines()[1:]
quantities_file.close()

#Create classes to compute quantities
class Sq_end_to_end_distance:
    
    def __init__(self, system):
        
        self.system = system
        
    def __call__(self, timestep):
        
        snapshot = self.system.take_snapshot()
        
        old_sq_end_to_end_distance = float(all_lines[int(timestep/return_period)].split('	')[1])
        
        #define positions of first and last monomer
        m_first = snapshot.particles.position[0]
        m_last = snapshot.particles.position[-1]
        
        current_sq_end_to_end_distance = ((m_last[0]-m_first[0])**2 +
                                          (m_last[1]-m_first[1])**2 +
                                          (m_last[2]-m_first[2])**2)
        
        #add current sq_end_to_end_distance
        summed_sq_end_to_end_distance = old_sq_end_to_end_distance + current_sq_end_to_end_distance
        
        #return summed end-to-end distance
        return (summed_sq_end_to_end_distance)


class Sq_distance_of_cm:
    
    def __init__(self, system):
        
        self.system = system
        
    def __call__(self, timestep):
        
        snapshot = self.system.take_snapshot()
        
        #old distance of centre of mass from origin (regarding to certain time step)
        old_sq_distance_of_cm = float(all_lines[int(timestep/return_period)].split('	')[2])
        
        #define lists of x,y and z positions of monomers
        list_of_x_positions = []
        list_of_y_positions = []
        list_of_z_positions = []
        
        #current position of centre of mass
        for monomers in range(Monomers):
            
            list_of_x_positions.append(snapshot.particles.position[monomers][0])
            list_of_y_positions.append(snapshot.particles.position[monomers][1])
            list_of_z_positions.append(snapshot.particles.position[monomers][2])
            
        current_position_of_cm = [np.mean(list_of_x_positions),
                                  np.mean(list_of_y_positions),
                                  np.mean(list_of_z_positions)]
        
        
        current_sq_distance_of_cm = ((current_position_of_cm[0]-initial_position_of_cm[0])**2 +
                                     (current_position_of_cm[1]-initial_position_of_cm[1])**2 +
                                     (current_position_of_cm[2]-initial_position_of_cm[2])**2)
        
        summed_sq_distance_of_cm = old_sq_distance_of_cm + current_sq_distance_of_cm
        
        return (summed_sq_distance_of_cm)



class Auto_corr_ee_vector:
    
    def __init__(self, system):
        
        self.system = system
        
    def __call__(self, timestep):
        
        snapshot = self.system.take_snapshot()
        
        old_auto_corr_ee_vector = float(all_lines[int(timestep/return_period)].split('	')[3])
        
        end_to_end_vector_of_t = [(snapshot.particles.position[-1][0] -
                                   snapshot.particles.position[0][0]),
                                  
                                  (snapshot.particles.position[-1][1] -
                                   snapshot.particles.position[0][1]),
                                  
                                  (snapshot.particles.position[-1][2] -
                                   snapshot.particles.position[0][2])]
        
        auto_corr_ee_vector = (end_to_end_vector_of_t[0] * initial_end_to_end_vector[0] +
                               end_to_end_vector_of_t[1] * initial_end_to_end_vector[1] +
                               end_to_end_vector_of_t[2] * initial_end_to_end_vector[2])
        
        summed_auto_corr_ee_vector = old_auto_corr_ee_vector + auto_corr_ee_vector
        
        return (summed_auto_corr_ee_vector)







#Create instances for classes and log the quantities
instance_sq_end_to_end_distance = Sq_end_to_end_distance(system)
instance_sq_distance_of_cm      = Sq_distance_of_cm(system)
instance_auto_corr_ee_vector    = Auto_corr_ee_vector(system)


logger = hoomd.analyze.log(filename='quantities_final.dat',
                           quantities=['summed_sq_end_to_end_distance',
                                       'summed_sq_distance_of_cm',
                                       'summed_auto_corr_ee_vector'],
                           
                           #return and write to file
                           period = return_period,
                           header_prefix='#',
                           overwrite=True)

#create a new quantity that is logged due to expression above to a text file
logger.register_callback(('summed_sq_end_to_end_distance'), instance_sq_end_to_end_distance)
logger.register_callback(('summed_sq_distance_of_cm'), instance_sq_distance_of_cm)
logger.register_callback(('summed_auto_corr_ee_vector'), instance_auto_corr_ee_vector)




hoomd.run(integration_steps)

