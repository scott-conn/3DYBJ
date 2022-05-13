#Code to run a 3D pseudospectral simulation of the YBJ equation using prescribed vorticity field and stratificaiton profile
#The horizontal directions are taken to be periodic while a finite depth ocean is achieved using an expansion in Chebyshev polynomails
#Ls - vector with elements Lx,Ly,Lz (domain length)
#ns - vector with elements nx,ny,nz (number of grid cells)
#time_step_T - time-step used for simulation in terms of the inertial period (i.e. timestep = time_step_T*T)
#strat_file - string with name of file containing the stratification profile to use
#nu - hyperviscosity coefficient in units of m^4/s
#t_end_T - end time of simulation in terms of the inertial period (i.e. t_end = t_end_T*T)
#psi_file - string with name of file containing the eddy field
#output_file - string with name of file to output state variables to
#saves - integer number of timesteps to save

from mpi4py import MPI
from dedalus import public as de
from dedalus.core import operators
from dedalus.extras import flow_tools
import numpy as np
import matplotlib.pyplot as plt
import time
import logging 
import h5py
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")
logger = logging.getLogger(__name__)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def simulate_YBJ(Ls,ns,time_step_T,strat_file,nu,t_end,psi_file,output_file,saves)
  
        #spatial grid
        Lx, Ly, Lz = (L[0],L[1],L[2])                                                                                      #length of domain in x,y and z directions (m)
        nx, ny, nz = (n[0],n[1],n[2])                                                                                      #number of grid cells in x,y and z directions
        Tx, Ty, Tz = (Lx/nx,Ly/ny,Lz/nz)                                                                                   #grid spacing in x,y and z directions (m)

        
        #create bases
        x_basis = de.Fourier('x', nx, interval=(-Lx/2, Lx/2), dealias=3/2)
        y_basis = de.Fourier('y', ny, interval=(-Ly/2, Ly/2), dealias=3/2)
        z_basis = de.Chebyshev('z', nz, interval=(-Lz, 0), dealias=3/2)

        
        #create problem with M M_z, M_zz and psi as the variables
        domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.complex128)
        problem = de.IVP(domain, variables=['M','Mz','Mzz','psi'])

        
        #problem parameters
        f_0 = 1e-4                                                                                                          #Coriolis parameter (/s)
        N_0 = 1.4e-2                                                                                                        #Stratification (/s)
        T = 2*np.pi/f_0                                                                                                     #inertial period (s)
        problem.parameters['f'] = f_0
        problem.parameters['nu'] = nu                                                                                       #hyperviscosity (m^4/s)
        problem.parameters['Lx'] = Lx
        problem.parameters['Ly'] = Ly
        problem.parameters['Lz'] = Lz
        
        
        #since N isn't constant in z we need to use the non-constant construction
        ncc = domain.new_field(name='c')
        ncc['g'] = strat_profile
        ncc.meta['x', 'y']['constant'] = True
        problem.paramters['N2'] = ncc                                                                                       #Stratification (/s^2)
       
      
        #substitutions
        problem.substitutions["mag2(f)"] = "f * conj(f)"                                                                    #magnitude of complex number
        problem.substitutions["J(f,g)"] = "dx(f)*dy(g)-dy(f)*dx(g)"                                                         #Jacobian
        problem.substitutions["nabla2(f)"] = "dx(dx(f)) + dy(dy(f))"                                                        #Laplacian
        problem.substitutions["nabla4(f)"] = "dx(dx(dx(dx(f))))+2*dx(dx(dy(dy(f))))+dy(dy(dy(dy(f))))"                      #Biharmonic operator

        
        #equations and b.c.'s
        problem.add_equation("dt(Mzz) + 1j*N2*nabla2(M)/2/f + nu*nabla4(Mz) = -J(psi,Mzz) -1j*nabla2(psi)*Mzz/2")           #YBJ equation in M-form
        problem.add_equation("dt(psi) = 0")                                                                                 #Don't evolve psi
        problem.add_equation("Mzz - dz(Mz) = 0")                                                                            #Definition of M_zz
        problem.add_equation("Mz - dz(M) = 0")                                                                              #Definition of M_z
        problem.add_bc("dt(left(M))=0")                                                                                     #Initialise M constant on vertical boundaries and keep constant
        problem.add_bc("dt(right(M))=0")

        
        #make solver using RK2 timestepper
        solver = problem.build_solver(de.timesteppers.RK222)

        
        #specify when to end the simulation
        solver.stop_sim_time = t_end                                                                                        #t_end in the simulation
        solver.stop_wall_time = np.inf                                                                                      #t_end in real life
        solver.stop_iteration = np.inf                                                                                      #number of iterations corresponding to t_end

        
        #initial conditions 
        z = domain.grid(2)
        y = domain.grid(1)
        x = domain.grid(0)
        M = solver.state['M']
        Mz = solver.state['Mz']
        Mzz = solver.state['Mzz']
        psi = solver.state['psi']
        
        hf = h5py.File(psi_file, 'r')                                                                                       #open file where psi ic is
        slices = domain.dist.grid_layout.slices(scales=1)                                                                   #allows us to read in file when paralellised
        psi_data = hf.get('psi')
        psi['g'] = psi_data[slices]
        hf.close()
        
        uNIW = 0.8                                                                                                          #initial wave velocity (m/s)  
        Hm = 50                                                                                                             #mixed layer depth (m)
        Mz['g'] = uNIW*np.exp(-(z/Hm)**2)                                                                                   #initialise waves in ML
        Mz.differentiate('z',out=Mzz)                                                                                       #calculate M_zz
        dt = 1e-10
        solver.step(dt)                                                                                                     #using very small timestep, step forward and hence invert for M

        
        #analysis
        state_file = output_file                                                                                            #file to store state variables
        analysis = solver.evaluator.add_file_handler(state_file, iter=2*saves)
        analysis.add_system(solver.state, layout='g')
        
        energy_file = output_file,+"_energy".                                                                               #file to store energy variables
        analysis_E = solver.evaluator.add_file_handler(energy_file, iter=saves)
        analysis_E.add_task('mag2(Mz)/2/f',name='E')
        analysis_E.add_task('integ(mag2(Mz)/2/f)/Lx/Ly/Lz',name='E_int')
        analysis_E.add_task('integ(N2*(dx(M)*dx(conj(M))+dy(M)*dy(conj(M)))/4/f)',name='P_int')
        analysis_E.add_task('integ(1j*nabla2(psi)*(M*conj(Mz)-conj(M)*Mz)/2)',name='G_zeta')
        analysis_E.add_task('integ(-1*(conj(M)*J(psi,M)+M*J(psi,conj(M)))/2)',name='G_adv')
        analysis_E.add_task("integ(mag2(Mz)/2/f/Ly,'y')",name='E_inty')
        analysis_E.add_task("integ(mag2(Mz)/2/f/Lz,'z')",name='E_intz')

        flux_file = output_file+"_flux"                                                                                     #file to store energy flux terms
        analysis_F = solver.evaluator.add_file_handler(flux_file, iter=2*saves)
        analysis_F.add_task('1j*N2*(M*dx(conj(M))-conj(M)*dx(M))/2/f',name='F_x')
        analysis_F.add_task('1j*N2*(M*dy(conj(M))-conj(M)*dy(M))/2/f',name='F_y')
        analysis_F.add_task('1j*nabla2(psi)*(M*conj(Mz)-conj(M)*Mz)/2',name='G_zeta')
        analysis_F.add_task('-1*(conj(M)*J(psi,M)+M*J(psi,conj(M)))/2',name='G_adv')

        
        #begin simulation
        logger.info('Starting loop')
        start_time = time.time()
        while solver.ok:
            dt = time_step_T*T                                                                                              #time-step (s)
            solver.step(dt)
            if solver.iteration % 100 == 0:
              logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

              
        #merge files 
        from dedalus.tools import post
        post.merge_process_files(state_file, cleanup=True)
        post.merge_process_files(energy_file, cleanup=True)
        post.merge_process_files(flux_file, cleanup=True)
