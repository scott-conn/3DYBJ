#code to solve 3D YBJ equation coupled with QG evolution
#QQ implementation is based on the QG example from (Burns, Keaton J., et al. "Dedalus: A flexible framework for numerical simulations with spectral methods." Physical Review Research 2.2 (2020): 023068.)
#https://github.com/DedalusProject/methods_paper_examples/tree/6f08b60361b721c20d0cf044f9b611104b4b1491/quasigeostrophic_flow
#Ls = vector containing length of domain in (x,y,z)
#ns = vector containing number of grid points in (x,y,z)
#f_0 = Coriolis parameter
#Hm = mixed layer depth
#event_file = file containing stratification, streamfunction and winds
#Nt = number of intertial periods to simulate
#Nw = spacing between writing output files
#time_step = fraction of inertial period to use for time-step

import h5py
from mpi4py import MPI
from dedalus import public as de
from dedalus.core import operators
from dedalus.extras import flow_tools
import numpy as np
import matplotlib.pyplot as plt
import time
import logging 
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")
logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if MPI.COMM_WORLD.size > (Nx // 2):
    mesh = [(Nx // 2), MPI.COMM_WORLD.size // (Nx // 2)]
else:
    mesh = None
    
def spin_up_QG(Ls,ns,p,init_file,strat_file)
    Lx, Ly, Lz = Ls                                                                     #length of each dimension
    nx, ny, nz = ns                                                                     #number of grid cells in each dimension
    Tx, Ty, Tz = (Lx/nx,Ly/ny,Lz/nz)
    x_basis = de.Fourier('x', nx, interval=(-Lx/2, Lx/2), dealias=3/2)                  #create x-dimension
    y_basis = de.Fourier('y', ny, interval=(-Ly/2, Ly/2), dealias=3/2)                  #create y-dimension 
    z_basis = de.Chebyshev('z', nz, interval=(-Lz, 0), dealias=3/2)                     #create z-dimension
    
    #parameters
    f_0 = p[1]
    T = 2*np.pi/f_0
    Nt = p[2]
    dt = p[3]
    nuq = p[4]
    kappa = p[5]
    N2 = domain.new_field()
    N2.meta['x','y']['constant'] = True
    hf = h5py.File(strat_file, 'r')                                                    
    N2_data = hf.get('N2')
    N2['g'] = N2_data
    hf.close 
    
    #create IVP with the streamfunction (psi) and vertical velocity (w) as parameters
    problem = de.IVP(domain, variables=['psi','w'],ncc_cutoff=1e-8)
    problem.meta[:]['z']['dirichlet'] = True

    #define differential operators, QGPV and buoyancy
    problem.substitutions['L(a)'] = "d(a,x=2) + d(a,y=2)"
    problem.substitutions['HD(a)'] = "L(L(L(L(a))))"   
    problem.substitutions["J(f,g)"] = "dx(f)*dy(g)-dy(f)*dx(g)"
    problem.substitutions['q'] = "L(psi) + dz(f**2*dz(psi)/N2)"
    problem.substitutions['b'] = "f*dz(psi)"
    
    problem.parameters['f'] = f_0
    problem.parameters['N2'] = N2
    probem.parameters['nu'] = nuq
    problem.parameters['kappa'] = kappa
    
    problem.add_equation("dt(q) + nu*HD(q)  = -J(psi,q)")
    problem.add_equation("dt(b) + N2*w  + kappa*HD(b) = -J(psi,b)")
    problem.add_bc(" left(w) = 0", condition="(nx != 0)  or (ny != 0)")
    problem.add_bc("right(w) = 0", condition="(nx != 0)  or (ny != 0)")
    problem.add_bc("left(psi) = 0", condition="(nx == 0) and (ny == 0)")
    problem.add_bc("right(psi) = 0", condition="(nx == 0) and (ny == 0)")
    
    psi_init, b_init, Q_init = domain.new_fields(3)
    for g in [psi_init, b_init, Q_init]:
        g.set_scales(domain.dealias)

    #slices ensures that in parallel the right part of the input file is given to the right node
    slices = domain.dist.grid_layout.slices(scales=1)                                           
    hf = h5py.File(init_file, 'r')  
    psi_data = hf.get('psi')                                                                
    psi_init['g'] = psi_data[:,:,:][slices]
    hf.close()
    
    #Invert the initial psi for w
    # Need to solve for W_init using dt(P) --> Pt_init as a slack variable.
    init_problem = de.LBVP(domain, variables=['Pt','w'])
    init_problem.meta[:]['z']['dirichlet'] = True
    init_problem.substitutions = problem.substitutions
    init_problem.parameters    = problem.parameters
    init_problem.parameters['psi'] = psi_init
    init_problem.add_equation(" L(Pt) =  nu*HD(q)    - J(psi,q)")
    init_problem.add_equation("f*dz(Pt) + N2*w  = J(psi,b) - kappa*HD(b)")
    init_problem.add_bc(" left(W) =   0", condition="(nx != 0)  or (ny != 0)")
    init_problem.add_bc("right(W)  = 0", condition="(nx != 0)  or (ny != 0)")
    init_problem.add_bc("left(Pt) = 0", condition="(nx == 0) and (ny == 0)")
    init_problem.add_bc("right(Pt) = 0", condition="(nx == 0) and (ny == 0)")

    # Init solver
    init_solver = init_problem.build_solver()
    init_solver.solve()

    psi = solver.state['psi']
    w = solver.state['w']
    for g in [psi,w]: g.set_scales(domain.dealias)

    psi['g'] = psi_init['g']

    init_solver.state['w'].set_scales(domain.dealias)
    w['g'] = init_solver.state['w']['g']
               
    #set spin-up time
    solver.stop_iteration = Nt*T
    solver.stop_sim_time  = np.inf
    solver.stop_wall_time = np.inf           
               
    #run simulation
    logger.info('Starting loop')
    start_time = time.time()
    dt = time_step
    while solver.ok:
        solver.step(dt)                                                             #advance simulation
        if (solver.iteration-1) % 1000 == 0:
            for field in solver.state.fields: field.require_grid_space()
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
               
    return solver     
               
def run_sim(Ls,ns,p,solver_QG,strat_file,wind_file):
    
    Lx, Ly, Lz = Ls                                                                     #length of each dimension
    nx, ny, nz = ns                                                                     #number of grid cells in each dimension
    Tx, Ty, Tz = (Lx/nx,Ly/ny,Lz/nz)
    x_basis = de.Fourier('x', nx, interval=(-Lx/2, Lx/2), dealias=3/2)                  #create x-dimension
    y_basis = de.Fourier('y', ny, interval=(-Ly/2, Ly/2), dealias=3/2)                  #create y-dimension 
    z_basis = de.Chebyshev('z', nz, interval=(-Lz, 0), dealias=3/2)                     #create z-dimension

    #create problem with M, Mz, Mzz, psi and w as the variables
    domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.complex128)
    problem = de.IVP(domain, variables=['M','Mz','Mzz','psi','w'],max_ncc_terms=10)
    
    #parameters
    f_0 = p[1]
    T = 2*np.pi/f_0
    Nt = p[2]
    dt = p[3]
    nuq = p[4]
    kappa = p[5]
    nuM = p[6]
    a = p[7]
    Hm = p[8]
    Nw = p[9]
    M0 = 0+0j
    N2 = domain.new_field()
    N2.meta['x','y']['constant'] = True
    hf = h5py.File(strat_file, 'r')                                                    
    N2_data = hf.get('N2')
    N2['g'] = N2_data
    hf.close
    hf = h5py.File(wind_file, 'r')
    wind_data = hf.get('F')[:]                                                          #extract wind data
    hf.close

    #parameters
    problem.parameters['f'] = f_0
    problem.parameters['Hm'] = Hm
    problem.parameters['nuM'] = nuM
    problem.parameters['nuq'] = nuq
    problem.parameters['kappa'] = kappa
    problem.parameters['Lx'] = Lx
    problem.parameters['Ly'] = Ly
    problem.parameters['Lz'] = Lz
    problem.parameters['a'] = a
    problem.parameters['M0'] = M0
    problem.parameters['N2'] = N2
    problem.parameters['W'] = wind_data[0]

    #substitutions
    problem.substitutions["mag2(f)"] = "f * conj(f)"
    problem.substitutions["forc_z"] = "a*(1+tanh(a*z/Hm+a))/Hm/(a+log(2*cosh(a)))"
    problem.substitutions["forc_zz"] = "a**2/(cosh(a*(z/Hm+1)))**2/Hm**2/(a+log(2*cosh(a)))"
    problem.substitutions["real(f)"] = "(f + conj(f))/2"
    problem.substitutions['L(a)'] = "d(a,x=2) + d(a,y=2)"
    problem.substitutions['HD(a)'] = "L(L(L(L(a))))"   
    problem.substitutions["J(f,g)"] = "dx(f)*dy(g)-dy(f)*dx(g)"
    problem.substitutions['q'] = "L(psi) + dz(f**2*dz(psi)/N2) + 1j*J(conj(Mz),Mz)/2/f + L(mag2(Mz))/4/f"
    problem.substitutions['b'] = "f*dz(psi)"
    
    #equations and b.c.'s
    problem.add_equation("dt(Mzz) + 1j*N2*L(M)/2/f + nuM*HD(Mzz) = -J(psi,Mzz) -1j*L(psi)*Mzz/2 + W*forc_zz") 
    problem.add_equation("Mzz - dz(Mz) = 0")
    problem.add_equation("Mz - dz(M) = 0")
    problem.add_equation("dt(q) + nuq*HD(q)  = -J(psi,q)")
    problem.add_equation("dt(b) + N2*w  + kappa*HD(b) = -J(psi,b)")
    problem.add_bc(" left(w) = 0", condition="(nx != 0)  or (ny != 0)")
    problem.add_bc("right(w) = 0", condition="(nx != 0)  or (ny != 0)")
    problem.add_bc("left(psi) = 0", condition="(nx == 0) and (ny == 0)")
    problem.add_bc("right(psi) = 0", condition="(nx == 0) and (ny == 0)")
    problem.add_bc("right(M) = M_0")
    problem.add_bc("left(M) = 0")

    #make solver
    solver = problem.build_solver(de.timesteppers.RK222)

    #specify when to end the simulation
    solver.stop_sim_time = Nt*T #t_end in the simulation
    solver.stop_wall_time = np.inf #t_end in real life
    solver.stop_iteration = np.inf #number of iterations corresponding to t_end

    #initial conditions
    M = solver.state['M']
    Mz = solver.state['Mz']
    Mzz = solver.state['Mzz']
    psi = solver.state['psi']
    w = solver.state['w']
    M['g'] = 0+0j
    Mz['g'] = 0+0j
    Mzz['g'] = 0+0j
    w['g'] = solver_QG.state['w']['g']
    psi['g'] = solver_QG.state['psi']['g']
    
    #set up output file
    state_file = "state" #output file name
    analysis = solver.evaluator.add_file_handler(state_file, iter=Nw)
    analysis.add_system(solver.state, layout='g')
    analysis.add_task('W',name='wind')
    analysis.add_task('q',name='PV')
    analysis.add_task('b',name='buoyancy')

    logger.info('Starting loop')
    start_time = time.time()
    dt = time_step
    while solver.ok:
        solver.step(dt)                                                             #advance simulation
        M0 = M0 + dt*wind_data[solver.iteration-1]                                  #solve ODE for top b.c.
        problem.namespace['M_0'].value = M0                                         #update top b.c.
        problem.namespace['W'].value = wind_data[solver.iteration]                  #update wind forcing
        if (solver.iteration-1) % 1000 == 0:
            for field in solver.state.fields: field.require_grid_space()
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
    hf.close()

    #cleanup output files
    from dedalus.tools import post
    post.merge_process_files(state_file, cleanup=True)
    post.merge_process_files(energy_file, cleanup=True)
    post.merge_process_files(budget_file, cleanup=True)
