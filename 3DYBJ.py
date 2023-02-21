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
    
def spin_up_QG(Ls,ns,init_file,event_file,Nt,time_step,f_0)
    Lx, Ly, Lz = Ls                                                                     #length of each dimension
    nx, ny, nz = ns                                                                     #number of grid cells in each dimension
    Tx, Ty, Tz = (Lx/nx,Ly/ny,Lz/nz)
    x_basis = de.Fourier('x', nx, interval=(-Lx/2, Lx/2), dealias=3/2)                  #create x-dimension
    y_basis = de.Fourier('y', ny, interval=(-Ly/2, Ly/2), dealias=3/2)                  #create y-dimension 
    z_basis = de.Chebyshev('z', nz, interval=(-Lz, 0), dealias=3/2)                     #create z-dimension

    #Stratification parameter
    N2 = domain.new_field()
    N2.meta['x','y']['constant'] = True
    hf = h5py.File(event_file, 'r')                                                     #open file containing simulation data
    N2_data = hf.get('N2')
    N2['g'] = N2_data
    close(hf)
    
    problem = de.IVP(domain, variables=['psi','w'],ncc_cutoff=1e-8)
    problem.meta[:]['z']['dirichlet'] = True

    problem.substitutions['L(a)'] = "d(a,x=2) + d(a,y=2)"
    problem.substitutions['HD(a)'] = "L(L(L(L(a))))"   
    problem.substitutions["J(f,g)"] = "dx(f)*dy(g)-dy(f)*dx(g)"
    problem.substitutions['q'] = "L(psi) + dz(f**2*dz(psi)/N2)"
    problem.substitutions['b'] = "f*dz(psi)"
    
    problem.parameters['f'] = f_0
    problem.parameters['N2'] = N2
    
    problem.add_equation("dt(q) + nu*HD(q)  = -J(psi,q)")
    problem.add_equation("dt(b) + N2*w  + kappa*HD(b) = -J(psi,b)")
    problem.add_bc(" left(w) = 0", condition="(nx != 0)  or (ny != 0)")
    problem.add_bc("right(w) = 0", condition="(nx != 0)  or (ny != 0)")
    problem.add_bc("left(psi) = 0", condition="(nx == 0) and (ny == 0)")
    problem.add_bc("right(psi) = 0", condition="(nx == 0) and (ny == 0)")
    
    psi_init, b_init, Q_init = domain.new_fields(3)
    for g in [psi_init, b_init, Q_init]:
        g.set_scales(domain.dealias)

    slices = domain.dist.grid_layout.slices(scales=1)                                           #take right part of psi array when in parallel
    hf = h5py.File(init_file, 'r')  
    psi_data = hf.get('psi')                                                                   #extract psi data
    psi_init['g'] = psi_data[:,:,:][slices]
    hf.close()
    
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
               
    solver.stop_iteration = Nt
    solver.stop_sim_time  = np.inf
    solver.stop_wall_time = np.inf           
               
    logger.info('Starting loop')
    start_time = time.time()
    dt = time_step
    while solver.ok:
        solver.step(dt)                                                             #advance simulation
        if (solver.iteration-1) % 1000 == 0:
            for field in solver.state.fields: field.require_grid_space()
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
               
    return solver     
               
def run_sim(Ls,ns,f_0,Hm,solver_QG,event_file,Nt,Nw,time_step):
    
    Lx, Ly, Lz = Ls                                                                     #length of each dimension
    nx, ny, nz = ns                                                                     #number of grid cells in each dimension
    Tx, Ty, Tz = (Lx/nx,Ly/ny,Lz/nz)
    x_basis = de.Fourier('x', nx, interval=(-Lx/2, Lx/2), dealias=3/2)                  #create x-dimension
    y_basis = de.Fourier('y', ny, interval=(-Ly/2, Ly/2), dealias=3/2)                  #create y-dimension 
    z_basis = de.Chebyshev('z', nz, interval=(-Lz, 0), dealias=3/2)                     #create z-dimension

    #create problem with M, Mz, Mzz, psi and w as the variables
    domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.complex128)
    problem = de.IVP(domain, variables=['M','Mz','Mzz','psi','w'],max_ncc_terms=10)

    #problem parameters
    z = domain.grid(2)
    T = 2*np.pi/f_0                                                                     #inertial period (s)
    a = 2                                                                               #parameter governing sharpness of forcing profile
    M0 = 0+0j                                                                           #start simulation with no NIWs, boundary term on is zero at top
    nuM = 3e6                                                                           #hyperdiffusivity parameter for NIWs (m^4/s)
    nuq = 3e6                                                                           #hyperdiffusivity parameter for q
    kappa = 3e6                                                                         #hyperdiffusivity parameter for b
    dt = time_step*T                                                                    #timestep (s)
    psi_update = 24*60*60//dt                                                           #number of timesteps per day (number of timesteps after which update psi)
    
    hf = h5py.File(event_file, 'r')                                                     #open file containing simulation data
    N2_data = hf.get('N2')                                                              #extract stratification profile
    
    #Dedalus procedure to add parameter varying in z
    ncc = domain.new_field(name='c')                                                    
    ncc['g'] = N2_data
    ncc.meta['x', 'y']['constant'] = True
    
    wind_data = hf.get('F')                                                             #extract wind data

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
    problem.parameters['N2'] = ncc
    problem.parameters['W'] = wind_data[0]

    #substitutions
    problem.substitutions["mag2(f)"] = "f * conj(f)"
    problem.substitutions["forc_z"] = "a*(1+tanh(a*z/Hm+a))/Hm/(a+log(2*cosh(a)))"
    problem.substitutions["forc_zz"] = "a**2/(cosh(a*(z/Hm+1)))**2/Hm**2/(a+log(2*cosh(a)))"
    problem.substitutions["real(f)"] = "(f + conj(f))/2"
    problem.substitutions['L(a)'] = "d(a,x=2) + d(a,y=2)"
    problem.substitutions['HD(a)'] = "L(L(L(L(a))))"   
    problem.substitutions["J(f,g)"] = "dx(f)*dy(g)-dy(f)*dx(g)"
    problem.substitutions['q'] = "L(psi) + dz(f**2*dz(psi)/N2) + 1j*J(conj(Mz),Mz)/2f + L(mag2(Mz))/4/f"
    problem.substitutions['b'] = "f*dz(psi)"
    

    #equations and b.c.'s
    problem.add_equation("dt(Mzz) + 1j*N2*L(M)/2/f + nu*HD(Mzz) = -J(psi,Mzz) -1j*L(psi)*Mzz/2 + W*forc_zz") 
    problem.add_equation("dt(psi) = 0")
    problem.add_equation("Mzz - dz(Mz) = 0")
    problem.add_equation("Mz - dz(M) = 0")
    problem.add_equation("dt(q) + nu*HD(q)  = -J(psi,q)")
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
    
    #set up output files
    #state variables
    state_file = "state" #output file name
    analysis = solver.evaluator.add_file_handler(state_file, iter=Nw)
    analysis.add_system(solver.state, layout='g')
    analysis.add_task('W',name='wind')
    analysis.add_task('nabla2(psi)',name='zeta')

    #energy budget terms
    energy_file = "energy"
    analysis_E = solver.evaluator.add_file_handler(energy_file,iter=Nw)
    analysis_E.add_task('1j*N2*(M*dx(conj(M))-conj(M)*dx(M))/4/f**2',name='Fx')
    analysis_E.add_task('1j*N2*(M*dy(conj(M))-conj(M)*dy(M))/4/f**2',name='Fy')
    analysis_E.add_task('1j*N2*(M*nabla2(conj(M))-conj(M)*nabla2(M))/4/f**2',name='divF')
    analysis_E.add_task('N2*M/f**2',name='Az')
    analysis_E.add_task('J(psi,mag2(Mz))/2/f',name='advection_A')
    analysis_E.add_task('(conj(Mz)*(W*forc_z)+Mz*conj(W*forc_z))/2/f',name='gamma_A')
    analysis_E.add_task('nu*real(conj(Mz)*nabla4(Mz))/f',name='d_A')
    analysis_E.add_task('N2*(dx(M)*dx(conj(M))+dy(M)*dy(conj(M)))/4/f**2',name='PE')

    #domain integrated budget terms
    budget_file = "budget"
    analysis_b = solver.evaluator.add_file_handler(budget_file,iter=Nw)
    analysis_b.add_task('integ(mag2(Mz)/2/f)',name='A')
    analysis_b.add_task('integ(conj(Mz)*(W*forc_z)+Mz*conj(W*forc_z))/2/f',name='Gamma_A')
    analysis_b.add_task('integ(nu*real(conj(Mz)*nabla4(Mz))/f)',name='D_A')
    analysis_b.add_task('integ(N2*(dx(M)*dx(conj(M))+dy(M)*dy(conj(M)))/4/f**2)',name='P')
    analysis_b.add_task('integ(1j*N2*nabla2(psi)*(-M*nabla2(conj(M))+conj(M)*nabla2(M))/8/f**2)',name='Gamma_P_ref')
    #analysis_b.add_task('integ(N2*real(dx(M)*J(dx(psi),conj(M)) + dy(M)*J(dy(psi),conj(M)))/2/f**2)',name='Gamma_P_adv')
    analysis_b.add_task('integ(nu*N2*real(nabla2(conj(M))*nabla4(M))/f**2)',name='D_P')

    logger.info('Starting loop')
    start_time = time.time()
    dt = time_step
    while solver.ok:
        solver.step(dt)                                                             #advance simulation
        M0 = M0 + dt*wind_data[solver.iteration-1]                                  #solve ODE for top b.c.
        problem.namespace['M_0'].value = M0                                         #update top b.c.
        problem.namespace['W'].value = wind_data[solver.iteration]                  #update wind forcing
        
        #check if time to update psi and do so if necessary
        if solver.iteration % psi_update == 0:                                      
            index = np.int64(solver.iteration/psi_update)
            psi.set_scales(1)
            psi_data = psi_data1[:,:,:,index]
            slices = domain.dist.grid_layout.slices(scales=1)
            psi['g'] = psi_data[slices]
            
        if (solver.iteration-1) % 1000 == 0:
            for field in solver.state.fields: field.require_grid_space()
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
    hf.close()

    #cleanup output files
    from dedalus.tools import post
    post.merge_process_files(state_file, cleanup=True)
    post.merge_process_files(energy_file, cleanup=True)
    post.merge_process_files(budget_file, cleanup=True)
