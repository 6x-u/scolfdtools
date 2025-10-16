import math as m

pi=m.pi
e=m.e
c=299792458
h=6.62607015e-34
hbar=h/(2*pi)
G=6.67430e-11
k=1.380649e-23
NA=6.02214076e23
epsilon0=8.8541878128e-12
mu0=1.25663706212e-6
me=9.1093837015e-31
mp=1.67262192369e-27
qe=1.602176634e-19
R=8.314462618

def velocity(d,t):return d/t
def acceleration(v,t):return v/t
def force(m,a):return m*a
def momentum(m,v):return m*v
def kinetic_energy(m,v):return 0.5*m*v**2
def potential_energy(m,g,h):return m*g*h
def work(f,d):return f*d
def power(w,t):return w/t
def impulse(f,t):return f*t
def angular_velocity(theta,t):return theta/t
def angular_acceleration(omega,t):return omega/t
def torque(f,r):return f*r
def moment_of_inertia_rod(m,l):return m*l**2/12
def moment_of_inertia_disk(m,r):return 0.5*m*r**2
def moment_of_inertia_sphere(m,r):return 0.4*m*r**2
def angular_momentum(I,omega):return I*omega
def centripetal_force(m,v,r):return m*v**2/r
def centripetal_acceleration(v,r):return v**2/r
def gravitational_force(m1,m2,r):return G*m1*m2/r**2
def escape_velocity(m,r):return m.sqrt(2*G*m/r)
def orbital_velocity(m,r):return m.sqrt(G*m/r)
def orbital_period(r,m):return 2*pi*m.sqrt(r**3/(G*m))
def gravitational_potential(m,r):return -G*m/r
def schwarzschild_radius(m):return 2*G*m/c**2
def time_dilation(v):return 1/m.sqrt(1-(v/c)**2)
def length_contraction(v):return m.sqrt(1-(v/c)**2)
def relativistic_mass(m0,v):return m0/m.sqrt(1-(v/c)**2)
def relativistic_energy(m,v):return m*c**2/m.sqrt(1-(v/c)**2)
def rest_energy(m):return m*c**2
def relativistic_momentum(m,v):return m*v/m.sqrt(1-(v/c)**2)
def lorentz_factor(v):return 1/m.sqrt(1-(v/c)**2)
def doppler_shift(v,f):return f*m.sqrt((c+v)/(c-v))
def compton_wavelength(m):return h/(m*c)
def de_broglie_wavelength(m,v):return h/(m*v)
def photoelectric_energy(f,phi):return h*f-phi
def photon_energy(f):return h*f
def photon_momentum(lam):return h/lam
def planck_law(lam,T):return 2*h*c**2/lam**5/(m.exp(h*c/(lam*k*T))-1)
def wien_displacement(T):return 2.898e-3/T
def stefan_boltzmann(T):return 5.670374419e-8*T**4
def rayleigh_jeans(lam,T):return 2*c*k*T/lam**4
def bohr_radius(n):return n**2*hbar**2/(me*qe**2*4*pi*epsilon0)
def bohr_energy(n):return -13.6/n**2
def rydberg_frequency(n1,n2):return 1.097e7*(1/n1**2-1/n2**2)
def fine_structure():return qe**2/(4*pi*epsilon0*hbar*c)
def compton_scattering(lam,theta):return lam+h/(me*c)*(1-m.cos(theta))
def pair_production_threshold():return 2*me*c**2
def bremsstrahlung_energy(Z,E):return Z**2*qe**4*E/(16*pi**2*epsilon0**2*hbar**2*c**2)
def cherenkov_angle(n,v):return m.acos(c/(n*v))
def schrodinger_1d(psi,V,E,m,x):return -hbar**2/(2*m)*psi+V*psi-E*psi
def heisenberg_uncertainty(dx,dp):return dx*dp>=hbar/2
def wave_function_norm(psi):return sum(abs(p)**2 for p in psi)
def expectation_value(psi,operator,x):return sum(psi[i].conjugate()*operator[i]*psi[i]for i in range(len(psi)))
def commutator(A,B):return[[sum(A[i][k]*B[k][j]-B[i][k]*A[k][j]for k in range(len(A)))for j in range(len(A[0]))]for i in range(len(A))]
def pauli_x():return[[0,1],[1,0]]
def pauli_y():return[[0,-1j],[1j,0]]
def pauli_z():return[[1,0],[0,-1]]
def spin_up():return[1,0]
def spin_down():return[0,1]
def density_matrix(psi):return[[psi[i]*psi[j].conjugate()for j in range(len(psi))]for i in range(len(psi))]
def von_neumann_entropy(rho):eig=[e for e in rho if e>0];return -sum(e*m.log(e)for e in eig)
def fidelity(psi1,psi2):return abs(sum(psi1[i].conjugate()*psi2[i]for i in range(len(psi1))))**2
def bloch_sphere(theta,phi):return[m.sin(theta)*m.cos(phi),m.sin(theta)*m.sin(phi),m.cos(theta)]
def cnot_gate():return[[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
def hadamard_gate():return[[1/m.sqrt(2),1/m.sqrt(2)],[1/m.sqrt(2),-1/m.sqrt(2)]]
def phase_gate(phi):return[[1,0],[0,m.exp(1j*phi)]]
def toffoli_gate():return[[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0]]
def bell_state():return[1/m.sqrt(2),0,0,1/m.sqrt(2)]
def ghz_state(n):return[1/m.sqrt(2)if i==0 or i==2**n-1 else 0 for i in range(2**n)]
def quantum_teleportation(psi,bell):return[psi[i]*bell[j]for i in range(len(psi))for j in range(len(bell))]
def grover_oracle(n,target):return[[-1 if i==target else 1,0][i==j]for i in range(2**n)for j in range(2**n)]
def grover_diffusion(n):return[[2/2**n-[1,0][i==j]for j in range(2**n)]for i in range(2**n)]
def shor_period_finding(a,N):import math;return math.gcd(a,N)
def quantum_fourier_transform(n):omega=m.exp(2j*pi/2**n);return[[omega**(i*j)/m.sqrt(2**n)for j in range(2**n)]for i in range(2**n)]
def coulomb_potential(q1,q2,r):return q1*q2/(4*pi*epsilon0*r)
def electric_field(q,r):return q/(4*pi*epsilon0*r**2)
def electric_potential(q,r):return q/(4*pi*epsilon0*r)
def capacitance_parallel(A,d):return epsilon0*A/d
def capacitance_spherical(r):return 4*pi*epsilon0*r
def capacitance_cylindrical(l,a,b):return 2*pi*epsilon0*l/m.log(b/a)
def electric_energy_density(E):return 0.5*epsilon0*E**2
def magnetic_field_wire(I,r):return mu0*I/(2*pi*r)
def magnetic_field_solenoid(n,I):return mu0*n*I
def magnetic_force(q,v,B):return q*v*B
def lorentz_force(q,E,v,B):return q*(E+v*B)
def magnetic_flux(B,A):return B*A
def inductance_solenoid(n,A,l):return mu0*n**2*A/l
def mutual_inductance(M,I):return M*I
def magnetic_energy_density(B):return B**2/(2*mu0)
def faraday_law(dphi,dt):return -dphi/dt
def lenz_law(I,B,A):return -I*B*A
def ampere_law(I,r):return mu0*I/(2*pi*r)
def biot_savart(I,dl,r):return mu0*I*dl/(4*pi*r**2)
def poynting_vector(E,B):return E*B/mu0
def electromagnetic_energy(E,B):return 0.5*(epsilon0*E**2+B**2/mu0)
def wave_equation_em(E,t,x):return E*m.sin(2*pi*(x/c-t))
def maxwell_displacement(E,t):return epsilon0*E/t
def skin_depth(f,sigma):return m.sqrt(2/(2*pi*f*mu0*sigma))
def reflection_coefficient(n1,n2):return((n1-n2)/(n1+n2))**2
def transmission_coefficient(n1,n2):return 4*n1*n2/(n1+n2)**2
def brewster_angle(n1,n2):return m.atan(n2/n1)
def critical_angle(n1,n2):return m.asin(n2/n1)
def snell_law(n1,theta1,n2):return m.asin(n1*m.sin(theta1)/n2)
def fresnel_equations(n1,n2,theta):rs=(n1*m.cos(theta)-n2*m.sqrt(1-(n1/n2*m.sin(theta))**2))/(n1*m.cos(theta)+n2*m.sqrt(1-(n1/n2*m.sin(theta))**2));return rs**2
def malus_law(I0,theta):return I0*m.cos(theta)**2
def rayleigh_scattering(lam):return 1/lam**4
def mie_scattering(r,lam):return(2*pi*r/lam)**2
def diffraction_single_slit(a,lam,theta):return m.sin(pi*a*m.sin(theta)/lam)/(pi*a*m.sin(theta)/lam)
def diffraction_double_slit(d,lam,theta):return m.cos(pi*d*m.sin(theta)/lam)**2
def diffraction_circular(D,lam,theta):return(2*1.22*lam/D)**2
def interferometer_path_diff(d,theta):return d*m.sin(theta)
def fabry_perot_transmission(R,delta):return 1/(1+4*R/(1-R)**2*m.sin(delta/2)**2)
def michelson_visibility(I1,I2):return 2*m.sqrt(I1*I2)/(I1+I2)
def coherence_length(delta_lam):return delta_lam**2/(2*pi)
def coherence_time(delta_f):return 1/delta_f
def thermal_expansion_linear(L0,alpha,dT):return L0*(1+alpha*dT)
def thermal_expansion_volume(V0,beta,dT):return V0*(1+beta*dT)
def heat_capacity(m,c,dT):return m*c*dT
def specific_heat_gas(n,R,dT):return n*R*dT
def latent_heat(m,L):return m*L
def heat_conduction(k,A,dT,dx):return k*A*dT/dx
def heat_convection(h,A,dT):return h*A*dT
def stefan_boltzmann_radiation(sigma,A,T):return sigma*A*T**4
def carnot_efficiency(Th,Tc):return 1-Tc/Th
def otto_efficiency(r,gamma):return 1-1/r**(gamma-1)
def diesel_efficiency(r,rc,gamma):return 1-1/r**(gamma-1)*(rc**gamma-1)/(gamma*(rc-1))
def entropy_change(Q,T):return Q/T
def gibbs_free_energy(H,T,S):return H-T*S
def helmholtz_free_energy(U,T,S):return U-T*S
def chemical_potential(G,n):return G/n
def clausius_clapeyron(L,T,dP,dT):return L/(T*dP/dT)
def van_der_waals(P,V,n,T):return(P+n**2*a/V**2)*(V-n*b)-n*R*T
def ideal_gas_law(P,V,n,T):return P*V-n*R*T
def maxwell_distribution(v,m,T):return(m/(2*pi*k*T))**(3/2)*4*pi*v**2*m.exp(-m*v**2/(2*k*T))
def boltzmann_distribution(E,T):return m.exp(-E/(k*T))
def partition_function(E,T):return sum(m.exp(-e/(k*T))for e in E)
def mean_free_path(d,n):return 1/(m.sqrt(2)*pi*d**2*n)
def collision_frequency(v,d,n):return m.sqrt(2)*pi*d**2*n*v
def diffusion_coefficient(v,l):return v*l/3
def viscosity_gas(m,T,d):return 5/16*m.sqrt(m*k*T/pi)/d**2
def thermal_conductivity_gas(cv,eta):return 2.5*cv*eta
def sound_speed(gamma,R,T,M):return m.sqrt(gamma*R*T/M)
def doppler_effect_sound(v,vs,f):return f*(v+vs)/v
def mach_number(v,vs):return v/vs
def shock_wave_angle(M):return m.asin(1/M)
def prandtl_meyer_angle(M,gamma):return m.sqrt((gamma+1)/(gamma-1))*m.atan(m.sqrt((gamma-1)/(gamma+1)*(M**2-1)))-m.atan(m.sqrt(M**2-1))
def reynolds_number(rho,v,L,mu):return rho*v*L/mu
def froude_number(v,g,L):return v/m.sqrt(g*L)
def weber_number(rho,v,L,sigma):return rho*v**2*L/sigma
def nusselt_number(h,L,k):return h*L/k
def prandtl_number(mu,cp,k):return mu*cp/k
def grashof_number(g,beta,dT,L,nu):return g*beta*dT*L**3/nu**2
def rayleigh_number(g,beta,dT,L,alpha,nu):return g*beta*dT*L**3/(alpha*nu)
def biot_number(h,L,k):return h*L/k
def fourier_number(alpha,t,L):return alpha*t/L**2
def bernoulli_equation(P1,rho,v1,g,h1,P2,v2,h2):return P1+0.5*rho*v1**2+rho*g*h1-(P2+0.5*rho*v2**2+rho*g*h2)
def poiseuille_flow(mu,L,r,dP):return pi*r**4*dP/(8*mu*L)
def stokes_drag(mu,r,v):return 6*pi*mu*r*v
def drag_force(rho,v,A,Cd):return 0.5*rho*v**2*A*Cd
def lift_force(rho,v,A,Cl):return 0.5*rho*v**2*A*Cl
def terminal_velocity(m,g,rho,A,Cd):return m.sqrt(2*m*g/(rho*A*Cd))
def archimedes_principle(rho,V,g):return rho*V*g
def pascal_law(F1,A1,F2,A2):return F1/A1-F2/A2
def young_modulus(stress,strain):return stress/strain
def shear_modulus(tau,gamma):return tau/gamma
def bulk_modulus(P,dV,V):return -P/(dV/V)
def poisson_ratio(lateral,axial):return -lateral/axial
def hooke_law(k,x):return k*x
def strain_energy(k,x):return 0.5*k*x**2
def stress_strain(E,epsilon):return E*epsilon
def torsion_angle(T,L,G,J):return T*L/(G*J)
def bending_moment(F,L):return F*L
def deflection_beam(F,L,E,I):return F*L**3/(3*E*I)
def critical_load_column(E,I,L):return pi**2*E*I/L**2
def fracture_toughness(K,sigma,a):return K/(sigma*m.sqrt(pi*a))
def fatigue_life(S,N):return S*N**(-1/3)
def creep_rate(sigma,T,Q,R):return m.exp(-Q/(R*T))*sigma**3
def dislocation_density(tau,G,b):return tau/(G*b)
def grain_boundary_energy(gamma):return 2*gamma
def surface_tension(F,L):return F/L
def contact_angle(gamma_sv,gamma_sl,gamma_lv):return m.acos((gamma_sv-gamma_sl)/gamma_lv)
def capillary_rise(gamma,theta,rho,g,r):return 2*gamma*m.cos(theta)/(rho*g*r)
def nucleation_rate(delta_G,k,T):return m.exp(-delta_G/(k*T))
def crystal_growth_rate(v,D,C):return v*D*C
def phase_diagram_lever_rule(X,Xa,Xb):return(X-Xa)/(Xb-Xa)
def diffusion_flux(D,dC,dx):return -D*dC/dx
def fick_second_law(D,d2C,dx2):return D*d2C/dx2
def arrhenius_equation(A,Ea,R,T):return A*m.exp(-Ea/(R*T))
def activation_energy(k1,k2,T1,T2,R):return R*m.log(k2/k1)/(1/T1-1/T2)
def reaction_rate(k,A,B):return k*A*B
def equilibrium_constant(products,reactants):return products/reactants
def nernst_equation(E0,R,T,n,F,Q):return E0-R*T/(n*F)*m.log(Q)
def butler_volmer(i0,alpha,n,F,eta,R,T):return i0*(m.exp(alpha*n*F*eta/(R*T))-m.exp(-(1-alpha)*n*F*eta/(R*T)))
def tafel_equation(i,i0,alpha,n,F,eta,R,T):return R*T/(alpha*n*F)*m.log(i/i0)
def electrochemical_potential(mu,z,F,phi):return mu+z*F*phi
def ionic_conductivity(z,F,D,c):return z**2*F**2*D*c/(R*T)
def debye_length(epsilon,k,T,z,e,n):return m.sqrt(epsilon*k*T/(z**2*e**2*n))
def zeta_potential(epsilon,eta,E):return eta*E/epsilon
def einstein_relation(D,mu,k,T):return D/(mu*k*T)
def smoluchowski_equation(dphi,dt,D,d2phi,dx2):return dphi/dt-D*d2phi/dx2
def langevin_equation(m,dv,dt,gamma,v,F):return m*dv/dt+gamma*v-F
def fokker_planck(dp,dt,d,dx,A,p,d2,dx2,B,p2):return dp/dt+d/dx*(A*p)-d2/dx2*(B*p2)
def kramers_rate(omega,gamma,E,k,T):return omega/(2*pi*gamma)*m.exp(-E/(k*T))
def marcus_theory(lambda_val,delta_G,k,T):return m.exp(-(lambda_val+delta_G)**2/(4*lambda_val*k*T))
def forster_resonance(R0,r):return 1/(1+(r/R0)**6)
def dexter_transfer(J,r):return J*m.exp(-2*r/L)
def fluorescence_lifetime(kr,knr):return 1/(kr+knr)
def quantum_yield(kr,knr):return kr/(kr+knr)
def extinction_coefficient(A,c,l):return A/(c*l)
def beer_lambert(I0,epsilon,c,l):return I0*m.exp(-epsilon*c*l)
def raman_shift(lam0,lam1):return 1/lam0-1/lam1
def ir_absorption(k,c,l):return m.exp(-k*c*l)
def nmr_chemical_shift(f,f0):return(f-f0)/f0*1e6
def epr_g_factor(h,f,mu_B,B):return h*f/(mu_B*B)
def mossbauer_isomer(delta,R):return delta/R
def xray_diffraction(n,lam,d,theta):return n*lam-2*d*m.sin(theta)
def compton_edge(E0,theta):return E0/(1+E0/(me*c**2)*(1-m.cos(theta)))
def pair_production(E):return E-2*me*c**2
def bremsstrahlung(Z,E):return Z**2*e**4*E/(16*pi**2*epsilon0**2*hbar**2*c**2)
def synchrotron_power(q,a,c):return q**2*a**2/(6*pi*epsilon0*c**3)
def cyclotron_frequency(q,B,m):return q*B/m
def plasma_frequency(n,e,m,epsilon0):return m.sqrt(n*e**2/(m*epsilon0))
def debye_shielding(n,T,e,epsilon0,k):return m.sqrt(epsilon0*k*T/(n*e**2))
def alfven_speed(B,rho,mu0):return B/m.sqrt(rho*mu0)
def magnetohydrodynamic(sigma,v,B,E):return sigma*(E+v*B)
def hall_coefficient(n,e):return 1/(n*e)
def seebeck_coefficient(dV,dT):return dV/dT
def peltier_coefficient(Q,I):return Q/I
def thomson_coefficient(dS,dT):return dS/dT
def wiedemann_franz(k,sigma,T):return k/(sigma*T)
def lorenz_number():return pi**2*k**2/(3*e**2)
def fermi_energy(n,h):return h**2/(8*me)*(3*n/pi)**(2/3)
def fermi_temperature(EF,k):return EF/k
def richardson_constant(m,e,k,h):return 4*pi*m*e*k**2/h**3
def work_function(phi,e):return phi/e
def schottky_barrier(phi_m,chi_s):return phi_m-chi_s
def tunneling_probability(E,V,d):return m.exp(-2*d*m.sqrt(2*me*(V-E))/hbar)
def josephson_current(Ic,delta):return Ic*m.sin(delta)
def shapiro_steps(V,f,n):return n*h*f/(2*e)
def flux_quantum():return h/(2*e)
def london_penetration(n,m,e,mu0):return m.sqrt(m/(mu0*n*e**2))
def coherence_length_sc(hbar,m,Tc):return hbar/m.sqrt(2*m*k*Tc)
def bcs_gap(Tc,k):return 1.76*k*Tc
def cooper_pair_size(hbar,m,EF):return hbar/m.sqrt(2*m*EF)

print(f"XIR Physics Engine Loaded: {len([x for x in dir() if not x.startswith('_')])} functions")
