
# coding: utf-8

# In[ ]:


import scipy.integrate as integrate
from scipy import interpolate
import matplotlib.pyplot as plt 
import numpy as np
import scipy.special as spl 
import scipy.optimize as opt
import time



t0 = time.time()

# Let's input g_star

S_star_data= np.transpose( np.loadtxt('S_star.txt' ) )

# Let's use spline interpolation

x_new = [2**i for i in range(-16,15) ]

star_S_splrep=interpolate.splrep(S_star_data[0], S_star_data[1])
star_splrep=interpolate.splrep(S_star_data[0], S_star_data[2])

S_star_S = interpolate.splev(x_new, star_S_splrep )
S_star = interpolate.splev(x_new, star_splrep )


#---------------------------------------------------------------
# Define the function of g_star


def g_star(T):
    
    g_star = interpolate.splev( T, star_splrep )
    
    return(g_star)


#---------------------------------------------------------------
# Define the function of g_star_S

def g_star_S(T):
    
    g_star_S = interpolate.splev( T,  star_S_splrep )
    
    return(g_star_S)





#---------------------------------------------------------------
# Let's define the Ib_eq(Good)  & Ib_eq2(Bad)  for equilibrium situation

# This is a good function (error is small)
def Ib_eq(x,R):
    
    #XR_Bound=23
    #XR_Bound=30
    XR_Bound=20
    
    
    if x*R<XR_Bound:
        
        Ib =  integrate.quad(lambda k: k**2/(np.exp(np.sqrt(k**2+(R*x)**2))-1), 0, Cutoff_k)[0] 
    
    if x*R>=XR_Bound:
        
        Ib = (R*x)**2 * spl.kv(2,R*x)
    
    return(Ib)



#  This is not a good function (error is large)
def Ib_eq2(x,R):
    
    Ib =  integrate.quad(lambda k: k**2/(np.exp(np.sqrt(k**2+(R*x)**2))-1), 0, Cutoff_k)[0] 
    
    return(Ib)



#.......................................................
# This is the Boltzmann distribution(Simple version)
def Yb_eq(x,R):
    
    Yb=  (45/(4*np.pi**4))*(1/g_star_S(m_chi/x))*(1/x**6)*Ib_eq(x,R)   
    
    return(Yb)

def Yb_eq2(x,R):
    
    g_starS = interpolate.splev(m_chi/x, interpolate.splrep(S_star_data[0], S_star_data[1]) )
    
    Ib = np.float128( integrate.quad(lambda k: k**2/(np.exp(np.sqrt(k**2+(R*x)**2))-1), 0, Cutoff_k)[0] )
    
    Yb=  np.float128( (45/(4*np.pi**4))*(1/g_starS)*(1/x**6)*Ib   )
    
    return(Yb)



print("The definition of   >>> g_star <<<    &   >>> g_star_S <<<    is finished")
print("The definition of   >>> Ib_eq  <<<    &   >>> Ib_eq2   <<<    is finished")
print("The definition of   >>> Yb_eq  <<<    &   >>> Yb_eq2   <<<    is finished")
print("")

t1 = time.time()



print("The time of initialization--I  is %s" %(t1 - t0) )
print("")
print("")




# ================================================================================================
# ================================================================================================
# ================================================================================================
# ================================================================================================
# ================================================================================================




class IODE(object):
    
    # Initialize the class "ODE", here some classes 
    def __init__(self, Nt0, T0 , Tf , function, r0):
        
        self.Nt0 = Nt0
        
        self.dt0 = (Tf - T0)/Nt0

        # Initialize the form of f(x,t) in ODE dx/dt = f(x,t)
        self.f = function
        
        # Initialize the t using
        self.T0 = T0
        
        # Initialize the r using  
        # (if r0 is a list, we change it into array)
        self.r0 = np.array(r0)
        
        # The dimension of r
        self.dim_r = int( np.shape(self.r0)[0] )
        
        # The dimension of f  (use~shape(r0,To))
        self.dim_f = int( np.shape(self.f(self.r0,T0)  )[0] )
        
        # Judge whether the dimension of f and r0 matches
        
        if self.dim_r == self.dim_f:
            
            print("dim_r == dim_f, the initialization is consistent")
            

        else:
                
                print("dim_r != dim_f, the initialization fails, the program is stopped")

        # Initialize t_array, r_array to store the consequence
        
        self.t_array = None
        
        self.r_array = None
        

        # Show that initialization is finished
        
        print("The initialization is finished")
        print("--------------------------------------")

    
    def plot(self):
        
        fig = plt.figure(1)
        
        ax1 = plt.subplot(111)
        
        EQ = [ ]
        
        for t in self.t_array:
            
            Y = Yb_eq(A*t,1)
            
            EQ.append(Y)

        ax1.plot( self.t_array, self.r_array[0], marker = 'o', color="blue", ls='', ms=3, mew=0.3, label=r'$Y(x)$')
        ax1.plot( self.t_array, EQ, marker = 'o', color="red", ls='', ms=3, mew=0.3, label=r'$Y^{(eq)}(x)$')
        #ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend( loc='best', fontsize=12 )
        ax1.set_xlabel( r'$x/10$', fontsize=12 )
        ax1.set_ylabel( r'$Y$', fontsize=12 )  
        plt.show()
        
    def output_t_r_array(self):
        
        print("=============================================== ")
        print("Now let us output the consequence --- t_r_array ")
        print("=============================================== ")
        
        t_r_array = np.vstack((self.t_array,self.r_array[0,:]))
        
        print("final Y firstly is %s" %t_r_array[1,self.Nt0] )
        
        return(t_r_array)
    
    def final_Y(self):
        
        print("=============================================== ")
        print("Now let us output the consequence --- final_Y ")
        print("=============================================== ")        
        
        final_Y = self.r_array[0,self.Nt0]
    
        return(final_Y) 
    
    
    
    #######################################################################################
    # Forward Euler
    def evolve_FEuler(self):
        
        # Here dt0 = self.dt0 (dt0 is local variable)
        
        dt0 = self.dt0
        
        # Built t_array to store time during evolution
        t_array = np.zeros(1+self.Nt0)
        
        
        # Built r_array to store r during evolution r = (x_1,x_2,...)
        # The dimension of r_array
        r_dim = self.dim_r
        r_array = np.zeros( [ r_dim , 1+self.Nt0 ], float )
        
        # Put the intial condition r0 into r_array
        r_array[:,0] = self.r0
        
        # Put the initial condition T0 to t_array
        t_array[0] = self.T0
        
        # The time evolution is 
        # t_0(r0)(For loop begins) --> t_1 --> t_2 -->...-->t_Nt0
        
        # i = 0,1,2,3,..., (Nt0-1)
        
        
        # Here we use the Forward Euler method 
        #  r_(i+1) = r_i + dt0*f(r,t)
        for i in range(0,self.Nt0):
 

            if i%100==0:
                
                print("It is %s th step" %i)
                
            
            t_array[i+1] = t_array[i] + dt0
            
            r_array[:,i+1] = r_array[:,i] + dt0 * self.f(r_array[:,i],t_array[i])      
        
        self.t_array = t_array
        self.r_array = r_array
        
        print("The consequence of FEuler is stored in self.t_array and self.r_array")
        print("----------------------------------------------")
    #######################################################################################
    
    
    #######################################################################################
    # RK2 
    def evolve_RK2(self):
        
        

        
                
        # Here dt0 = self.dt0 (dt0 is local variable)
        
        dt0 = self.dt0
        
        # Built t_array to store time during evolution
        t_array = np.zeros(1+self.Nt0)
        
        
        # Built r_array to store r during evolution r = (x_1,x_2,...)
        # The dimension of r_array
        r_dim = self.dim_r
        r_array = np.zeros( [ r_dim , 1+self.Nt0 ], float )
        
        # Put the intial condition r0 into r_array
        r_array[:,0] = self.r0
        
        # Put the initial condition T0 to t_array
        t_array[0] = self.T0
              
        # The time evolution is 
        # t_0(r0)(For loop begins) --> t_1 --> t_2 -->...-->t_Nt0
        
        # i = 0,1,2,3,..., (Nt0-1)
        

        # Here we use RK2:
        #  k1 = h f(x(t),t)
        #  k2 = h f(x(t)+0.5*k1, t+0.5*h)
        #  x(t+h) = x(t) + k2
        
        for i in range(0,self.Nt0):

            if i%100==0:
                
                print("It is %s th step" %i)
            
            
            t_array[i+1] = t_array[i] + dt0
            
            k1i =  dt0 * self.f(r_array[:,i],t_array[i])
            
            k2i =  dt0 * self.f(r_array[:,i]+0.5*k1i,t_array[i]+0.5*dt0)
            
            r_array[:,i+1] =  r_array[:,i] + k2i 
     
        
        self.t_array = t_array
        self.r_array = r_array
        
        print("The consequence of RK2 is stored in self.t_array and self.r_array")
        print("----------------------------------------------")


    #######################################################################################
    
    #######################################################################################
    # RK4
    def evolve_RK4(self):
        
        # Here dt0 = self.dt0 (dt0 is local variable)
        
        dt0 = self.dt0
        
        # Built t_array to store time during evolution
        t_array = np.zeros(1+self.Nt0)
        
        
        # Built r_array to store r during evolution r = (x_1,x_2,...)
        # The dimension of r_array
        r_dim = self.dim_r
        r_array = np.zeros( [ r_dim , 1+self.Nt0 ], float )
        
        # Put the intial condition r0 into r_array
        r_array[:,0] = self.r0
        
        # Put the initial condition T0 to t_array
        t_array[0] = self.T0
        
        
        # The time evolution is 
        # t_0(r0)(For loop begins) --> t_1 --> t_2 -->...-->t_Nt0
        
        # i = 0,1,2,3,..., (Nt0-1)

        
        # Here we use RK4:
        #  k1 = h f(x(t),t)
        #  k2 = h f(x(t)+0.5*k1, t+0.5*h)
        #  x(t+h) = x(t) + k2
        
        for i in range(0,self.Nt0):
            
            if i%100==0:
                
                print("It is %s th step" %i)
            
            t_array[i+1] = t_array[i] + dt0
            
            k1i =  dt0 * self.f(r_array[:,i],t_array[i])
            
            k2i =  dt0 * self.f(r_array[:,i]+0.5*k1i,t_array[i]+0.5*dt0)
            
            k3i =  dt0 * self.f(r_array[:,i]+0.5*k2i,t_array[i]+0.5*dt0)
            
            k4i =  dt0 * self.f(r_array[:,i]+k3i,t_array[i]+dt0)  
            
            
            r_array[:,i+1] =  r_array[:,i] + (1/6)*( k1i + 2*k2i + 2*k3i + k4i )   
        
        
        self.t_array = t_array
        self.r_array = r_array
        
        print("The consequence of RK4 is stored in self.t_array and self.r_array")
        print("----------------------------------------------")


    #######################################################################################
    
    ######################################################################################
    # RK4+adaptive
    
    def evolve_RK4_adp(self,delta):
        
        # Here dt0 = self.dt0 (dt0 is local variable)
        
        dt0 = self.dt0
        
        dt = dt0
        
#        print("initial dt0 before using adaptive-RK4 is %s" %dt0)
        
        # Built t_array to store time during evolution
        t_array = np.zeros(1+self.Nt0)
        
        
        # Built r_array to store r during evolution r = (x_1,x_2,...)
        # The dimension of r_array
        r_dim = self.dim_r
        r_array = np.zeros( [ r_dim , 1+self.Nt0 ], float )
        
        # Put the intial condition r0 into r_array
        r_array[:,0] = self.r0
        
        # Put the initial condition T0 to t_array
        t_array[0] = self.T0
        
        
        # The time evolution is 
        # t_0(r0)(For loop begins) --> t_1 --> t_2 -->...-->t_Nt0
        
        # i = 0,1,2,3,..., (Nt0-1)
        
        # Here we use adaptive RK4:
        # Other is the same as non-adaptive RK4  
        
        for i in range(0,self.Nt0):
            
            if i%100==0:
                
                print("It is %s th step" %i)
            
            
            # Let's initialize rho<1, so the while loop will begin
            # Every time rho will be changed back to be 0.5 
            #      so every for loop, the while loop will begin
            rho = 0.5
            
            
            while rho<1:
                
                #let's calculate the x1 using "dt"
                # Here: x(t+dt) = x1 + c*dt^5
            
                # So: x(t+2dt) = x1 + 2c*dt^5
                
                t1_1 = t_array[i] + dt
                
                k1i_I_1 =  dt * self.f(r_array[:,i],t_array[i])
            
                k2i_I_1 =  dt * self.f(r_array[:,i]+0.5*k1i_I_1,t_array[i]+0.5*dt)
            
                k3i_I_1 =  dt * self.f(r_array[:,i]+0.5*k2i_I_1,t_array[i]+0.5*dt)
            
                k4i_I_1 =  dt * self.f(r_array[:,i]+k3i_I_1,t_array[i]+dt)  
                
                x1_1 =   r_array[:,i] + (1/6)*( k1i_I_1 + 2*k2i_I_1 + 2*k3i_I_1 + k4i_I_1 )
                
                
                
                t1_2 = t1_1 + dt
                
                k1i_I_2 =  dt * self.f(x1_1,t1_1)
                
                k2i_I_2 =  dt * self.f(x1_1+0.5*k1i_I_2,t1_1+0.5*dt)
                
                k3i_I_2 =  dt * self.f(x1_1+0.5*k2i_I_2,t1_1+0.5*dt)
                
                k4i_I_2 =  dt * self.f(x1_1+k3i_I_2,t1_1+dt)  
                
                x1_2 =   x1_1 + (1/6)*( k1i_I_2 + 2*k2i_I_2 + 2*k3i_I_2 + k4i_I_2 )
                
                t1 = t1_2
                
                x1 = x1_2
            
                #let's calculate the x2 using "2dt"
                # Here: x(t+2dt) = x2 + 32c*dt^5
            
                two_dt = 2*dt
            
            
            
                k1i_II =  two_dt * self.f(r_array[:,i],t_array[i])
            
                k2i_II =  two_dt * self.f(r_array[:,i]+0.5*k1i_II,t_array[i]+0.5*two_dt )
            
                k3i_II =  two_dt * self.f(r_array[:,i]+0.5*k2i_II,t_array[i]+0.5*two_dt )
            
                k4i_II =  two_dt * self.f(r_array[:,i]+k3i_II,t_array[i]+two_dt)
            
                x2  =   r_array[:,i] + (1/6)*( k1i_II + 2*k2i_II + 2*k3i_II + k4i_II )
            
            
                # Now we have the x1\x2, now let's calculate the rho
                
                print("norm(x2-x1) is %s" %np.linalg.norm(x2-x1) )
                
                rho = 30*dt*delta/np.linalg.norm(x2-x1)
                
            
                if rho>=1:
                    
                    t_array[i+1]   = t1
                    
                    r_array[:,i+1] = x1
                    
                    print("Now two_dt is %s" %two_dt)
                    
                    print("t1 is %s" %t1)
                    print("x1 is %s" %x1)
                    break
                
                if rho<1:
                    
                    print("dt = %s is too large" %dt)
                
                    dt = dt*np.sqrt(np.sqrt(rho))
                    
                    print("Now dt = %s " %dt)
                    
#            print("t_array is %s" %t_array)
#            print("r_array is %s" %r_array)        
            
            
        self.t_array = t_array
        self.r_array = r_array
        
        print("The consequence of RK2 is stored in self.t_array and self.r_array")
        print("----------------------------------------------")
        
        
    ######################################################################################
    
    ######################################################################################
            
    # BackEuler
    
    
    def evolve_BackEuler(self):
        
        
        # Here dt0 = self.dt0 (dt0 is local variable)
        
        dt0 = self.dt0
        
        # Built t_array to store time during evolution
        t_array = np.zeros(1+self.Nt0)
        
        
        # Built r_array to store r during evolution r = (x_1,x_2,...)
        # The dimension of r_array
        r_dim = self.dim_r
        r_array = np.zeros( [ r_dim , 1+self.Nt0 ], float )
        
        # Put the intial condition r0 into r_array
        r_array[:,0] = self.r0
        
        # Put the initial condition T0 to t_array
        t_array[0] = self.T0
        
        # The time evolution is 
        # t_0(r0)(For loop begins) --> t_1 --> t_2 -->...-->t_Nt0
        
        # i = 0,1,2,3,..., (Nt0-1)
        
        
        # Here we use the Backward Euler method 
        #  r_(i+1) = r_i + dt0 * f(r_(i+1),t_(i+1)) 
        
        # Then we need to solve 
        
        for i in range(0,self.Nt0):
            
            if i%100==0:
                
                print("It is %s th step" %i)
            
            t_array[i+1] = t_array[i] + dt0
            
            #r_array[:,i+1] = r_array[:,i] + dt0 * self.f(r_array[:,i],t_array[i])      
        
            def Equation_LHS(x):
                
                LHS = x - dt0*F(x,t_array[i+1]) - r_array[:,i]
                
                return LHS
        
            sol = opt.root( Equation_LHS , np.ones(r_dim) )
            
            r_array[:,i+1] = sol.x
        
        self.t_array = t_array
        self.r_array = r_array
        
        print("The consequence of BackEuler is stored in self.t_array and self.r_array")
        print("----------------------------------------------")
        
        
    
    #######################################################################################
    
    
    
    #######################################################################################
    # Crank-Nicolson
    
    
    
    def evolve_CrankNicolson(self):
        
        
        # Here dt0 = self.dt0 (dt0 is local variable)
        
        dt0 = self.dt0
        
        # Built t_array to store time during evolution
        t_array = np.zeros(1+self.Nt0)
        
        
        # Built r_array to store r during evolution r = (x_1,x_2,...)
        # The dimension of r_array
        r_dim = self.dim_r
        r_array = np.zeros( [ r_dim , 1+self.Nt0 ], float )
        
        # Put the intial condition r0 into r_array
        r_array[:,0] = self.r0
        
        # Put the initial condition T0 to t_array
        t_array[0] = self.T0
        
        # The time evolution is 
        # t_0(r0)(For loop begins) --> t_1 --> t_2 -->...-->t_Nt0
        
        # i = 0,1,2,3,..., (Nt0-1)
        
        
        # Here we use the Backward Euler method 
        #  r_(i+1) = r_i + dt0 * f(r_(i+1),t_(i+1)) 
        
        # Then we need to solve 
        
        for i in range(0,self.Nt0):

            if i%100==0:
                
                print("It is %s th step" %i)
            
            
            t_array[i+1] = t_array[i] + dt0
            
            #r_array[:,i+1] = r_array[:,i] + dt0 * self.f(r_array[:,i],t_array[i])      
        
            def Equation_LHS(x):
                
                LHS = x - (dt0/2)*F(x,t_array[i+1]) - ( r_array[:,i] + (dt0/2)*F(r_array[:,i],t_array[i])  )
                
                return LHS
        
            sol = opt.root( Equation_LHS , np.ones(r_dim) )
            
            r_array[:,i+1] = sol.x
        
        self.t_array = t_array
        self.r_array = r_array
        
        print("The consequence of Crank-Nicolson is stored in self.t_array and self.r_array")
        print("----------------------------------------------")
        
                
            

    #######################################################################################
    

               
        
t2 = time.time()

print("The time of initialization--II is %s" %(t2 - t1) )
print("")
print("")

# ================================================================================================
# ================================================================================================
# ================================================================================================
# ================================================================================================
# ================================================================================================




#---------------------------------------------------------------
# Define f_function(r)    or    f(r) in the literature
    
def f_function(r):
    
    f = (r**2+r+2)**2/( np.sqrt(2) * (r-2)**2 * r**(9/2) * (r+1)**(7/2) )

    return f


# Define the number of time_step for the ode_solver

time_step = 500



# define the parameter to scan 

parameter_ratio = [ 0.05, 0.1, 0.2 , 0.3 , 0.5 ,  0.7 ,  0.9  ,1]

#parameter_ratio = [0.000000001,0.0000001, 0.00001, 0.001, 0.1, 0.2 , 0.3 , 0.5 ,  0.7 ,  0.9 , 0.95 , 0.99 ,0.999 ,1]

#parameter_ratio = [ 0.01, 0.05 , 0.07 , 0.1, 0.15 , 0.2 , 0.3 , 0.5 ,  0.7 ,  0.9 , 0.95 , 0.99 ,0.999 ,1]


# Calculate the length of parameter_ratio 

len_parameter_ratio = len(parameter_ratio )

print("******************************************************")
print("The parameter_ratio we want to scan is %s" %parameter_ratio)
print("The length of parameter ratio we want to scan is %s "  %len_parameter_ratio)
print("******************************************************")
print("")

# Now initialize the array to store the consequence 
#      We should know that the whole grids number of x_array & Y_array is "1+time_step"
#              x_array[0,:]            &      Y_array[0,:]
#              x_array[1,:]            &      Y_array[1,:]
#              ......
#              x_array[time_step,:]    &      Y_array[time_step,:]


x_array = np.zeros([ len_parameter_ratio ,  1 + time_step  ])
Y_array = np.zeros([ len_parameter_ratio ,  1 + time_step  ])

# We also build up the array to store the final_Y

final_Y_array = np.zeros([len_parameter_ratio ])


for I in range( 0 , len_parameter_ratio ):
    
    
    ratio = parameter_ratio[ I ]
    #ratio=0.5
    
    print(  "In %s th step, the ratio is %s" %(I,ratio)  )

    x=1

    DEL=0.1
    y=0.5


    # Define the mass m_chi of dark matter particle chi 
    # m_chi = 0.001Gev = 1MeV

    m_chi = 10**(-3)


    # Mpl=10**19 GeV

    Mpl= 4.86 * 10**(18)


    Cutoff_k=700

    A=10

    B=1


    delta_mixing=10**(-5)

    #Here Mpl_mchi_A13 = (Mpl/m_chi)*(1/A)

    #Mpl_mchi_A = (Mpl/m_chi)* (delta_mixing**2)*(1/A)

    Mpl_mchi_deltasquare_A =(Mpl/m_chi)* (delta_mixing**2) * (1/A) 

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Mpl                    = %s" %Mpl )
    print("m_chi                  = %s" %m_chi )
    print("delta_mixing           = %s" %delta_mixing)
    print("")
    print("Mpl_mchi_deltasquare_A = %s" %Mpl_mchi_deltasquare_A )
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("")







    # Now let's define the function of evolution 
    # in ODE 
    def F(r,t):
    
        #F = np.sin(3*t)+np.sin(2*t)+np.cos(t)
    
    
        Term1 =    (-1) * ( (45**(3/2))/( 2**(7/2)*(np.pi**8) )  )* ( (1+DEL)**(3/2) * (DEL)**(1/2) ) * f_function(ratio) * y**4 * Mpl_mchi_deltasquare_A   
    

        Term2 =    ( 1/(g_star(m_chi/(A*t)))**(3/2)  )  * ( (1/t**2) * np.exp(-DEL*A*t) * Ib_eq(A*t,ratio) * Ib_eq(A*t,1+DEL)   )
    
    
        YY_Ratio = ( r[0] )/(  B*Yb_eq(A*t,1) )
        #YY_Ratio = ( r[0]*np.exp(A*t) )/(  B*Yb_eq(A*t,1)*np.exp(A*t)  )
    
    
        Term3 =      YY_Ratio - 1

    
        F =  Term1*Term2*Term3
    
        return np.array([F])

    t0=0.1


    # Initial condition is 
    R0 = [B*Yb_eq(A*t0,1)]
        
    #  For IODE
    #  The initialized 2condition is:
    #  __init__(self, N_t_max0, T_i , T_f, function, r0 )

    #      We should note that since A=10, so when we choose T_f = 30, 
    #                                    in fact the largest x = 300
    solution1 = IODE(time_step,t0,30,F, R0)
    #solution1.evolve_CrankNicolson()
    #solution1.evolve_RK4()
    solution1.evolve_BackEuler()

    #solution1.plot()
    
    # Here we output the consequence from the class
    #     In I th interation:
    #              x_Y_array[0,:]  ==>  x_array[I,:]
    #              x_Y_array[1,:]  ==>  Y_array[I,:]    
    #              final_Y         ==>  final_Y_array[I]
    
    x_Y_array=solution1.output_t_r_array()
    final_Y = solution1.final_Y()
    
    
    #........................................
    # Now let's store the consequence   x_array // Y_array // final_Y_array
    
    x_array[I,:]     =  x_Y_array[0,:]
    Y_array[I,:]     =  x_Y_array[1,:]
    final_Y_array[I] =  final_Y
    
    #........................................

    #print("t_r_array is %s" %t_r_array)
    print("final Y is %s" %final_Y)
    print("")
    print("#################################################################################")
    print("#################################################################################")
    print("")
    

    

        


# ================================================================================================
# ================================================================================================
# ================================================================================================
# ================================================================================================
# ================================================================================================

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Now let's do the final plot  and  final analysis

fig = plt.figure(1)



#***********************************************************************
# Let's plot the freeze out diagram

ax1 = plt.subplot(111)




#===============================================================
# This is equilibrium state
Y_EQ = [ ]
        
for x in  A*x_array[0,:]:
            
    Y = Yb_eq(x,1)
            
    Y_EQ.append(Y)
    
ax1.plot(  A*x_array[0,:], Y_EQ/Y_array[0,0],marker='', ls='-',alpha=0.5, color="red", lw=5, label=r'$Y^{(eq)}(x)$')
 
#===============================================================



for I in range(0,len_parameter_ratio):
    
    ax1.plot( A*x_array[I,:] , Y_array[I,:]/Y_array[I,0] ,  color =( ( 0.9/len_parameter_ratio)*I, ( 0.5/len_parameter_ratio)*I , 0.9)  , ls='-',lw=( -2/len_parameter_ratio)*I+3  , label =' r = %s' %parameter_ratio[ I ] )
    #ax1.plot( A*x_array[I,:] , Y_array[I,:]/Y_array[I,0] ,  color ="red"  , ls='-',lw=3  , label =' r = %s' %2 )

    
    
ax1.legend( loc='best',frameon=False )
ax1.set_xlabel( r'$x = m_{\chi}/T$ ', fontsize=15 )
ax1.set_ylabel( r'$Y_{\chi}(x)/Y_{\chi}(x=1) $', fontsize=15 )
ax1.grid()

plt.title(r' The figure of DM freezing out when $r=m_{\phi}/m_{\chi}$ varies ', fontsize =12)

ax1.set_xscale('log')
ax1.set_yscale('log')


#====================================
# Let's set up the ticks

x = [1, 5 , 10, 20, 30, 40,50,70 , 100 ,150]

labels = ['Begin:1','5' , '10' , '20', '30', '40' , '50', '70' ,'100' ,'150']


# You can specify a rotation for the tick labels in degrees or with keywords.
plt.xticks(x, labels, rotation=0)


# This is to annotate some informations on the diagram
ax1.annotate(r'$\delta = 10^{-5}$',      xy=(3, 2), xytext=(10, 10**(-50)), fontsize = 12)
ax1.annotate(r'$m_{\chi} = 10^{-3}GeV$', xy=(3, 2), xytext=(10, 10**(-60)), fontsize = 12)
ax1.annotate(r'$\Delta = 0.1 $', xy=(3, 2), xytext=(10, 10**(-70)), fontsize = 12)
ax1.annotate(r'$f(r) = \frac{(r^2+r+2)^2}{\sqrt{2} (r-2)^2 r^{9/2} (r+1)^{7/2}  } $', xy=(3, 2), xytext=(5, 10**(-100)), fontsize = 12)
ax1.annotate(r'$\langle \sigma_{\chi \rightarrow \psi } v \rangle (x) \approx ( \frac{m_{\psi}}{m_{\chi}} )^{\frac{3}{2}} e^{-\Delta x} f(r) \sqrt{\Delta} \frac{y^4 \delta^2}{2 \pi m_{\chi}^2} $', xy=(3, 2), xytext=(5, 10**(-120)), fontsize = 12)




#***********************************************************************
# Let's plot the diagram of ratio and final_Y
fig = plt.figure(2)
ax2 = plt.subplot(111)


plt.title(r' How $Y_{final}$ changes as $r=m_{\phi}/m_{\chi}$ varies ', fontsize =12)


ax2.plot(  parameter_ratio, final_Y_array ,marker='', ls='--',alpha=0.9, color="green", lw=5, label=r'$Y_{Freeze\quad out}$')




plt.text(0.5, 10**(-50), r'$\delta = 10^{-5}$', fontsize=12)
plt.text(0.5, 10**(-60), r'$m_{\chi} = 10^{-3}GeV$', fontsize=12)
plt.text(0.5, 10**(-70), r'$\Delta = 0.1 $', fontsize=12)


plt.text(0.3, 10**(-90), r'$f(r) = \frac{(r^2+r+2)^2}{\sqrt{2} (r-2)^2 r^{9/2} (r+1)^{7/2}  } $', fontsize=12)
plt.text(0.3, 10**(-105), r'$\langle \sigma_{\chi \rightarrow \psi } v \rangle (x) \approx ( \frac{m_{\psi}}{m_{\chi}} )^{\frac{3}{2}} e^{-\Delta x} f(r) \sqrt{\Delta} \frac{y^4 \delta^2}{2 \pi m_{\chi}^2} $', fontsize=12)



ax2.set_yscale('log')

ax2.grid()
ax2.legend( loc='best',frameon=True )
ax2.set_xlabel( r'$r = m_{\phi}/m_{\chi}$ ', fontsize=15 )
ax2.set_ylabel( r'$Y_{Freeze\quad out}$', fontsize=15 )
ax2.set_xlim(-0.05,1.05)

plt.show()


#***********************************************************************

