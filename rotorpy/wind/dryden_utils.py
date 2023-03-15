"""
Utility scripts for the Dryden wind model
The Dryden Gust model is implemented using this package: 
https://github.com/goromal/wind-dynamics

"""
import numpy as np

class GustModelBase():
    """
    """

    def __init__(self, V, L, sigma, dt=0.05):
        """
        Inputs: 
            V, average velocity through medium, m/s
            L, characteristic length scale, m
            sigma, turbulence intensity
        """

        self.dt = dt
        b = 2*np.sqrt(3)*L/V
        c = 2*L/V

        self.alpha = sigma*np.sqrt(2*L/np.pi/V)
        self.beta = self.alpha*b
        self.delta = 2*c
        self.gamma = c*c

        self.u_km1 = 0
        self.u_km2 = 0
        self.y_km1 = 0
        self.y_km2 = 0

        self.initialized = True
        return 
    
    def run(self, dt):
        """
        """

        C1 = 1.0 + 2*self.delta/dt + 4*self.gamma/dt/dt
        C2 = 2.0 - 8*self.gamma/dt/dt
        C3 = 1.0 - 2*self.delta/dt + 4*self.gamma/dt/dt
        C4 = self.alpha + 2*self.beta/dt
        C5 = 2*self.alpha
        C6 = self.alpha - 2*self.beta/dt

        u_k = np.random.uniform(-1,1)
        y_k = (C4*u_k + C5*self.u_km1 + C6*self.u_km2 - C2*self.y_km1 - C3*self.y_km2)/C1

        self.u_km2 = self.u_km1
        self.u_km1 = u_k
        self.y_km2 = self.y_km1
        self.y_km1 = y_k

        return y_k
    
    def integrate(self, dt):
        """
        """

        if (dt > self.dt):
            t = 0
            y = 0
            while (t < dt):
                t_inc = min(self.dt, dt - t)
                y = self.run(t_inc)
                t += t_inc
            return y
        else:
            return self.run(dt)
        
class DrydenWind():
    """
    """

    def __init__(self, wx_nominal, wy_nominal, wz_nominal, 
                       wx_sigma, wy_sigma, wz_sigma, 
                       altitude=2.0):
        
        self.wx_nominal, self.wy_nominal, self.wz_nominal = wx_nominal, wy_nominal, wz_nominal
        self.altitude = altitude

        Lz_ft = 3.281 * altitude
        Lx_ft = Lz_ft / ((0.177 + 0.000823 * Lz_ft)**(1.2))
        Ly_ft = Lx_ft

        self.wx_gust = GustModelBase(1.0, Lx_ft/3.281, wx_sigma)
        self.wy_gust = GustModelBase(1.0, Ly_ft/3.281, wy_sigma)
        self.wz_gust = GustModelBase(1.0, Lz_ft/3.281, wz_sigma)

        self.initialized = True

        return
    
    def getWind(self, dt):
        """
        """

        if self.initialized:
            wind_vector = np.array([self.wx_nominal, self.wy_nominal, self.wz_nominal]) + np.array([self.wx_gust.integrate(dt), self.wy_gust.integrate(dt), self.wz_gust.integrate(dt)])
        else:
            wind_vector = np.array([0,0,0])

        return wind_vector
    

if __name__=="__main__":

    import matplotlib.pyplot as plt
    import numpy as np

    t = 100.0
    dt = 0.05
    x_mean = 0.0
    y_mean = 1.0
    z_mean = 0.0
    x_sigma = 10.0
    y_sigma = 0.0
    z_sigma = 70.0

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })

    wind = DrydenWind(x_mean, y_mean, z_mean, x_sigma, y_sigma, z_sigma)

    n = int(np.floor(t / dt)) + 1
    x = np.linspace(0, t, n)
    y = np.zeros((3,x.size))

    for i in range(n):
        y[:,i] = wind.getWind(dt)

    fig, axs = plt.subplots(3)
    fig.suptitle('Dryden Gust Velocities\n$\sigma_x=%f$, $\sigma_y=%f$, $\sigma_z=%f$' % (x_sigma, y_sigma, z_sigma))
    axs[0].plot([0, t],[x_mean, x_mean])
    axs[0].plot(x, y[0,:])
    axs[0].set_ylabel('X Velocity')
    axs[1].plot([0, t],[y_mean, y_mean])
    axs[1].plot(x, y[1,:])
    axs[1].set_ylabel('Y Velocity')
    axs[2].plot([0, t],[z_mean, z_mean])
    axs[2].plot(x, y[2,:])
    axs[2].set_ylabel('Z Velocity')
    axs[2].set_xlabel('Time (s)')

    plt.show()