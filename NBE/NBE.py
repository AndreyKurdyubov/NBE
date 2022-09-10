import numpy as np
from scipy.integrate import solve_ivp

def ex_(A, dec, t):
    return A*np.exp(-dec*t)   

def ivp(xdata, N0, k, gd, kex, kc, gnr, bgr):
    '''
    solve initial value problem 
    '''
    
    def scatter(t, y, N0, k, gd, kex, kc, gnr):
        nx_hot = ex_(N0*(1-k), gd, t)
        n, nx = y
        return [gd*nx_hot - kex*n**2 - gnr*n, 
                kex*n**2 - kc*2*n*nx]
    
    n0 = [0, bgr + N0*k]
    tspan = (0, xdata[-1])
    fun = lambda t, y: scatter(t, y, N0, k, gd, kex, kc, gnr)
    sol = solve_ivp(fun, tspan, n0, t_eval=xdata)
    
    return sol

def broad(N0, k, gd, kex, kc, gnr, sigm=3, 
          xlim=[-100, 5000], ylim=[-10, 50], 
          ndstart=52, ndskip=52, ndend=-1, temp='6.2K', gnr_range=None, kc_range=None):
    '''
    ndstart - first point for fit
    ndskip - point of zero delay
    ndend - last point
    '''
    X = Xd[temp]  # data for given pump power
    bgr = neg[temp]/sigm  # signal in negative delays
    
    Delay = X[ndstart:ndend, 0] - X[ndskip, 0]
    Delay_skip = X[0:ndstart, 0] - X[ndskip, 0]
    Delay_skip2 = X[ndend:, 0] - X[ndskip, 0]
    dskip = np.r_[Delay_skip, Delay_skip2]
    
    Gh = X[ndstart:ndend, 1]
    G_skip = X[0:ndstart, 1]
    G_skip2 = X[ndend:, 1]
    gskip = np.r_[G_skip, G_skip2]
    
    time = np.linspace(0, Delay[-1], 1000)
    sol = ivp(time, N0, k, gd, kex, kc, gnr, bgr) # solve initial value problem
    
    t = sol.t  # == time
    y = sol.y
    
    nx_hot = ex_(N0*(1-k), gd, t)

    font = {'family': 'serif',
        'weight': 'normal',
        'size': 16,
        }
    
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    
    plt.plot(Delay, Gh, '.', label='Data', color='skyblue')
    plt.plot(Delay-13000, Gh, '.', color='skyblue')
    plt.plot(dskip, gskip, '.', color='peru', label='Data_skip')
    plt.plot(t, sigm*(y[0, :]), '-*r', linewidth=2, label='e,h')
    plt.plot(t, sigm*(y[1, :]), '--', color='#808800', linewidth=2, label='dark X')
    plt.plot(t, sigm*(2*y[0, :] + y[1, :] + nx_hot), '-k', linewidth=2, label='broadening')
    plt.plot(t, sigm*nx_hot, '--', color='#800000', linewidth=2, label='hot-X')
    
    plt.plot(t-13000, sigm*(y[0, :]), '-*r', linewidth=2)
    plt.plot(t-13000, sigm*(y[1, :]), '--', color='#808800', linewidth=2)
    plt.plot(t-13000, sigm*(2*y[0, :] + y[1, :] + nx_hot), '-k', linewidth=2)
    plt.plot(t-13000, sigm*nx_hot, '--', color='#800000', linewidth=2)

    
    plt.text((xlim[1] + xlim[0])*0.4,
             (ylim[1] + ylim[0])*0.9, 
             f'$\sigma$ = {sigm}', fontsize=16)
    
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.locator_params(axis='x', nbins=6)
    plt.xlabel('Delay, ps', fontdict=font)
    plt.ylabel(r'$\Delta\hbar\Gamma$, $\mu$eV', fontdict=font)
    plt.legend(loc=1, fontsize=12)
    
    return sol, fig1
    
if __name__ == '__main__':
   print('Functions to calculate nonradiative broadening')