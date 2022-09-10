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

def test(x: int) -> 'np.array()':
   return np.array([x]*5)

if __name__ == '__main__':
   print('Functions to calculate nonradiative broadening')