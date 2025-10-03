import mpmath
mpmath.mp.dps = 30
zeros_im = [mpmath.zetazero(k).imag for k in range(1,11)]
print('First 10 RH Zeros Im:', [float(z) for z in zeros_im])
