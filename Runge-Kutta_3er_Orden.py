import numpy as np;
import matplotlib.pyplot as plt;

### Función del método de Runge-Kutta de 3er orden
def metodo_RungeKutta_3er(x,y0,h):
    y1    = np.zeros(len(x))    
    y1[0] = y0    
    for i in range(1,len(x),1):        
        k1 = fx_edo(x[i-1],y1[i-1])        
        k2 = fx_edo(x[i-1]+0.5*h,y1[i-1]+0.5*k1*h)        
        k3 = fx_edo(x[i-1]+h,y1[i-1]-k1*h+2*k2*h)        
        y1[i] = y1[i-1] + (1.0/6.0)*(k1 + 4.0*k2 + k3)*h

    #End for    
    print("Método de Runge-Kutta de tercer orden")    
    for i in range(0,len(y1),1):        
        print("x:",x[i],",y[",i,"]:",y1[i])   
        #End return
    
    # Llamado de la solución real para comparar el método    
    y_r = fx_r(x)

    fig = plt.figure(figsize=(6,5))    
    plt.plot(x,y1,'r-',lw = 1.5,label='Aproximación: Método de RK 3er orden')    
    plt.plot(x,y_r,'b-',lw = 1.5,label='Solución real')    
    plt.legend(frameon=True,fontsize=14,loc=0,ncol=1)    
    plt.yticks(fontsize=10)    
    plt.xlabel("x",fontsize = 16, color = 'black')    
    plt.ylabel("y",fontsize = 16, color = 'black')    
    titulog = 'Fig_metodo_RK_3er_orden.png'    
    plt.title('Soluciones de 3er Orden \n',fontsize=10,fontweight='bold')    
    plt.grid(True)    
    plt.grid(color = '0.5', linestyle = '--', linewidth = 1)    
    plt.xticks(rotation=0,fontweight='bold')    
    plt.yticks(fontweight='bold')    
    plt.show()    
    plt.savefig(titulog, dpi = 600)    
    # fig.clear()    
    plt.close(fig)
#End function

""" Declaración de parámetros"""
### Condiciones iniciales
x0 = 0.0          # Valor de variable independiente inicial
y0 = 1.0          # Valor de variable dependiente inicial
### Valor final de variable independiente
xf = 4.0
### Intervalo o paso
h  = 0.4#
## Variables de x a evaluar
x = np.arange(x0,xf+h,h)

### Declaración de EDO
def fx_edo(x,y):    
    fx = -2.0*(x**3.0) + 12.0*(x**2.0) - 20.0*(x) + 8.5    
    return fx
#End function
# ### Si queremos poner la función real para comparar
def fx_r(x):    
    fx = -0.5*(x**4.0) + 4.0*(x**3.0) - 10.0*(x**2.0) + 8.5*x + 1.0    
    return fx
#End function
### Llamado del método
metodo_RungeKutta_3er(x,y0,h)