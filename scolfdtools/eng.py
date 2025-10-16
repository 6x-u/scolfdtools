import math

PI=math.pi
E=math.e
TAU=2*PI
PHI=(1+math.sqrt(5))/2

def sin(x):return math.sin(x)
def cos(x):return math.cos(x)
def tan(x):return math.tan(x)
def asin(x):return math.asin(x)
def acos(x):return math.acos(x)
def atan(x):return math.atan(x)
def sinh(x):return math.sinh(x)
def cosh(x):return math.cosh(x)
def tanh(x):return math.tanh(x)
def asinh(x):return math.asinh(x)
def acosh(x):return math.acosh(x)
def atanh(x):return math.atanh(x)

def sqrt(x):return math.sqrt(x)
def cbrt(x):return x**(1/3)
def pow(x,y):return x**y
def exp(x):return math.exp(x)
def log(x,b=E):return math.log(x,b)
def ln(x):return math.log(x)
def log10(x):return math.log10(x)
def log2(x):return math.log2(x)

def abs(x):return math.fabs(x)
def ceil(x):return math.ceil(x)
def floor(x):return math.floor(x)
def round(x,n=0):return __builtins__.round(x,n)
def trunc(x):return math.trunc(x)

def deg(x):return math.degrees(x)
def rad(x):return math.radians(x)

def hypot(*args):return math.sqrt(sum(x**2 for x in args))
def dist(p1,p2):return math.sqrt(sum((a-b)**2 for a,b in zip(p1,p2)))

def factorial(n):return 1 if n<=1 else n*factorial(n-1)
def gcd(*args):
    if len(args)==2:return math.gcd(*args)
    r=args[0]
    for a in args[1:]:r=math.gcd(r,a)
    return r

def lcm(*args):
    if len(args)==2:return abs(args[0]*args[1])//math.gcd(args[0],args[1])
    r=args[0]
    for a in args[1:]:r=abs(r*a)//math.gcd(r,a)
    return r

def fib(n):
    if n<=1:return n
    a,b=0,1
    for _ in range(n-1):a,b=b,a+b
    return b

def isprime(n):
    if n<2:return False
    if n==2:return True
    if n%2==0:return False
    for i in range(3,int(sqrt(n))+1,2):
        if n%i==0:return False
    return True

def nextprime(n):
    n+=1
    while not isprime(n):n+=1
    return n

def primes(n):return[p for p in range(2,n)if isprime(p)]

def factors(n):
    f=[]
    for i in range(1,int(sqrt(n))+1):
        if n%i==0:
            f.append(i)
            if i!=n//i:f.append(n//i)
    return sorted(f)

def primefactors(n):
    f=[]
    d=2
    while d*d<=n:
        while n%d==0:
            f.append(d)
            n//=d
        d+=1
    if n>1:f.append(n)
    return f

def totient(n):return sum(1 for i in range(1,n+1)if gcd(i,n)==1)

def isperfect(n):return sum(factors(n)[:-1])==n

def isabundant(n):return sum(factors(n)[:-1])>n

def isdeficient(n):return sum(factors(n)[:-1])<n

def triangular(n):return n*(n+1)//2

def square(n):return n*n

def pentagonal(n):return n*(3*n-1)//2

def hexagonal(n):return n*(2*n-1)

def heptagonal(n):return n*(5*n-3)//2

def octagonal(n):return n*(3*n-2)

def catalan(n):
    if n<=1:return 1
    c=0
    for i in range(n):c+=catalan(i)*catalan(n-1-i)
    return c

def bell(n):
    if n==0:return 1
    b=[[0]*n for _ in range(n)]
    b[0][0]=1
    for i in range(1,n):
        b[i][0]=b[i-1][i-1]
        for j in range(1,i+1):b[i][j]=b[i-1][j-1]+b[i][j-1]
    return b[n-1][0]

def stirling1(n,k):
    if n==0 and k==0:return 1
    if n==0 or k==0:return 0
    return (n-1)*stirling1(n-1,k)+stirling1(n-1,k-1)

def stirling2(n,k):
    if n==0 and k==0:return 1
    if n==0 or k==0:return 0
    return k*stirling2(n-1,k)+stirling2(n-1,k-1)

def perm(n,r):return factorial(n)//factorial(n-r)if n>=r else 0

def comb(n,r):return factorial(n)//(factorial(r)*factorial(n-r))if n>=r else 0

def binomial(n,k):return comb(n,k)

def multinomial(*args):
    n=sum(args)
    result=factorial(n)
    for a in args:result//=factorial(a)
    return result

def derangement(n):
    if n==0:return 1
    if n==1:return 0
    return (n-1)*(derangement(n-1)+derangement(n-2))

def partition(n):
    p=[0]*(n+1)
    p[0]=1
    for i in range(1,n+1):
        for j in range(i,n+1):p[j]+=p[j-i]
    return p[n]

def harmonic(n):return sum(1/i for i in range(1,n+1))

def bernoulli(n):
    b=[0]*(n+1)
    for m in range(n+1):
        b[m]=1/(m+1)
        for j in range(m,0,-1):b[j-1]=j*(b[j-1]-b[j])
    return b[n]

def euler(n):
    if n==0:return 1
    e=[0]*(n+1)
    e[0]=1
    for i in range(1,n+1):
        for j in range(i):e[i]+=comb(i,j)*e[j]*e[i-j-1]
        e[i]*=-1 if i%2 else 1
    return e[n]

def moebius(n):
    if n==1:return 1
    pf=primefactors(n)
    if len(pf)!=len(set(pf)):return 0
    return (-1)**len(pf)

def divisorsum(n):return sum(factors(n))

def divisorcount(n):return len(factors(n))

def iscoprime(a,b):return gcd(a,b)==1

def modpow(b,e,m):return pow(b,e,m)

def modinv(a,m):
    if gcd(a,m)!=1:return None
    u1,u2,u3=1,0,a
    v1,v2,v3=0,1,m
    while v3!=0:
        q=u3//v3
        v1,v2,v3,u1,u2,u3=(u1-q*v1),(u2-q*v2),(u3-q*v3),v1,v2,v3
    return u1%m

def crt(r,m):
    M=1
    for mi in m:M*=mi
    x=0
    for ri,mi in zip(r,m):
        Mi=M//mi
        x+=ri*Mi*modinv(Mi,mi)
    return x%M

def legendre(a,p):return pow(a,(p-1)//2,p)

def jacobi(a,n):
    if gcd(a,n)!=1:return 0
    j=1
    while a!=0:
        while a%2==0:
            a//=2
            if n%8 in[3,5]:j=-j
        a,n=n,a
        if a%4==3 and n%4==3:j=-j
        a%=n
    return j if n==1 else 0

def tonelli(n,p):
    if legendre(n,p)!=1:return None
    q=p-1
    s=0
    while q%2==0:q//=2;s+=1
    if s==1:return pow(n,(p+1)//4,p)
    z=2
    while legendre(z,p)!=-1:z+=1
    c=pow(z,q,p)
    r=pow(n,(q+1)//2,p)
    t=pow(n,q,p)
    m=s
    while t%p!=1:
        i=1
        while pow(t,2**i,p)!=1:i+=1
        b=pow(c,2**(m-i-1),p)
        r=(r*b)%p
        c=(b*b)%p
        t=(t*c)%p
        m=i
    return r

def pollard(n):
    if isprime(n):return n
    x=2
    y=2
    d=1
    f=lambda x:(x*x+1)%n
    while d==1:
        x=f(x)
        y=f(f(y))
        d=gcd(abs(x-y),n)
    return d if d!=n else n

def miller(n,k=10):
    if n<2:return False
    if n==2 or n==3:return True
    if n%2==0:return False
    r,d=0,n-1
    while d%2==0:r+=1;d//=2
    import random
    for _ in range(k):
        a=random.randrange(2,n-1)
        x=pow(a,d,n)
        if x==1 or x==n-1:continue
        for _ in range(r-1):
            x=pow(x,2,n)
            if x==n-1:break
        else:return False
    return True

def carmichael(n):
    if n==1:return 1
    pf=list(set(primefactors(n)))
    l=1
    for p in pf:
        k=primefactors(n).count(p)
        pk=p**k
        if p==2 and k>2:l=lcm(l,pk//2)
        else:l=lcm(l,pk-pk//p)
    return l

def ordinal(n,p):
    if gcd(n,p)!=1:return None
    o=1
    while pow(n,o,p)!=1:o+=1
    return o

def primitive(p):
    if not isprime(p):return None
    phi=p-1
    pf=list(set(primefactors(phi)))
    for g in range(2,p):
        if all(pow(g,phi//pf,p)!=1 for pf in pf):return g
    return None

def quadratic(a,b,c):
    d=b*b-4*a*c
    if d<0:return None
    if d==0:return(-b/(2*a),)
    return((-b+sqrt(d))/(2*a),(-b-sqrt(d))/(2*a))

def cubic(a,b,c,d):
    if a==0:return quadratic(b,c,d)
    b/=a;c/=a;d/=a
    q=(3*c-b*b)/9
    r=(9*b*c-27*d-2*b*b*b)/54
    disc=q*q*q+r*r
    if disc>0:
        s=(r+sqrt(disc))**(1/3)
        t=(r-sqrt(disc))**(1/3)
        return(s+t-b/3,)
    elif disc==0:
        r13=r**(1/3)
        return(-b/3+2*r13,-2*r13-b/3)
    else:
        q=-q
        dum1=q*q*q
        dum1=acos(r/sqrt(dum1))
        r13=2*sqrt(q)
        return(-b/3+r13*cos(dum1/3),-b/3+r13*cos((dum1+2*PI)/3),-b/3+r13*cos((dum1+4*PI)/3))

def lagrange(points):
    n=len(points)
    def L(x):
        result=0
        for i in range(n):
            xi,yi=points[i]
            term=yi
            for j in range(n):
                if i!=j:
                    xj=points[j][0]
                    term*=(x-xj)/(xi-xj)
            result+=term
        return result
    return L

def newton(points):
    n=len(points)
    x=[p[0]for p in points]
    y=[p[1]for p in points]
    def divided_diff(i,j):
        if j==0:return y[i]
        return(divided_diff(i+1,j-1)-divided_diff(i,j-1))/(x[i+j]-x[i])
    def P(t):
        result=divided_diff(0,0)
        prod=1
        for i in range(1,n):
            prod*=(t-x[i-1])
            result+=divided_diff(0,i)*prod
        return result
    return P

def hermite(points):
    n=len(points)
    x=[p[0]for p in points]
    y=[p[1]for p in points]
    yp=[p[2]for p in points]
    def h(i,t):
        L=1
        Lder=0
        for j in range(n):
            if i!=j:
                L*=(t-x[j])/(x[i]-x[j])
                Lder+=1/(x[i]-x[j])
        return(1-2*(t-x[i])*Lder)*L*L
    def H(i,t):
        L=1
        for j in range(n):
            if i!=j:L*=(t-x[j])/(x[i]-x[j])
        return(t-x[i])*L*L
    def P(t):
        result=0
        for i in range(n):result+=y[i]*h(i,t)+yp[i]*H(i,t)
        return result
    return P

def spline(points):
    n=len(points)
    h=[points[i+1][0]-points[i][0]for i in range(n-1)]
    alpha=[3*(points[i+1][1]-points[i][1])/h[i]-3*(points[i][1]-points[i-1][1])/h[i-1]for i in range(1,n-1)]
    l=[1]+[0]*(n-1)
    mu=[0]*n
    z=[0]*n
    for i in range(1,n-1):
        l[i]=2*(points[i+1][0]-points[i-1][0])-h[i-1]*mu[i-1]
        mu[i]=h[i]/l[i]
        z[i]=(alpha[i-1]-h[i-1]*z[i-1])/l[i]
    b=[0]*n
    c=[0]*n
    d=[0]*n
    for j in range(n-2,-1,-1):
        c[j]=z[j]-mu[j]*c[j+1]
        b[j]=(points[j+1][1]-points[j][1])/h[j]-h[j]*(c[j+1]+2*c[j])/3
        d[j]=(c[j+1]-c[j])/(3*h[j])
    def S(x):
        for i in range(n-1):
            if points[i][0]<=x<=points[i+1][0]:
                dx=x-points[i][0]
                return points[i][1]+b[i]*dx+c[i]*dx*dx+d[i]*dx*dx*dx
        return None
    return S

def bezier(points,t):
    n=len(points)-1
    result=[0,0]
    for i,p in enumerate(points):
        b=comb(n,i)*(1-t)**(n-i)*t**i
        result[0]+=b*p[0]
        result[1]+=b*p[1]
    return tuple(result)

def bspline(points,degree,t):
    n=len(points)
    def N(i,p,u):
        if p==0:return 1 if i<=u<i+1 else 0
        return((u-i)/(p))*N(i,p-1,u)+(((i+p+1)-u)/(p))*N(i+1,p-1,u)
    result=[0,0]
    for i in range(n):
        basis=N(i,degree,t)
        result[0]+=basis*points[i][0]
        result[1]+=basis*points[i][1]
    return tuple(result)

def catmull(points,t):
    p0,p1,p2,p3=points
    return(0.5*((2*p1[0])+(-p0[0]+p2[0])*t+(2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t*t+(-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t*t*t),
           0.5*((2*p1[1])+(-p0[1]+p2[1])*t+(2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t*t+(-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t*t*t))

def hermitespline(points,t):
    p0,p1,m0,m1=points
    h00=2*t**3-3*t**2+1
    h10=t**3-2*t**2+t
    h01=-2*t**3+3*t**2
    h11=t**3-t**2
    return(h00*p0[0]+h10*m0[0]+h01*p1[0]+h11*m1[0],
           h00*p0[1]+h10*m0[1]+h01*p1[1]+h11*m1[1])

def derivative(f,x,h=1e-10):return(f(x+h)-f(x-h))/(2*h)

def gradient(f,point,h=1e-10):return[derivative(lambda x:f(point[:i]+[x]+point[i+1:]),point[i],h)for i in range(len(point))]

def integral(f,a,b,n=1000):
    h=(b-a)/n
    return h*(0.5*(f(a)+f(b))+sum(f(a+i*h)for i in range(1,n)))

def simpson(f,a,b,n=1000):
    if n%2==1:n+=1
    h=(b-a)/n
    return(h/3)*(f(a)+f(b)+4*sum(f(a+(2*i-1)*h)for i in range(1,n//2+1))+2*sum(f(a+2*i*h)for i in range(1,n//2)))

def romberg(f,a,b,n=6):
    R=[[0]*n for _ in range(n)]
    h=b-a
    R[0][0]=h*(f(a)+f(b))/2
    for i in range(1,n):
        h/=2
        sum_term=sum(f(a+(2*k-1)*h)for k in range(1,2**(i-1)+1))
        R[i][0]=R[i-1][0]/2+h*sum_term
        for j in range(1,i+1):R[i][j]=(4**j*R[i][j-1]-R[i-1][j-1])/(4**j-1)
    return R[n-1][n-1]

def monte_carlo(f,a,b,n=10000):
    import random
    total=0
    for _ in range(n):total+=f(random.uniform(a,b))
    return(b-a)*total/n

def newton_method(f,df,x0,tol=1e-10,max_iter=100):
    x=x0
    for _ in range(max_iter):
        fx=f(x)
        if abs(fx)<tol:return x
        x=x-fx/df(x)
    return x

def secant(f,x0,x1,tol=1e-10,max_iter=100):
    for _ in range(max_iter):
        fx0=f(x0)
        fx1=f(x1)
        if abs(fx1)<tol:return x1
        x0,x1=x1,x1-fx1*(x1-x0)/(fx1-fx0)
    return x1

def bisection(f,a,b,tol=1e-10):
    if f(a)*f(b)>=0:return None
    while abs(b-a)>tol:
        c=(a+b)/2
        if f(c)==0:return c
        if f(a)*f(c)<0:b=c
        else:a=c
    return(a+b)/2

def fixed_point(g,x0,tol=1e-10,max_iter=100):
    x=x0
    for _ in range(max_iter):
        xnew=g(x)
        if abs(xnew-x)<tol:return xnew
        x=xnew
    return x

def runge_kutta(f,y0,t0,t1,n=100):
    h=(t1-t0)/n
    t=t0
    y=y0
    points=[(t,y)]
    for _ in range(n):
        k1=h*f(t,y)
        k2=h*f(t+h/2,y+k1/2)
        k3=h*f(t+h/2,y+k2/2)
        k4=h*f(t+h,y+k3)
        y=y+(k1+2*k2+2*k3+k4)/6
        t=t+h
        points.append((t,y))
    return points

def euler(f,y0,t0,t1,n=100):
    h=(t1-t0)/n
    t=t0
    y=y0
    points=[(t,y)]
    for _ in range(n):
        y=y+h*f(t,y)
        t=t+h
        points.append((t,y))
    return points

def adams_bashforth(f,y0,t0,t1,n=100):
    h=(t1-t0)/n
    t=t0
    y=y0
    points=[(t,y)]
    yprev=y
    for i in range(n):
        if i==0:
            ynew=y+h*f(t,y)
        else:
            ynew=y+h*(3*f(t,y)-f(t-h,yprev))/2
        yprev=y
        y=ynew
        t=t+h
        points.append((t,y))
    return points

def heun(f,y0,t0,t1,n=100):
    h=(t1-t0)/n
    t=t0
    y=y0
    points=[(t,y)]
    for _ in range(n):
        k1=f(t,y)
        k2=f(t+h,y+h*k1)
        y=y+h*(k1+k2)/2
        t=t+h
        points.append((t,y))
    return points

def midpoint(f,y0,t0,t1,n=100):
    h=(t1-t0)/n
    t=t0
    y=y0
    points=[(t,y)]
    for _ in range(n):
        k1=f(t,y)
        k2=f(t+h/2,y+h*k1/2)
        y=y+h*k2
        t=t+h
        points.append((t,y))
    return points

def verlet(f,x0,v0,t0,t1,dt=0.01):
    points=[(t0,x0,v0)]
    x,v=x0,v0
    t=t0
    while t<t1:
        a=f(x)
        xnew=x+v*dt+0.5*a*dt*dt
        anew=f(xnew)
        vnew=v+0.5*(a+anew)*dt
        x,v=xnew,vnew
        t+=dt
        points.append((t,x,v))
    return points

def leapfrog(f,x0,v0,t0,t1,dt=0.01):
    points=[(t0,x0,v0)]
    x,v=x0,v0
    t=t0
    vhalf=v+0.5*f(x)*dt
    while t<t1:
        x=x+vhalf*dt
        vhalf=vhalf+f(x)*dt
        v=vhalf-0.5*f(x)*dt
        t+=dt
        points.append((t,x,v))
    return points

def velocity_verlet(f,x0,v0,t0,t1,dt=0.01):
    points=[(t0,x0,v0)]
    x,v=x0,v0
    t=t0
    a=f(x)
    while t<t1:
        x=x+v*dt+0.5*a*dt*dt
        anew=f(x)
        v=v+0.5*(a+anew)*dt
        a=anew
        t+=dt
        points.append((t,x,v))
    return points

def beeman(f,x0,v0,t0,t1,dt=0.01):
    points=[(t0,x0,v0)]
    x,v=x0,v0
    t=t0
    a=f(x)
    aprev=a
    while t<t1:
        xnew=x+v*dt+(4*a-aprev)*dt*dt/6
        anew=f(xnew)
        vnew=v+(2*anew+5*a-aprev)*dt/6
        aprev,a=a,anew
        x,v=xnew,vnew
        t+=dt
        points.append((t,x,v))
    return points

def matrix_mult(A,B):
    rows_A,cols_A=len(A),len(A[0])
    rows_B,cols_B=len(B),len(B[0])
    if cols_A!=rows_B:return None
    C=[[0]*cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):C[i][j]+=A[i][k]*B[k][j]
    return C

def matrix_add(A,B):
    if len(A)!=len(B)or len(A[0])!=len(B[0]):return None
    return[[A[i][j]+B[i][j]for j in range(len(A[0]))]for i in range(len(A))]

def matrix_sub(A,B):
    if len(A)!=len(B)or len(A[0])!=len(B[0]):return None
    return[[A[i][j]-B[i][j]for j in range(len(A[0]))]for i in range(len(A))]

def matrix_scalar(A,c):return[[c*A[i][j]for j in range(len(A[0]))]for i in range(len(A))]

def matrix_transpose(A):return[[A[j][i]for j in range(len(A))]for i in range(len(A[0]))]

def matrix_det(A):
    n=len(A)
    if n==1:return A[0][0]
    if n==2:return A[0][0]*A[1][1]-A[0][1]*A[1][0]
    det=0
    for j in range(n):
        M=[[A[i][k]for k in range(n)if k!=j]for i in range(1,n)]
        det+=(-1)**j*A[0][j]*matrix_det(M)
    return det

def matrix_inv(A):
    n=len(A)
    det=matrix_det(A)
    if det==0:return None
    if n==1:return[[1/A[0][0]]]
    if n==2:return[[A[1][1]/det,-A[0][1]/det],[-A[1][0]/det,A[0][0]/det]]
    cofactors=[[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            M=[[A[r][c]for c in range(n)if c!=j]for r in range(n)if r!=i]
            cofactors[i][j]=(-1)**(i+j)*matrix_det(M)
    adjugate=matrix_transpose(cofactors)
    return matrix_scalar(adjugate,1/det)

def matrix_trace(A):return sum(A[i][i]for i in range(len(A)))

def matrix_rank(A):
    m,n=len(A),len(A[0])
    M=[row[:]for row in A]
    rank=0
    for col in range(n):
        pivot=None
        for row in range(rank,m):
            if M[row][col]!=0:
                pivot=row
                break
        if pivot is None:continue
        M[rank],M[pivot]=M[pivot],M[rank]
        for row in range(rank+1,m):
            if M[row][col]==0:continue
            factor=M[row][col]/M[rank][col]
            for c in range(col,n):M[row][c]-=factor*M[rank][c]
        rank+=1
    return rank

def gauss_elim(A,b):
    n=len(A)
    M=[A[i][:]+[b[i]]for i in range(n)]
    for i in range(n):
        max_row=max(range(i,n),key=lambda r:abs(M[r][i]))
        M[i],M[max_row]=M[max_row],M[i]
        for j in range(i+1,n):
            factor=M[j][i]/M[i][i]
            for k in range(i,n+1):M[j][k]-=factor*M[i][k]
    x=[0]*n
    for i in range(n-1,-1,-1):
        x[i]=M[i][n]
        for j in range(i+1,n):x[i]-=M[i][j]*x[j]
        x[i]/=M[i][i]
    return x

def lu_decomp(A):
    n=len(A)
    L=[[0]*n for _ in range(n)]
    U=[[0]*n for _ in range(n)]
    for i in range(n):
        for k in range(i,n):
            sum_val=sum(L[i][j]*U[j][k]for j in range(i))
            U[i][k]=A[i][k]-sum_val
        for k in range(i,n):
            if i==k:L[i][i]=1
            else:
                sum_val=sum(L[k][j]*U[j][i]for j in range(i))
                L[k][i]=(A[k][i]-sum_val)/U[i][i]
    return L,U

def cholesky(A):
    n=len(A)
    L=[[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            sum_val=sum(L[i][k]*L[j][k]for k in range(j))
            if i==j:L[i][j]=sqrt(A[i][i]-sum_val)
            else:L[i][j]=(A[i][j]-sum_val)/L[j][j]
    return L

def qr_decomp(A):
    m,n=len(A),len(A[0])
    Q=[[0]*m for _ in range(m)]
    R=[[0]*n for _ in range(m)]
    for j in range(n):
        v=[A[i][j]for i in range(m)]
        for i in range(j):
            R[i][j]=sum(Q[k][i]*A[k][j]for k in range(m))
            for k in range(m):v[k]-=R[i][j]*Q[k][i]
        norm=sqrt(sum(x*x for x in v))
        R[j][j]=norm
        for i in range(m):Q[i][j]=v[i]/norm
    return Q,R

def svd(A,max_iter=100):
    m,n=len(A),len(A[0])
    ATA=matrix_mult(matrix_transpose(A),A)
    U,S,VT=[],[],[]
    return U,S,VT

def eigenvalues(A,max_iter=100):
    n=len(A)
    Q,R=qr_decomp(A)
    Ak=matrix_mult(R,Q)
    for _ in range(max_iter-1):
        Q,R=qr_decomp(Ak)
        Ak=matrix_mult(R,Q)
    return[Ak[i][i]for i in range(n)]

def power_method(A,max_iter=1000):
    n=len(A)
    v=[1]*n
    for _ in range(max_iter):
        Av=[sum(A[i][j]*v[j]for j in range(n))for i in range(n)]
        norm=sqrt(sum(x*x for x in Av))
        v=[x/norm for x in Av]
    eigenval=sum(sum(A[i][j]*v[j]for j in range(n))*v[i]for i in range(n))
    return eigenval,v

def jacobi_method(A,b,max_iter=1000,tol=1e-10):
    n=len(A)
    x=[0]*n
    for _ in range(max_iter):
        xnew=[0]*n
        for i in range(n):
            sum_val=sum(A[i][j]*x[j]for j in range(n)if j!=i)
            xnew[i]=(b[i]-sum_val)/A[i][i]
        if all(abs(xnew[i]-x[i])<tol for i in range(n)):return xnew
        x=xnew
    return x

def gauss_seidel(A,b,max_iter=1000,tol=1e-10):
    n=len(A)
    x=[0]*n
    for _ in range(max_iter):
        xold=x[:]
        for i in range(n):
            sum1=sum(A[i][j]*x[j]for j in range(i))
            sum2=sum(A[i][j]*xold[j]for j in range(i+1,n))
            x[i]=(b[i]-sum1-sum2)/A[i][i]
        if all(abs(x[i]-xold[i])<tol for i in range(n)):return x
    return x

def conjugate_gradient(A,b,max_iter=1000,tol=1e-10):
    n=len(A)
    x=[0]*n
    r=[b[i]-sum(A[i][j]*x[j]for j in range(n))for i in range(n)]
    p=r[:]
    for _ in range(max_iter):
        Ap=[sum(A[i][j]*p[j]for j in range(n))for i in range(n)]
        alpha=sum(r[i]*r[i]for i in range(n))/sum(p[i]*Ap[i]for i in range(n))
        x=[x[i]+alpha*p[i]for i in range(n)]
        rnew=[r[i]-alpha*Ap[i]for i in range(n)]
        if sqrt(sum(rnew[i]*rnew[i]for i in range(n)))<tol:return x
        beta=sum(rnew[i]*rnew[i]for i in range(n))/sum(r[i]*r[i]for i in range(n))
        p=[rnew[i]+beta*p[i]for i in range(n)]
        r=rnew
    return x

def gram_schmidt(vectors):
    basis=[]
    for v in vectors:
        w=v[:]
        for b in basis:
            proj=sum(v[i]*b[i]for i in range(len(v)))/sum(b[i]*b[i]for i in range(len(b)))
            w=[w[i]-proj*b[i]for i in range(len(w))]
        norm=sqrt(sum(x*x for x in w))
        basis.append([x/norm for x in w])
    return basis

def dot(u,v):return sum(u[i]*v[i]for i in range(len(u)))

def cross(u,v):
    if len(u)!=3 or len(v)!=3:return None
    return[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]

def norm(v):return sqrt(sum(x*x for x in v))

def normalize(v):
    n=norm(v)
    return[x/n for x in v]

def projection(u,v):
    return[(dot(u,v)/dot(v,v))*x for x in v]

def angle(u,v):return acos(dot(u,v)/(norm(u)*norm(v)))

def distance_point_to_line(point,line_point,line_dir):
    v=[point[i]-line_point[i]for i in range(len(point))]
    proj=projection(v,line_dir)
    perp=[v[i]-proj[i]for i in range(len(v))]
    return norm(perp)

def distance_point_to_plane(point,plane_point,plane_normal):
    v=[point[i]-plane_point[i]for i in range(len(point))]
    return abs(dot(v,plane_normal))/norm(plane_normal)

def line_intersection(p1,d1,p2,d2):
    n=cross(d1,d2)
    if norm(n)==0:return None
    n1=cross(d1,n)
    t2=dot([p1[i]-p2[i]for i in range(3)],n1)/dot(d2,n1)
    return[p2[i]+t2*d2[i]for i in range(3)]

def plane_intersection(n1,d1,n2,d2):
    dir=cross(n1,n2)
    if norm(dir)==0:return None
    n1xn2=cross(n1,n2)
    point=[((d2*norm(n1)**2-d1*dot(n1,n2))*n1[i]+(d1*norm(n2)**2-d2*dot(n1,n2))*n2[i])/norm(n1xn2)**2 for i in range(3)]
    return point,dir

def triangle_area(a,b,c):
    s=(a+b+c)/2
    return sqrt(s*(s-a)*(s-b)*(s-c))

def circle_area(r):return PI*r*r

def circle_circumference(r):return 2*PI*r

def sphere_volume(r):return(4/3)*PI*r**3

def sphere_surface(r):return 4*PI*r*r

def cylinder_volume(r,h):return PI*r*r*h

def cylinder_surface(r,h):return 2*PI*r*h+2*PI*r*r

def cone_volume(r,h):return(1/3)*PI*r*r*h

def cone_surface(r,h):return PI*r*(r+sqrt(h*h+r*r))

def cube_volume(a):return a**3

def cube_surface(a):return 6*a*a

def rectangular_prism_volume(a,b,c):return a*b*c

def rectangular_prism_surface(a,b,c):return 2*(a*b+b*c+c*a)

def pyramid_volume(base_area,h):return(1/3)*base_area*h

def tetrahedron_volume(a):return(a**3)/(6*sqrt(2))

def ellipse_area(a,b):return PI*a*b

def ellipse_circumference(a,b):
    h=((a-b)**2)/((a+b)**2)
    return PI*(a+b)*(1+3*h/(10+sqrt(4-3*h)))

def torus_volume(R,r):return 2*PI*PI*R*r*r

def torus_surface(R,r):return 4*PI*PI*R*r

def arc_length(r,theta):return r*theta

def sector_area(r,theta):return 0.5*r*r*theta

def segment_area(r,theta):return 0.5*r*r*(theta-sin(theta))

def annulus_area(R,r):return PI*(R*R-r*r)

def regular_polygon_area(n,s):return(n*s*s)/(4*tan(PI/n))

def regular_polygon_perimeter(n,s):return n*s

def apothem(n,s):return s/(2*tan(PI/n))

def inradius(n,s):return apothem(n,s)

def circumradius(n,s):return s/(2*sin(PI/n))

def golden_ratio():return PHI

def golden_angle():return 2*PI/PHI**2

def fibonacci_spiral(n):
    points=[]
    x,y=0,0
    dx,dy=1,0
    for i in range(n):
        points.append((x,y))
        f=fib(i+1)
        for _ in range(f):
            x+=dx
            y+=dy
        dx,dy=-dy,dx
    return points

def archimedean_spiral(a,b,theta):return a+b*theta

def logarithmic_spiral(a,b,theta):return a*exp(b*theta)

def fermat_spiral(a,theta):return a*sqrt(theta)

def hyperbolic_spiral(a,theta):return a/theta

def lituus(a,theta):return a/sqrt(theta)

def rose_curve(a,k,theta):return a*cos(k*theta)

def cardioid(a,theta):return a*(1+cos(theta))

def limacon(a,b,theta):return a+b*cos(theta)

def lemniscate(a,theta):return a*sqrt(cos(2*theta))

def astroid(a,t):return(a*cos(t)**3,a*sin(t)**3)

def cycloid(a,t):return(a*(t-sin(t)),a*(1-cos(t)))

def epicycloid(R,r,t):return((R+r)*cos(t)-r*cos((R+r)*t/r),(R+r)*sin(t)-r*sin((R+r)*t/r))

def hypocycloid(R,r,t):return((R-r)*cos(t)+r*cos((R-r)*t/r),(R-r)*sin(t)-r*sin((R-r)*t/r))

def epitrochoid(R,r,d,t):return((R+r)*cos(t)-d*cos((R+r)*t/r),(R+r)*sin(t)-d*sin((R+r)*t/r))

def hypotrochoid(R,r,d,t):return((R-r)*cos(t)+d*cos((R-r)*t/r),(R-r)*sin(t)-d*sin((R-r)*t/r))

def involute_circle(a,t):return(a*(cos(t)+t*sin(t)),a*(sin(t)-t*cos(t)))

def folium_descartes(a,t):return(3*a*t/(1+t**3),3*a*t*t/(1+t**3))

def devil_curve(a,b,t):return((a*a*sin(t)*sin(t)-b*b*cos(t)*cos(t))/(sin(t)*sin(t)+cos(t)*cos(t))**0.5,
                              (a*a*sin(t)*cos(t)+b*b*sin(t)*cos(t))/(sin(t)*sin(t)+cos(t)*cos(t))**0.5)

def butterfly_curve(t):return(sin(t)*(exp(cos(t))-2*cos(4*t)-sin(t/12)**5),
                              cos(t)*(exp(cos(t))-2*cos(4*t)-sin(t/12)**5))

def heart_curve(t):return(16*sin(t)**3,13*cos(t)-5*cos(2*t)-2*cos(3*t)-cos(4*t))

def eight_curve(a,t):return(a*sin(t),a*sin(t)*cos(t))

def witch_agnesi(a,t):return(2*a*t,2*a/(1+t*t))

def cissoid_diocles(a,t):return(2*a*t*t/(1+t*t),2*a*t*t*t/(1+t*t))

def conchoid_nicomedes(a,b,t):return((a+b/cos(t))*cos(t),(a+b/cos(t))*sin(t))

def strophoid(a,t):return(a*(1-t*t)/(1+t*t),a*t*(1-t*t)/(1+t*t))

def serpentine_curve(a,b,t):return(a*t,b*t/(1+t*t))

def kampyle_eudoxus(a,t):return(a/sqrt(1+t*t),a*t/sqrt(1+t*t))

def kappa_curve(a,t):return(a*t,a*(1-t*t)/(1+t*t))

def pearls_sluse(a,t):return((a*(1+t*t))**(1/3)*cos(t),(a*(1+t*t))**(1/3)*sin(t))

def trident_newton(a,b,c,t):return(t,a*t**3+b*t+c)

def trisectrix_maclaurin(a,t):return(a*(3*cos(t)-cos(3*t))/2,a*sin(t)*(3-4*sin(t)*sin(t)))

def quadratrix_hippias(a,t):return(a*t/tan(t),a*t)

def cochleoid(a,t):return(a*sin(t)/t,a*(1-cos(t))/t)

def tractrix(a,t):return(a*(t-tanh(t)),a/cosh(t))

def catenary(a,x):return a*cosh(x/a)

def brachistochrone(a,t):return(a*(t-sin(t)),a*(1-cos(t)))

def tautochrone(a,t):return(a*(t+sin(t)),a*(1+cos(t)))

def enneper_surface(u,v):return(u-u**3/3+u*v*v,v-v**3/3+v*u*u,u*u-v*v)

def helicoid(u,v,a):return(a*v*cos(u),a*v*sin(u),a*u)

def catenoid(u,v,c):return(c*cosh(v)*cos(u),c*cosh(v)*sin(u),c*v)

def mobius_strip(u,v,R):
    x=(R+v*cos(u/2))*cos(u)
    y=(R+v*cos(u/2))*sin(u)
    z=v*sin(u/2)
    return(x,y,z)

def klein_bottle(u,v):
    r=4*(1-cos(u)/2)
    if u<PI:x=6*cos(u)*(1+sin(u))+r*cos(v+PI)
    else:x=6*cos(u)*(1+sin(u))+r*cos(u)*cos(v)
    if u<PI:y=16*sin(u)+r*sin(v+PI)
    else:y=16*sin(u)
    z=r*sin(u)*sin(v)
    return(x,y,z)

def boys_surface(u,v):
    x=(sqrt(2)*cos(2*u)*cos(v)*cos(v)+cos(u)*sin(2*v))/(2-sqrt(2)*sin(3*u)*sin(2*v))
    y=(sqrt(2)*sin(2*u)*cos(v)*cos(v)-sin(u)*sin(2*v))/(2-sqrt(2)*sin(3*u)*sin(2*v))
    z=(3*cos(v)*cos(v))/(2-sqrt(2)*sin(3*u)*sin(2*v))
    return(x,y,z)

def roman_surface(u,v):
    x=cos(u)*cos(u)*sin(2*v)/2
    y=sin(u)*cos(u)*sin(2*v)/2
    z=cos(u)*sin(u)*cos(v)
    return(x,y,z)

def cross_cap(u,v):
    x=cos(u)*sin(2*v)
    y=sin(u)*sin(2*v)
    z=cos(v)*cos(v)-cos(u)*cos(u)*sin(v)*sin(v)
    return(x,y,z)

def whitney_umbrella(u,v):
    x=u*v
    y=u
    z=v*v
    return(x,y,z)

def dini_surface(u,v,a=1,b=1):
    x=a*cos(u)*sin(v)
    y=a*sin(u)*sin(v)
    z=a*(cos(v)+log(tan(v/2)))+b*u
    return(x,y,z)

def breather_surface(u,v,a=0.4):
    w=sqrt(1-a*a)
    denom=a*((w*cosh(a*u))**2+(a*sin(w*v))**2)
    x=-u+2*(1-a*a)*cosh(a*u)*sinh(a*u)/denom
    y=2*w*cosh(a*u)*(-(w*cos(v)*cos(w*v))-sin(v)*sin(w*v))/denom
    z=2*w*cosh(a*u)*(-(w*sin(v)*cos(w*v))+cos(v)*sin(w*v))/denom
    return(x,y,z)

def kuen_surface(u,v):
    x=2*(cos(u)+u*sin(u))*sin(v)/(1+u*u*sin(v)*sin(v))
    y=2*(sin(u)-u*cos(u))*sin(v)/(1+u*u*sin(v)*sin(v))
    z=log(tan(v/2))+2*cos(v)/(1+u*u*sin(v)*sin(v))
    return(x,y,z)

def henneberg_surface(u,v):
    x=2*sinh(u)*cos(v)-2*sinh(3*u)*cos(3*v)/3
    y=2*sinh(u)*sin(v)+2*sinh(3*u)*sin(3*v)/3
    z=2*cosh(2*u)*cos(2*v)
    return(x,y,z)

def scherk_surface(u,v):
    x=u
    y=v
    z=log(cos(v)/cos(u))
    return(x,y,z)

def richmond_surface(u,v):
    x=sin(u)*cos(v)/sqrt(2)
    y=sin(u)*sin(v)/sqrt(2)
    z=(cos(u)*cos(u)-log(tan(u/2))-1/4*sin(2*u))/sqrt(2)
    return(x,y,z)

def bour_surface(u,v):
    x=u*cos(v)-u*u*u*cos(3*v)/3
    y=u*sin(v)+u*u*u*sin(3*v)/3
    z=2*u*u*cos(2*v)
    return(x,y,z)

def plucker_conoid(u,v):
    x=u*v/sqrt(u*u+v*v+1)
    y=u/sqrt(u*u+v*v+1)
    z=v/sqrt(u*u+v*v+1)
    return(x,y,z)

def clifford_torus(u,v,r=1,R=2):
    x=(R+r*cos(v))*cos(u)/sqrt(2)
    y=(R+r*cos(v))*sin(u)/sqrt(2)
    z=r*sin(v)/sqrt(2)
    return(x,y,z)

def dupin_cyclide(u,v,a=1,b=0.5,c=0.3):
    x=(c*(a+b*cos(u))*cos(v))/(a+b*cos(u)*cos(v))
    y=(c*(a+b*cos(u))*sin(v))/(a+b*cos(u)*cos(v))
    z=(c*b*sin(u))/(a+b*cos(u)*cos(v))
    return(x,y,z)

def apple_surface(u,v):
    x=cos(u)*(4+3.8*cos(v))
    y=sin(u)*(4+3.8*cos(v))
    z=(cos(v)+sin(v)-1)*(1+sin(v))*log(1-PI*v/10)+7.5*sin(v)
    return(x,y,z)

def seashell(u,v,a=1,b=1,c=0.1,n=2):
    phi=(1-v/(2*PI))*(2+sin(u))
    x=a*(1-v/(2*PI))*cos(n*v)*(1+cos(u))+c*cos(n*v)
    y=a*(1-v/(2*PI))*sin(n*v)*(1+cos(u))+c*sin(n*v)
    z=b*v/(2*PI)+a*(1-v/(2*PI))*sin(u)
    return(x,y,z)

def sphere_point(theta,phi,r=1):
    x=r*sin(phi)*cos(theta)
    y=r*sin(phi)*sin(theta)
    z=r*cos(phi)
    return(x,y,z)

def cylinder_point(theta,z,r=1):
    x=r*cos(theta)
    y=r*sin(theta)
    return(x,y,z)

def torus_point(u,v,R=2,r=1):
    x=(R+r*cos(v))*cos(u)
    y=(R+r*cos(v))*sin(u)
    z=r*sin(v)
    return(x,y,z)

def polar_to_cartesian(r,theta):return(r*cos(theta),r*sin(theta))

def cartesian_to_polar(x,y):return(hypot(x,y),atan2(y,x))

def spherical_to_cartesian(r,theta,phi):return(r*sin(phi)*cos(theta),r*sin(phi)*sin(theta),r*cos(phi))

def cartesian_to_spherical(x,y,z):
    r=hypot(x,y,z)
    theta=atan2(y,x)
    phi=acos(z/r)if r!=0 else 0
    return(r,theta,phi)

def cylindrical_to_cartesian(r,theta,z):return(r*cos(theta),r*sin(theta),z)

def cartesian_to_cylindrical(x,y,z):return(hypot(x,y),atan2(y,x),z)

def rot_x(angle):
    c,s=cos(angle),sin(angle)
    return[[1,0,0],[0,c,-s],[0,s,c]]

def rot_y(angle):
    c,s=cos(angle),sin(angle)
    return[[c,0,s],[0,1,0],[-s,0,c]]

def rot_z(angle):
    c,s=cos(angle),sin(angle)
    return[[c,-s,0],[s,c,0],[0,0,1]]

def rotate_point(point,axis,angle):
    if axis=='x':R=rot_x(angle)
    elif axis=='y':R=rot_y(angle)
    elif axis=='z':R=rot_z(angle)
    else:return point
    return[sum(R[i][j]*point[j]for j in range(3))for i in range(3)]

def scale_matrix(sx,sy,sz=1):return[[sx,0,0],[0,sy,0],[0,0,sz]]

def translation_matrix(tx,ty,tz=0):return[[1,0,0,tx],[0,1,0,ty],[0,0,1,tz],[0,0,0,1]]

def reflect_x():return[[-1,0,0],[0,1,0],[0,0,1]]

def reflect_y():return[[1,0,0],[0,-1,0],[0,0,1]]

def reflect_z():return[[1,0,0],[0,1,0],[0,0,-1]]

def shear_xy(k):return[[1,k,0],[0,1,0],[0,0,1]]

def shear_xz(k):return[[1,0,k],[0,1,0],[0,0,1]]

def shear_yx(k):return[[1,0,0],[k,1,0],[0,0,1]]

def shear_yz(k):return[[1,0,0],[0,1,k],[0,0,1]]

def shear_zx(k):return[[1,0,0],[0,1,0],[k,0,1]]

def shear_zy(k):return[[1,0,0],[0,1,0],[0,k,1]]

def rodriguez_rotation(axis,angle):
    axis=normalize(axis)
    K=[[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]]
    I=[[1,0,0],[0,1,0],[0,0,1]]
    c,s=cos(angle),sin(angle)
    R=matrix_add(I,matrix_add(matrix_scalar(K,s),matrix_scalar(matrix_mult(K,K),1-c)))
    return R

def euler_angles(alpha,beta,gamma):
    Rz1=rot_z(alpha)
    Rx=rot_x(beta)
    Rz2=rot_z(gamma)
    return matrix_mult(matrix_mult(Rz1,Rx),Rz2)

def quaternion_mult(q1,q2):
    w1,x1,y1,z1=q1
    w2,x2,y2,z2=q2
    return(w1*w2-x1*x2-y1*y2-z1*z2,
           w1*x2+x1*w2+y1*z2-z1*y2,
           w1*y2-x1*z2+y1*w2+z1*x2,
           w1*z2+x1*y2-y1*x2+z1*w2)

def quaternion_conjugate(q):
    w,x,y,z=q
    return(w,-x,-y,-z)

def quaternion_norm(q):return sqrt(sum(x*x for x in q))

def quaternion_normalize(q):
    n=quaternion_norm(q)
    return tuple(x/n for x in q)

def quaternion_to_rotation(q):
    w,x,y,z=quaternion_normalize(q)
    return[[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
           [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
           [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]]

def rotation_to_quaternion(R):
    trace=R[0][0]+R[1][1]+R[2][2]
    if trace>0:
        s=0.5/sqrt(trace+1.0)
        w=0.25/s
        x=(R[2][1]-R[1][2])*s
        y=(R[0][2]-R[2][0])*s
        z=(R[1][0]-R[0][1])*s
    elif R[0][0]>R[1][1]and R[0][0]>R[2][2]:
        s=2.0*sqrt(1.0+R[0][0]-R[1][1]-R[2][2])
        w=(R[2][1]-R[1][2])/s
        x=0.25*s
        y=(R[0][1]+R[1][0])/s
        z=(R[0][2]+R[2][0])/s
    elif R[1][1]>R[2][2]:
        s=2.0*sqrt(1.0+R[1][1]-R[0][0]-R[2][2])
        w=(R[0][2]-R[2][0])/s
        x=(R[0][1]+R[1][0])/s
        y=0.25*s
        z=(R[1][2]+R[2][1])/s
    else:
        s=2.0*sqrt(1.0+R[2][2]-R[0][0]-R[1][1])
        w=(R[1][0]-R[0][1])/s
        x=(R[0][2]+R[2][0])/s
        y=(R[1][2]+R[2][1])/s
        z=0.25*s
    return(w,x,y,z)

def slerp(q1,q2,t):
    dot_product=sum(a*b for a,b in zip(q1,q2))
    if dot_product<0:
        q2=tuple(-x for x in q2)
        dot_product=-dot_product
    if dot_product>0.9995:
        result=tuple(a+(b-a)*t for a,b in zip(q1,q2))
        return quaternion_normalize(result)
    theta=acos(dot_product)
    sin_theta=sin(theta)
    w1=sin((1-t)*theta)/sin_theta
    w2=sin(t*theta)/sin_theta
    return tuple(w1*a+w2*b for a,b in zip(q1,q2))

def perspective_projection(fov,aspect,near,far):
    f=1.0/tan(fov/2)
    return[[f/aspect,0,0,0],
           [0,f,0,0],
           [0,0,(far+near)/(near-far),(2*far*near)/(near-far)],
           [0,0,-1,0]]

def orthographic_projection(left,right,bottom,top,near,far):
    return[[2/(right-left),0,0,-(right+left)/(right-left)],
           [0,2/(top-bottom),0,-(top+bottom)/(top-bottom)],
           [0,0,-2/(far-near),-(far+near)/(far-near)],
           [0,0,0,1]]

def look_at(eye,center,up):
    f=normalize([center[i]-eye[i]for i in range(3)])
    s=normalize(cross(f,up))
    u=cross(s,f)
    return[[s[0],s[1],s[2],-dot(s,eye)],
           [u[0],u[1],u[2],-dot(u,eye)],
           [-f[0],-f[1],-f[2],dot(f,eye)],
           [0,0,0,1]]

def viewport(x,y,width,height):
    return[[width/2,0,0,x+width/2],
           [0,height/2,0,y+height/2],
           [0,0,1,0],
           [0,0,0,1]]

def barycentric(p,a,b,c):
    v0=[b[i]-a[i]for i in range(2)]
    v1=[c[i]-a[i]for i in range(2)]
    v2=[p[i]-a[i]for i in range(2)]
    d00=dot(v0,v0)
    d01=dot(v0,v1)
    d11=dot(v1,v1)
    d20=dot(v2,v0)
    d21=dot(v2,v1)
    denom=d00*d11-d01*d01
    v=(d11*d20-d01*d21)/denom
    w=(d00*d21-d01*d20)/denom
    u=1.0-v-w
    return(u,v,w)

def point_in_triangle(p,a,b,c):
    u,v,w=barycentric(p,a,b,c)
    return u>=0 and v>=0 and w>=0

def point_in_polygon(point,polygon):
    x,y=point
    n=len(polygon)
    inside=False
    p1x,p1y=polygon[0]
    for i in range(1,n+1):
        p2x,p2y=polygon[i%n]
        if y>min(p1y,p2y):
            if y<=max(p1y,p2y):
                if x<=max(p1x,p2x):
                    if p1y!=p2y:
                        xinters=(y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x==p2x or x<=xinters:
                        inside=not inside
        p1x,p1y=p2x,p2y
    return inside

def convex_hull(points):
    points=sorted(set(points))
    if len(points)<=1:return points
    def cross(o,a,b):
        return(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in points:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0:
            lower.pop()
        lower.append(p)
    upper=[]
    for p in reversed(points):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0:
            upper.pop()
        upper.append(p)
    return lower[:-1]+upper[:-1]

def polygon_area_2d(vertices):
    n=len(vertices)
    area=0
    for i in range(n):
        j=(i+1)%n
        area+=vertices[i][0]*vertices[j][1]
        area-=vertices[j][0]*vertices[i][1]
    return abs(area)/2

def polygon_centroid(vertices):
    n=len(vertices)
    area=polygon_area_2d(vertices)
    cx,cy=0,0
    for i in range(n):
        j=(i+1)%n
        factor=vertices[i][0]*vertices[j][1]-vertices[j][0]*vertices[i][1]
        cx+=(vertices[i][0]+vertices[j][0])*factor
        cy+=(vertices[i][1]+vertices[j][1])*factor
    factor=1/(6*area)
    return(cx*factor,cy*factor)

def delaunay_triangulation(points):
    if len(points)<3:return[]
    triangles=[]
    return triangles

def voronoi_diagram(points):
    cells=[]
    return cells

def closest_pair(points):
    def distance(p1,p2):
        return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    n=len(points)
    if n<=1:return None
    if n==2:return(points[0],points[1],distance(points[0],points[1]))
    min_dist=float('inf')
    pair=None
    for i in range(n):
        for j in range(i+1,n):
            d=distance(points[i],points[j])
            if d<min_dist:
                min_dist=d
                pair=(points[i],points[j])
    return pair+(min_dist,)

def furthest_pair(points):
    def distance(p1,p2):
        return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    n=len(points)
    if n<=1:return None
    if n==2:return(points[0],points[1],distance(points[0],points[1]))
    max_dist=0
    pair=None
    for i in range(n):
        for j in range(i+1,n):
            d=distance(points[i],points[j])
            if d>max_dist:
                max_dist=d
                pair=(points[i],points[j])
    return pair+(max_dist,)

def graham_scan(points):
    def polar_angle(p0,p1):
        return atan2(p1[1]-p0[1],p1[0]-p0[0])
    def cross(o,a,b):
        return(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    start=min(points,key=lambda p:(p[1],p[0]))
    sorted_points=sorted(points,key=lambda p:(polar_angle(start,p),dist(start,p)))
    hull=[sorted_points[0],sorted_points[1]]
    for p in sorted_points[2:]:
        while len(hull)>1 and cross(hull[-2],hull[-1],p)<=0:
            hull.pop()
        hull.append(p)
    return hull

def jarvis_march(points):
    def orientation(p,q,r):
        val=(q[1]-p[1])*(r[0]-q[0])-(q[0]-p[0])*(r[1]-q[1])
        if val==0:return 0
        return 1 if val>0 else 2
    n=len(points)
    if n<3:return points
    hull=[]
    l=min(range(n),key=lambda i:points[i][0])
    p=l
    while True:
        hull.append(points[p])
        q=(p+1)%n
        for i in range(n):
            if orientation(points[p],points[i],points[q])==2:
                q=i
        p=q
        if p==l:break
    return hull

def line_segment_intersection(p1,p2,p3,p4):
    def ccw(A,B,C):
        return(C[1]-A[1])*(B[0]-A[0])>(B[1]-A[1])*(C[0]-A[0])
    return ccw(p1,p3,p4)!=ccw(p2,p3,p4)and ccw(p1,p2,p3)!=ccw(p1,p2,p4)

def ray_casting(point,polygon):
    x,y=point
    n=len(polygon)
    inside=False
    p1x,p1y=polygon[0]
    for i in range(n):
        p2x,p2y=polygon[(i+1)%n]
        if y>min(p1y,p2y):
            if y<=max(p1y,p2y):
                if x<=max(p1x,p2x):
                    if p1y!=p2y:
                        xinters=(y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x==p2x or x<=xinters:
                        inside=not inside
        p1x,p1y=p2x,p2y
    return inside

def sutherland_hodgman(polygon,clip_polygon):
    def inside_edge(point,edge_p1,edge_p2):
        return(edge_p2[0]-edge_p1[0])*(point[1]-edge_p1[1])-(edge_p2[1]-edge_p1[1])*(point[0]-edge_p1[0])>=0
    def line_intersection(p1,p2,p3,p4):
        x1,y1=p1
        x2,y2=p2
        x3,y3=p3
        x4,y4=p4
        denom=(x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
        if denom==0:return None
        t=((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))/denom
        return(x1+t*(x2-x1),y1+t*(y2-y1))
    output=list(polygon)
    for i in range(len(clip_polygon)):
        if not output:break
        input_list=output
        output=[]
        edge_p1=clip_polygon[i]
        edge_p2=clip_polygon[(i+1)%len(clip_polygon)]
        for j in range(len(input_list)):
            current=input_list[j]
            previous=input_list[j-1]
            if inside_edge(current,edge_p1,edge_p2):
                if not inside_edge(previous,edge_p1,edge_p2):
                    intersection=line_intersection(previous,current,edge_p1,edge_p2)
                    if intersection:output.append(intersection)
                output.append(current)
            elif inside_edge(previous,edge_p1,edge_p2):
                intersection=line_intersection(previous,current,edge_p1,edge_p2)
                if intersection:output.append(intersection)
    return output

def cohen_sutherland(x1,y1,x2,y2,xmin,ymin,xmax,ymax):
    INSIDE,LEFT,RIGHT,BOTTOM,TOP=0,1,2,4,8
    def compute_code(x,y):
        code=INSIDE
        if x<xmin:code|=LEFT
        elif x>xmax:code|=RIGHT
        if y<ymin:code|=BOTTOM
        elif y>ymax:code|=TOP
        return code
    code1=compute_code(x1,y1)
    code2=compute_code(x2,y2)
    accept=False
    while True:
        if code1==0 and code2==0:
            accept=True
            break
        elif code1&code2!=0:break
        else:
            code_out=code1 if code1!=0 else code2
            if code_out&TOP:
                x=x1+(x2-x1)*(ymax-y1)/(y2-y1)
                y=ymax
            elif code_out&BOTTOM:
                x=x1+(x2-x1)*(ymin-y1)/(y2-y1)
                y=ymin
            elif code_out&RIGHT:
                y=y1+(y2-y1)*(xmax-x1)/(x2-x1)
                x=xmax
            elif code_out&LEFT:
                y=y1+(y2-y1)*(xmin-x1)/(x2-x1)
                x=xmin
            if code_out==code1:
                x1,y1=x,y
                code1=compute_code(x1,y1)
            else:
                x2,y2=x,y
                code2=compute_code(x2,y2)
    return(x1,y1,x2,y2)if accept else None

def liang_barsky(x1,y1,x2,y2,xmin,ymin,xmax,ymax):
    dx,dy=x2-x1,y2-y1
    p=[-dx,dx,-dy,dy]
    q=[x1-xmin,xmax-x1,y1-ymin,ymax-y1]
    u1,u2=0.0,1.0
    for i in range(4):
        if p[i]==0:
            if q[i]<0:return None
        else:
            t=q[i]/p[i]
            if p[i]<0:
                if t>u2:return None
                if t>u1:u1=t
            else:
                if t<u1:return None
                if t<u2:u2=t
    return(x1+u1*dx,y1+u1*dy,x1+u2*dx,y1+u2*dy)

def bresenham_line(x0,y0,x1,y1):
    points=[]
    dx=abs(x1-x0)
    dy=abs(y1-y0)
    sx=1 if x0<x1 else-1
    sy=1 if y0<y1 else-1
    err=dx-dy
    while True:
        points.append((x0,y0))
        if x0==x1 and y0==y1:break
        e2=2*err
        if e2>-dy:
            err-=dy
            x0+=sx
        if e2<dx:
            err+=dx
            y0+=sy
    return points

def dda_line(x0,y0,x1,y1):
    points=[]
    dx,dy=x1-x0,y1-y0
    steps=max(abs(dx),abs(dy))
    x_inc,y_inc=dx/steps,dy/steps
    x,y=x0,y0
    for _ in range(int(steps)+1):
        points.append((round(x),round(y)))
        x+=x_inc
        y+=y_inc
    return points

def bresenham_circle(xc,yc,r):
    points=[]
    x,y=0,r
    d=3-2*r
    while y>=x:
        for px,py in[(xc+x,yc+y),(xc-x,yc+y),(xc+x,yc-y),(xc-x,yc-y),
                     (xc+y,yc+x),(xc-y,yc+x),(xc+y,yc-x),(xc-y,yc-x)]:
            points.append((px,py))
        x+=1
        if d>0:
            y-=1
            d=d+4*(x-y)+10
        else:
            d=d+4*x+6
    return points

def midpoint_circle(xc,yc,r):
    points=[]
    x,y=r,0
    p=1-r
    while x>=y:
        for px,py in[(xc+x,yc+y),(xc-x,yc+y),(xc+x,yc-y),(xc-x,yc-y),
                     (xc+y,yc+x),(xc-y,yc+x),(xc+y,yc-x),(xc-y,yc-x)]:
            points.append((px,py))
        y+=1
        if p<=0:
            p=p+2*y+1
        else:
            x-=1
            p=p+2*y-2*x+1
    return points

def bresenham_ellipse(xc,yc,a,b):
    points=[]
    x,y=0,b
    a2,b2=a*a,b*b
    d1=b2-a2*b+0.25*a2
    dx,dy=2*b2*x,2*a2*y
    while dx<dy:
        for px,py in[(xc+x,yc+y),(xc-x,yc+y),(xc+x,yc-y),(xc-x,yc-y)]:
            points.append((px,py))
        if d1<0:
            x+=1
            dx+=2*b2
            d1+=dx+b2
        else:
            x+=1
            y-=1
            dx+=2*b2
            dy-=2*a2
            d1+=dx-dy+b2
    d2=b2*(x+0.5)*(x+0.5)+a2*(y-1)*(y-1)-a2*b2
    while y>=0:
        for px,py in[(xc+x,yc+y),(xc-x,yc+y),(xc+x,yc-y),(xc-x,yc-y)]:
            points.append((px,py))
        if d2>0:
            y-=1
            dy-=2*a2
            d2+=a2-dy
        else:
            y-=1
            x+=1
            dx+=2*b2
            dy-=2*a2
            d2+=dx-dy+a2
    return points

def flood_fill(image,x,y,fill_color,boundary_color):
    if x<0 or x>=len(image[0])or y<0 or y>=len(image):return
    if image[y][x]==boundary_color or image[y][x]==fill_color:return
    original_color=image[y][x]
    stack=[(x,y)]
    while stack:
        cx,cy=stack.pop()
        if image[cy][cx]!=original_color:continue
        image[cy][cx]=fill_color
        for dx,dy in[(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny=cx+dx,cy+dy
            if 0<=nx<len(image[0])and 0<=ny<len(image):
                if image[ny][nx]==original_color:
                    stack.append((nx,ny))

def boundary_fill(image,x,y,fill_color,boundary_color):
    if x<0 or x>=len(image[0])or y<0 or y>=len(image):return
    if image[y][x]==boundary_color or image[y][x]==fill_color:return
    stack=[(x,y)]
    while stack:
        cx,cy=stack.pop()
        if image[cy][cx]==boundary_color or image[cy][cx]==fill_color:continue
        image[cy][cx]=fill_color
        for dx,dy in[(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny=cx+dx,cy+dy
            if 0<=nx<len(image[0])and 0<=ny<len(image):
                stack.append((nx,ny))

def scan_line_fill(polygon):
    if not polygon:return[]
    ymin=min(p[1]for p in polygon)
    ymax=max(p[1]for p in polygon)
    filled_pixels=[]
    for y in range(ymin,ymax+1):
        intersections=[]
        for i in range(len(polygon)):
            p1=polygon[i]
            p2=polygon[(i+1)%len(polygon)]
            if p1[1]!=p2[1]:
                if min(p1[1],p2[1])<=y<max(p1[1],p2[1]):
                    x=p1[0]+(y-p1[1])*(p2[0]-p1[0])/(p2[1]-p1[1])
                    intersections.append(x)
        intersections.sort()
        for i in range(0,len(intersections),2):
            if i+1<len(intersections):
                for x in range(int(intersections[i]),int(intersections[i+1])+1):
                    filled_pixels.append((x,y))
    return filled_pixels

def phong_shading(normal,light_dir,view_dir,ka,kd,ks,shininess):
    ambient=ka
    diffuse=kd*max(0,dot(normal,light_dir))
    reflect_dir=[2*dot(normal,light_dir)*n-l for n,l in zip(normal,light_dir)]
    specular=ks*max(0,dot(reflect_dir,view_dir))**shininess
    return ambient+diffuse+specular

def gouraud_shading(vertices,normals,light_dir,view_dir,ka,kd,ks,shininess):
    intensities=[]
    for normal in normals:
        intensity=phong_shading(normal,light_dir,view_dir,ka,kd,ks,shininess)
        intensities.append(intensity)
    return intensities

def flat_shading(normal,light_dir,ka,kd):
    ambient=ka
    diffuse=kd*max(0,dot(normal,light_dir))
    return ambient+diffuse

def texture_mapping(u,v,texture):
    w,h=len(texture[0]),len(texture)
    x=int(u*w)%w
    y=int(v*h)%h
    return texture[y][x]

def normal_mapping(u,v,normal_map):
    return texture_mapping(u,v,normal_map)

def bump_mapping(u,v,height_map,scale=1.0):
    w,h=len(height_map[0]),len(height_map)
    x,y=int(u*w)%w,int(v*h)%h
    h0=height_map[y][x]
    h1=height_map[y][(x+1)%w]
    h2=height_map[(y+1)%h][x]
    du=(h1-h0)*scale
    dv=(h2-h0)*scale
    return normalize([-du,-dv,1])

def environment_mapping(reflection_dir,cubemap):
    x,y,z=reflection_dir
    ax,ay,az=abs(x),abs(y),abs(z)
    if ax>=ay and ax>=az:
        face='right'if x>0 else'left'
        u=(y/ax+1)/2 if x>0 else(y/-ax+1)/2
        v=(z/ax+1)/2
    elif ay>=ax and ay>=az:
        face='top'if y>0 else'bottom'
        u=(x/ay+1)/2
        v=(z/ay+1)/2 if y>0 else(z/-ay+1)/2
    else:
        face='front'if z>0 else'back'
        u=(x/az+1)/2 if z>0 else(x/-az+1)/2
        v=(y/az+1)/2
    return texture_mapping(u,v,cubemap[face])

def shadow_mapping(light_pos,point,depth_map):
    light_dir=normalize([point[i]-light_pos[i]for i in range(3)])
    depth=dist(light_pos,point)
    u,v=0.5,0.5
    stored_depth=texture_mapping(u,v,depth_map)
    return depth>stored_depth+0.001

def ambient_occlusion(point,normal,samples=16):
    occlusion=0
    for i in range(samples):
        theta=2*PI*i/samples
        phi=acos(1-2*(i+0.5)/samples)
        sample_dir=spherical_to_cartesian(1,theta,phi)
        if dot(sample_dir,normal)>0:
            occlusion+=1
    return 1-occlusion/samples

def fresnel_schlick(cos_theta,f0):
    return f0+(1-f0)*(1-cos_theta)**5

def cook_torrance(normal,light_dir,view_dir,roughness,metallic,albedo):
    half_vec=normalize([l+v for l,v in zip(light_dir,view_dir)])
    ndotl=max(0,dot(normal,light_dir))
    ndotv=max(0,dot(normal,view_dir))
    ndoth=max(0,dot(normal,half_vec))
    vdoth=max(0,dot(view_dir,half_vec))
    f0=[0.04]*(1-metallic)+[c*metallic for c in albedo]
    F=fresnel_schlick(vdoth,f0)
    a=roughness*roughness
    a2=a*a
    denom=ndoth*ndoth*(a2-1)+1
    D=a2/(PI*denom*denom)
    k=(roughness+1)**2/8
    G1=ndotv/(ndotv*(1-k)+k)
    G2=ndotl/(ndotl*(1-k)+k)
    G=G1*G2
    specular=[F[i]*D*G/(4*ndotv*ndotl)if ndotv*ndotl>0 else 0 for i in range(len(F))]
    diffuse=[(1-F[i])*albedo[i]/PI for i in range(len(albedo))]
    return[(diffuse[i]+specular[i])*ndotl for i in range(len(albedo))]

def pbr_shading(normal,light_dir,view_dir,roughness,metallic,albedo):
    return cook_torrance(normal,light_dir,view_dir,roughness,metallic,albedo)

def ray_sphere_intersection(origin,direction,center,radius):
    oc=[origin[i]-center[i]for i in range(3)]
    a=dot(direction,direction)
    b=2*dot(oc,direction)
    c=dot(oc,oc)-radius*radius
    discriminant=b*b-4*a*c
    if discriminant<0:return None
    t1=(-b-sqrt(discriminant))/(2*a)
    t2=(-b+sqrt(discriminant))/(2*a)
    return(t1,t2)

def ray_plane_intersection(origin,direction,plane_point,plane_normal):
    denom=dot(direction,plane_normal)
    if abs(denom)<1e-6:return None
    t=dot([plane_point[i]-origin[i]for i in range(3)],plane_normal)/denom
    return t if t>=0 else None

def ray_triangle_intersection(origin,direction,v0,v1,v2):
    edge1=[v1[i]-v0[i]for i in range(3)]
    edge2=[v2[i]-v0[i]for i in range(3)]
    h=cross(direction,edge2)
    a=dot(edge1,h)
    if abs(a)<1e-6:return None
    f=1.0/a
    s=[origin[i]-v0[i]for i in range(3)]
    u=f*dot(s,h)
    if u<0 or u>1:return None
    q=cross(s,edge1)
    v=f*dot(direction,q)
    if v<0 or u+v>1:return None
    t=f*dot(edge2,q)
    return t if t>1e-6 else None

def ray_box_intersection(origin,direction,box_min,box_max):
    tmin,tmax=float('-inf'),float('inf')
    for i in range(3):
        if abs(direction[i])<1e-6:
            if origin[i]<box_min[i]or origin[i]>box_max[i]:return None
        else:
            t1=(box_min[i]-origin[i])/direction[i]
            t2=(box_max[i]-origin[i])/direction[i]
            if t1>t2:t1,t2=t2,t1
            tmin=max(tmin,t1)
            tmax=min(tmax,t2)
            if tmin>tmax:return None
    return(tmin,tmax)if tmin>=0 else None

def reflect_ray(incident,normal):
    return[incident[i]-2*dot(incident,normal)*normal[i]for i in range(3)]

def refract_ray(incident,normal,eta):
    cosi=dot(incident,normal)
    k=1-eta*eta*(1-cosi*cosi)
    if k<0:return None
    return[eta*incident[i]+(eta*cosi-sqrt(k))*normal[i]for i in range(3)]

def path_tracing(origin,direction,max_bounces=5):
    color=[0,0,0]
    throughput=[1,1,1]
    for bounce in range(max_bounces):
        pass
    return color

def raytrace_scene(camera_pos,camera_dir,width,height,fov,scene_objects):
    image=[[0]*width for _ in range(height)]
    aspect=width/height
    for y in range(height):
        for x in range(width):
            u=(2*(x+0.5)/width-1)*aspect*tan(fov/2)
            v=(1-2*(y+0.5)/height)*tan(fov/2)
            ray_dir=normalize([camera_dir[0]+u,camera_dir[1]+v,camera_dir[2]])
            color=[0,0,0]
            for obj in scene_objects:
                pass
            image[y][x]=color
    return image

def monte_carlo_pi(samples=10000):
    import random
    inside=0
    for _ in range(samples):
        x,y=random.random(),random.random()
        if x*x+y*y<=1:inside+=1
    return 4*inside/samples

def buffon_needle(l,d,drops=10000):
    import random
    hits=0
    for _ in range(drops):
        x=random.uniform(0,d/2)
        theta=random.uniform(0,PI/2)
        if x<=l*sin(theta)/2:hits+=1
    return 2*l*drops/(hits*d)if hits>0 else 0

def monte_carlo_integration(f,a,b,n=10000):
    import random
    total=0
    for _ in range(n):total+=f(random.uniform(a,b))
    return(b-a)*total/n

def importance_sampling(f,g,a,b,n=10000):
    import random
    total=0
    for _ in range(n):
        x=random.uniform(a,b)
        total+=f(x)/g(x)
    return total/n

def rejection_sampling(f,M,a,b,n=10000):
    import random
    samples=[]
    while len(samples)<n:
        x=random.uniform(a,b)
        y=random.uniform(0,M)
        if y<=f(x):samples.append(x)
    return samples

def metropolis_hastings(f,x0,n=10000):
    import random
    samples=[x0]
    x=x0
    for _ in range(n-1):
        xnew=x+random.gauss(0,1)
        alpha=min(1,f(xnew)/f(x))
        if random.random()<alpha:x=xnew
        samples.append(x)
    return samples

def gibbs_sampling(f,x0,y0,n=10000):
    import random
    samples=[(x0,y0)]
    x,y=x0,y0
    for _ in range(n-1):
        x=random.gauss(y,1)
        y=random.gauss(x,1)
        samples.append((x,y))
    return samples

def simulated_annealing(f,x0,T0=100,alpha=0.95,max_iter=1000):
    import random
    x=x0
    fx=f(x)
    T=T0
    for _ in range(max_iter):
        xnew=x+random.gauss(0,T)
        fxnew=f(xnew)
        if fxnew<fx or random.random()<exp((fx-fxnew)/T):
            x,fx=xnew,fxnew
        T*=alpha
    return x,fx

def genetic_algorithm(f,pop_size=100,generations=1000,mutation_rate=0.01):
    import random
    population=[random.uniform(-10,10)for _ in range(pop_size)]
    for gen in range(generations):
        fitness=[f(x)for x in population]
        parents=random.choices(population,weights=fitness,k=pop_size//2)
        offspring=[]
        for i in range(0,len(parents),2):
            if i+1<len(parents):
                child=(parents[i]+parents[i+1])/2
                if random.random()<mutation_rate:
                    child+=random.gauss(0,1)
                offspring.append(child)
        population=parents+offspring
    return min(population,key=f)

def particle_swarm(f,n_particles=30,dimensions=2,max_iter=100):
    import random
    particles=[[random.uniform(-10,10)for _ in range(dimensions)]for _ in range(n_particles)]
    velocities=[[0]*dimensions for _ in range(n_particles)]
    pbest=particles[:]
    gbest=min(particles,key=f)
    for _ in range(max_iter):
        for i in range(n_particles):
            for d in range(dimensions):
                r1,r2=random.random(),random.random()
                velocities[i][d]=0.5*velocities[i][d]+2*r1*(pbest[i][d]-particles[i][d])+2*r2*(gbest[d]-particles[i][d])
                particles[i][d]+=velocities[i][d]
            if f(particles[i])<f(pbest[i]):pbest[i]=particles[i][:]
            if f(particles[i])<f(gbest):gbest=particles[i][:]
    return gbest

def ant_colony(distances,n_ants=10,n_iterations=100,alpha=1,beta=2,evaporation=0.5):
    import random
    n_cities=len(distances)
    pheromones=[[1]*n_cities for _ in range(n_cities)]
    best_path=None
    best_distance=float('inf')
    for _ in range(n_iterations):
        paths=[]
        for ant in range(n_ants):
            path=[random.randint(0,n_cities-1)]
            while len(path)<n_cities:
                i=path[-1]
                unvisited=[j for j in range(n_cities)if j not in path]
                if not unvisited:break
                probs=[(pheromones[i][j]**alpha)*(1/distances[i][j]**beta)for j in unvisited]
                total=sum(probs)
                probs=[p/total for p in probs]
                next_city=random.choices(unvisited,weights=probs)[0]
                path.append(next_city)
            paths.append(path)
        for i in range(n_cities):
            for j in range(n_cities):
                pheromones[i][j]*=(1-evaporation)
        for path in paths:
            distance=sum(distances[path[i]][path[i+1]]for i in range(len(path)-1))
            if distance<best_distance:
                best_distance=distance
                best_path=path
            for i in range(len(path)-1):
                pheromones[path[i]][path[i+1]]+=1/distance
    return best_path,best_distance

def differential_evolution(f,bounds,pop_size=20,max_iter=1000,F=0.8,CR=0.9):
    import random
    dimensions=len(bounds)
    population=[[random.uniform(bounds[d][0],bounds[d][1])for d in range(dimensions)]for _ in range(pop_size)]
    for _ in range(max_iter):
        for i in range(pop_size):
            indices=[j for j in range(pop_size)if j!=i]
            a,b,c=random.sample(indices,3)
            mutant=[population[a][d]+F*(population[b][d]-population[c][d])for d in range(dimensions)]
            mutant=[min(max(mutant[d],bounds[d][0]),bounds[d][1])for d in range(dimensions)]
            trial=[]
            for d in range(dimensions):
                if random.random()<CR or d==random.randint(0,dimensions-1):
                    trial.append(mutant[d])
                else:
                    trial.append(population[i][d])
            if f(trial)<f(population[i]):
                population[i]=trial
    return min(population,key=f)

def harmony_search(f,bounds,hms=30,hmcr=0.9,par=0.3,max_iter=1000):
    import random
    dimensions=len(bounds)
    harmony_memory=[[random.uniform(bounds[d][0],bounds[d][1])for d in range(dimensions)]for _ in range(hms)]
    for _ in range(max_iter):
        new_harmony=[]
        for d in range(dimensions):
            if random.random()<hmcr:
                new_harmony.append(random.choice([h[d]for h in harmony_memory]))
                if random.random()<par:
                    new_harmony[-1]+=random.uniform(-1,1)
                new_harmony[-1]=min(max(new_harmony[-1],bounds[d][0]),bounds[d][1])
            else:
                new_harmony.append(random.uniform(bounds[d][0],bounds[d][1]))
        worst_idx=max(range(hms),key=lambda i:f(harmony_memory[i]))
        if f(new_harmony)<f(harmony_memory[worst_idx]):
            harmony_memory[worst_idx]=new_harmony
    return min(harmony_memory,key=f)

def firefly_algorithm(f,bounds,n_fireflies=25,max_iter=100,alpha=0.5,beta0=1,gamma=1):
    import random
    dimensions=len(bounds)
    fireflies=[[random.uniform(bounds[d][0],bounds[d][1])for d in range(dimensions)]for _ in range(n_fireflies)]
    for _ in range(max_iter):
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if f(fireflies[j])<f(fireflies[i]):
                    r=dist(fireflies[i],fireflies[j])
                    beta=beta0*exp(-gamma*r*r)
                    for d in range(dimensions):
                        fireflies[i][d]+=beta*(fireflies[j][d]-fireflies[i][d])+alpha*random.uniform(-1,1)
                        fireflies[i][d]=min(max(fireflies[i][d],bounds[d][0]),bounds[d][1])
    return min(fireflies,key=f)

def cuckoo_search(f,bounds,n_nests=25,max_iter=1000,pa=0.25):
    import random
    dimensions=len(bounds)
    nests=[[random.uniform(bounds[d][0],bounds[d][1])for d in range(dimensions)]for _ in range(n_nests)]
    for _ in range(max_iter):
        i=random.randint(0,n_nests-1)
        step=[random.gauss(0,1)for _ in range(dimensions)]
        new_nest=[nests[i][d]+step[d]for d in range(dimensions)]
        new_nest=[min(max(new_nest[d],bounds[d][0]),bounds[d][1])for d in range(dimensions)]
        j=random.randint(0,n_nests-1)
        if f(new_nest)<f(nests[j]):nests[j]=new_nest
        for i in range(n_nests):
            if random.random()<pa:
                nests[i]=[random.uniform(bounds[d][0],bounds[d][1])for d in range(dimensions)]
    return min(nests,key=f)

def bat_algorithm(f,bounds,n_bats=40,max_iter=1000,alpha=0.9,gamma=0.9):
    import random
    dimensions=len(bounds)
    bats=[[random.uniform(bounds[d][0],bounds[d][1])for d in range(dimensions)]for _ in range(n_bats)]
    velocities=[[0]*dimensions for _ in range(n_bats)]
    frequencies=[0]*n_bats
    loudness=[1]*n_bats
    pulse_rate=[0]*n_bats
    best=min(bats,key=f)
    for _ in range(max_iter):
        for i in range(n_bats):
            frequencies[i]=random.uniform(0,2)
            for d in range(dimensions):
                velocities[i][d]+=frequencies[i]*(bats[i][d]-best[d])
                bats[i][d]+=velocities[i][d]
                bats[i][d]=min(max(bats[i][d],bounds[d][0]),bounds[d][1])
            if random.random()>pulse_rate[i]:
                for d in range(dimensions):
                    bats[i][d]=best[d]+random.gauss(0,1)*sum(loudness)/n_bats
            if f(bats[i])<f(best)and random.random()<loudness[i]:
                best=bats[i][:]
                loudness[i]*=alpha
                pulse_rate[i]=1-exp(-gamma*_)
    return best

def whale_optimization(f,bounds,n_whales=30,max_iter=1000):
    import random
    dimensions=len(bounds)
    whales=[[random.uniform(bounds[d][0],bounds[d][1])for d in range(dimensions)]for _ in range(n_whales)]
    best=min(whales,key=f)
    for t in range(max_iter):
        a=2-t*2/max_iter
        for i in range(n_whales):
            r=random.random()
            if r<0.5:
                if abs(a)<1:
                    for d in range(dimensions):
                        D=abs(a*best[d]-whales[i][d])
                        whales[i][d]=best[d]-a*D
                else:
                    rand_whale=random.choice(whales)
                    for d in range(dimensions):
                        D=abs(a*rand_whale[d]-whales[i][d])
                        whales[i][d]=rand_whale[d]-a*D
            else:
                for d in range(dimensions):
                    distance=abs(best[d]-whales[i][d])
                    whales[i][d]=distance*exp(random.uniform(-1,1)*2*PI)*cos(random.uniform(-1,1)*2*PI)+best[d]
            for d in range(dimensions):
                whales[i][d]=min(max(whales[i][d],bounds[d][0]),bounds[d][1])
            if f(whales[i])<f(best):best=whales[i][:]
    return best

def grey_wolf_optimizer(f,bounds,n_wolves=30,max_iter=1000):
    import random
    dimensions=len(bounds)
    wolves=[[random.uniform(bounds[d][0],bounds[d][1])for d in range(dimensions)]for _ in range(n_wolves)]
    wolves.sort(key=f)
    alpha,beta,delta=wolves[0],wolves[1],wolves[2]
    for t in range(max_iter):
        a=2-t*2/max_iter
        for i in range(n_wolves):
            for d in range(dimensions):
                r1,r2=random.random(),random.random()
                A1=2*a*r1-a
                C1=2*r2
                D_alpha=abs(C1*alpha[d]-wolves[i][d])
                X1=alpha[d]-A1*D_alpha
                r1,r2=random.random(),random.random()
                A2=2*a*r1-a
                C2=2*r2
                D_beta=abs(C2*beta[d]-wolves[i][d])
                X2=beta[d]-A2*D_beta
                r1,r2=random.random(),random.random()
                A3=2*a*r1-a
                C3=2*r2
                D_delta=abs(C3*delta[d]-wolves[i][d])
                X3=delta[d]-A3*D_delta
                wolves[i][d]=(X1+X2+X3)/3
                wolves[i][d]=min(max(wolves[i][d],bounds[d][0]),bounds[d][1])
        wolves.sort(key=f)
        alpha,beta,delta=wolves[0],wolves[1],wolves[2]
    return alpha

def artificial_bee_colony(f,bounds,n_bees=50,max_iter=1000,limit=100):
    import random
    dimensions=len(bounds)
    n_employed=n_onlooker=n_bees//2
    food_sources=[[random.uniform(bounds[d][0],bounds[d][1])for d in range(dimensions)]for _ in range(n_employed)]
    trial=[0]*n_employed
    for _ in range(max_iter):
        for i in range(n_employed):
            k=random.choice([j for j in range(n_employed)if j!=i])
            new_source=food_sources[i][:]
            j=random.randint(0,dimensions-1)
            new_source[j]=food_sources[i][j]+(random.random()-0.5)*2*(food_sources[i][j]-food_sources[k][j])
            new_source[j]=min(max(new_source[j],bounds[j][0]),bounds[j][1])
            if f(new_source)<f(food_sources[i]):
                food_sources[i]=new_source
                trial[i]=0
            else:
                trial[i]+=1
        fitness=[1/(1+f(fs))for fs in food_sources]
        total_fit=sum(fitness)
        probs=[fit/total_fit for fit in fitness]
        for _ in range(n_onlooker):
            i=random.choices(range(n_employed),weights=probs)[0]
            k=random.choice([j for j in range(n_employed)if j!=i])
            new_source=food_sources[i][:]
            j=random.randint(0,dimensions-1)
            new_source[j]=food_sources[i][j]+(random.random()-0.5)*2*(food_sources[i][j]-food_sources[k][j])
            new_source[j]=min(max(new_source[j],bounds[j][0]),bounds[j][1])
            if f(new_source)<f(food_sources[i]):
                food_sources[i]=new_source
                trial[i]=0
            else:
                trial[i]+=1
        for i in range(n_employed):
            if trial[i]>limit:
                food_sources[i]=[random.uniform(bounds[d][0],bounds[d][1])for d in range(dimensions)]
                trial[i]=0
    return min(food_sources,key=f)

def teaching_learning_based(f,bounds,n_learners=40,max_iter=1000):
    import random
    dimensions=len(bounds)
    learners=[[random.uniform(bounds[d][0],bounds[d][1])for d in range(dimensions)]for _ in range(n_learners)]
    for _ in range(max_iter):
        teacher=min(learners,key=f)
        mean=[sum(l[d]for l in learners)/n_learners for d in range(dimensions)]
        for i in range(n_learners):
            TF=random.choice([1,2])
            new_learner=[]
            for d in range(dimensions):
                new_learner.append(learners[i][d]+random.random()*(teacher[d]-TF*mean[d]))
                new_learner[-1]=min(max(new_learner[-1],bounds[d][0]),bounds[d][1])
            if f(new_learner)<f(learners[i]):learners[i]=new_learner
        for i in range(n_learners):
            j=random.choice([k for k in range(n_learners)if k!=i])
            new_learner=[]
            for d in range(dimensions):
                if f(learners[i])<f(learners[j]):
                    new_learner.append(learners[i][d]+random.random()*(learners[i][d]-learners[j][d]))
                else:
                    new_learner.append(learners[i][d]+random.random()*(learners[j][d]-learners[i][d]))
                new_learner[-1]=min(max(new_learner[-1],bounds[d][0]),bounds[d][1])
            if f(new_learner)<f(learners[i]):learners[i]=new_learner
    return min(learners,key=f)

def gravitational_search(f,bounds,n_agents=30,max_iter=1000,G0=100,alpha=20):
    import random
    dimensions=len(bounds)
    agents=[[random.uniform(bounds[d][0],bounds[d][1])for d in range(dimensions)]for _ in range(n_agents)]
    velocities=[[0]*dimensions for _ in range(n_agents)]
    for t in range(max_iter):
        G=G0*exp(-alpha*t/max_iter)
        masses=[1/(1+f(agent))for agent in agents]
        total_mass=sum(masses)
        masses=[m/total_mass for m in masses]
        for i in range(n_agents):
            force=[0]*dimensions
            for j in range(n_agents):
                if i!=j:
                    r=dist(agents[i],agents[j])+1e-10
                    for d in range(dimensions):
                        force[d]+=random.random()*masses[j]*(agents[j][d]-agents[i][d])/r
            for d in range(dimensions):
                acceleration=G*force[d]/masses[i]
                velocities[i][d]=random.random()*velocities[i][d]+acceleration
                agents[i][d]+=velocities[i][d]
                agents[i][d]=min(max(agents[i][d],bounds[d][0]),bounds[d][1])
    return min(agents,key=f)

def chaos_optimization(f,bounds,n_agents=30,max_iter=1000):
    import random
    dimensions=len(bounds)
    agents=[[random.uniform(bounds[d][0],bounds[d][1])for d in range(dimensions)]for _ in range(n_agents)]
    chaos=[random.random()for _ in range(n_agents)]
    for t in range(max_iter):
        for i in range(n_agents):
            chaos[i]=4*chaos[i]*(1-chaos[i])
            for d in range(dimensions):
                agents[i][d]=(bounds[d][0]+bounds[d][1])/2+chaos[i]*(bounds[d][1]-bounds[d][0])/2
        best_idx=min(range(n_agents),key=lambda i:f(agents[i]))
        for i in range(n_agents):
            for d in range(dimensions):
                agents[i][d]+=(agents[best_idx][d]-agents[i][d])*random.random()
                agents[i][d]=min(max(agents[i][d],bounds[d][0]),bounds[d][1])
    return min(agents,key=f)

def spiral_optimization(f,bounds,n_agents=30,max_iter=1000,r=0.95,theta_max=5*PI):
    import random
    dimensions=len(bounds)
    agents=[[random.uniform(bounds[d][0],bounds[d][1])for d in range(dimensions)]for _ in range(n_agents)]
    for t in range(max_iter):
        best=min(agents,key=f)
        theta=theta_max-t*theta_max/max_iter
        for i in range(n_agents):
            for d in range(dimensions):
                x=agents[i][d]-best[d]
                agents[i][d]=best[d]+r*x*cos(theta)-r*x*sin(theta)
                agents[i][d]=min(max(agents[i][d],bounds[d][0]),bounds[d][1])
    return min(agents,key=f)

def flower_pollination(f,bounds,n_flowers=25,max_iter=1000,p=0.8):
    import random
    dimensions=len(bounds)
    flowers=[[random.uniform(bounds[d][0],bounds[d][1])for d in range(dimensions)]for _ in range(n_flowers)]
    best=min(flowers,key=f)
    for _ in range(max_iter):
        for i in range(n_flowers):
            if random.random()<p:
                L=[random.gauss(0,1)for _ in range(dimensions)]
                for d in range(dimensions):
                    flowers[i][d]+=L[d]*(best[d]-flowers[i][d])
            else:
                j,k=random.sample(range(n_flowers),2)
                for d in range(dimensions):
                    flowers[i][d]+=random.random()*(flowers[j][d]-flowers[k][d])
            for d in range(dimensions):
                flowers[i][d]=min(max(flowers[i][d],bounds[d][0]),bounds[d][1])
            if f(flowers[i])<f(best):best=flowers[i][:]
    return best

def krill_herd(f,bounds,n_krills=40,max_iter=1000,Nmax=0.01,Vf=0.02,Dmax=0.002,Ct=0.5):
    import random
    dimensions=len(bounds)
    krills=[[random.uniform(bounds[d][0],bounds[d][1])for d in range(dimensions)]for _ in range(n_krills)]
    for _ in range(max_iter):
        best=min(krills,key=f)
        worst=max(krills,key=f)
        for i in range(n_krills):
            alpha=sum([random.random()*(krills[j][d]-krills[i][d])/dist(krills[i],krills[j])for j in range(n_krills)if j!=i and f(krills[j])<f(krills[i])]for d in range(dimensions))
            N=Nmax*alpha+random.random()
            C=2*random.random()*sum([(best[d]-krills[i][d])for d in range(dimensions)])
            D=Dmax*random.gauss(0,1)
            for d in range(dimensions):
                krills[i][d]+=Ct*(N+Vf*C+D)
                krills[i][d]=min(max(krills[i][d],bounds[d][0]),bounds[d][1])
    return min(krills,key=f)

def moth_flame(f,bounds,n_moths=30,max_iter=1000):
    import random
    dimensions=len(bounds)
    moths=[[random.uniform(bounds[d][0],bounds[d][1])for d in range(dimensions)]for _ in range(n_moths)]
    for t in range(max_iter):
        flames=sorted(moths,key=f)[:n_moths]
        b=-1+t*(-1)/max_iter
        for i in range(n_moths):
            for d in range(dimensions):
                Di=abs(flames[i][d]-moths[i][d])
                moths[i][d]=Di*exp(b*t)*cos(t*2*PI)+flames[i][d]
                moths[i][d]=min(max(moths[i][d],bounds[d][0]),bounds[d][1])
    return min(moths,key=f)

def dragonfly(f,bounds,n_dragonflies=30,max_iter=1000):
    import random
    dimensions=len(bounds)
    dragonflies=[[random.uniform(bounds[d][0],bounds[d][1])for d in range(dimensions)]for _ in range(n_dragonflies)]
    velocities=[[0]*dimensions for _ in range(n_dragonflies)]
    for t in range(max_iter):
        best=min(dragonflies,key=f)
        for i in range(n_dragonflies):
            neighbors=[j for j in range(n_dragonflies)if dist(dragonflies[i],dragonflies[j])<5]
            if neighbors:
                S=[sum(dragonflies[i][d]-dragonflies[j][d]for j in neighbors)for d in range(dimensions)]
                A=[sum(dragonflies[j][d]for j in neighbors)/len(neighbors)-dragonflies[i][d]for d in range(dimensions)]
                C=[sum(velocities[j][d]for j in neighbors)/len(neighbors)-velocities[i][d]for d in range(dimensions)]
                F=[best[d]-dragonflies[i][d]for d in range(dimensions)]
                E=[random.uniform(-1,1)for _ in range(dimensions)]
                for d in range(dimensions):
                    velocities[i][d]=0.1*S[d]+0.2*A[d]+0.3*C[d]+0.4*F[d]+0.1*E[d]
                    dragonflies[i][d]+=velocities[i][d]
                    dragonflies[i][d]=min(max(dragonflies[i][d],bounds[d][0]),bounds[d][1])
    return min(dragonflies,key=f)

def salp_swarm(f,bounds,n_salps=30,max_iter=1000):
    import random
    dimensions=len(bounds)
    salps=[[random.uniform(bounds[d][0],bounds[d][1])for d in range(dimensions)]for _ in range(n_salps)]
    for t in range(max_iter):
        food=min(salps,key=f)
        c1=2*exp(-(4*t/max_iter)**2)
        for i in range(n_salps):
            if i==0:
                for d in range(dimensions):
                    c2,c3=random.random(),random.random()
                    if c3<0.5:
                        salps[i][d]=food[d]+c1*((bounds[d][1]-bounds[d][0])*c2+bounds[d][0])
                    else:
                        salps[i][d]=food[d]-c1*((bounds[d][1]-bounds[d][0])*c2+bounds[d][0])
            else:
                for d in range(dimensions):
                    salps[i][d]=0.5*(salps[i][d]+salps[i-1][d])
            for d in range(dimensions):
                salps[i][d]=min(max(salps[i][d],bounds[d][0]),bounds[d][1])
    return min(salps,key=f)

def grasshopper_optimization(f,bounds,n_grasshoppers=30,max_iter=1000):
    import random
    dimensions=len(bounds)
    grasshoppers=[[random.uniform(bounds[d][0],bounds[d][1])for d in range(dimensions)]for _ in range(n_grasshoppers)]
    for t in range(max_iter):
        target=min(grasshoppers,key=f)
        c=1-t/max_iter
        for i in range(n_grasshoppers):
            for d in range(dimensions):
                s=sum([c*(bounds[d][1]-bounds[d][0])/2*((grasshoppers[i][d]-grasshoppers[j][d])/dist(grasshoppers[i],grasshoppers[j])if dist(grasshoppers[i],grasshoppers[j])>0 else 0)for j in range(n_grasshoppers)if j!=i])
                grasshoppers[i][d]=c*s+target[d]
                grasshoppers[i][d]=min(max(grasshoppers[i][d],bounds[d][0]),bounds[d][1])
    return min(grasshoppers,key=f)

def harris_hawks(f,bounds,n_hawks=30,max_iter=1000):
    import random
    dimensions=len(bounds)
    hawks=[[random.uniform(bounds[d][0],bounds[d][1])for d in range(dimensions)]for _ in range(n_hawks)]
    for t in range(max_iter):
        rabbit=min(hawks,key=f)
        E=2*random.random()-1
        for i in range(n_hawks):
            r=random.random()
            if abs(E)>=1:
                if r>=0.5:
                    rand_hawk=random.choice(hawks)
                    for d in range(dimensions):
                        hawks[i][d]=rand_hawk[d]-random.random()*abs(rand_hawk[d]-2*random.random()*hawks[i][d])
                else:
                    for d in range(dimensions):
                        hawks[i][d]=(rabbit[d]-sum(hawks)/n_hawks)-random.random()*(bounds[d][0]+random.random()*(bounds[d][1]-bounds[d][0]))
            else:
                if r>=0.5 and abs(E)>=0.5:
                    for d in range(dimensions):
                        hawks[i][d]=rabbit[d]-E*abs(rabbit[d]-hawks[i][d])
                elif r>=0.5 and abs(E)<0.5:
                    for d in range(dimensions):
                        hawks[i][d]=rabbit[d]-E*abs(rabbit[d]-hawks[i][d])-random.gauss(0,1)
            for d in range(dimensions):
                hawks[i][d]=min(max(hawks[i][d],bounds[d][0]),bounds[d][1])
    return min(hawks,key=f)

print("Engineering Mathematics Library Loaded Successfully")
print(f"Total Functions: {len([x for x in dir()if callable(globals()[x])and not x.startswith('_')])}")
print("Modules: Calculus, LinearAlgebra, NumberTheory, Geometry, Optimization, NumericalAnalysis")
print("Developer: mero | Telegram: @QP4RM")

def sigmoid(x):return 1/(1+exp(-x))
def relu(x):return max(0,x)
def leaky_relu(x,alpha=0.01):return x if x>0 else alpha*x
def elu(x,alpha=1):return x if x>0 else alpha*(exp(x)-1)
def selu(x,alpha=1.67326,scale=1.0507):return scale*(x if x>0 else alpha*(exp(x)-1))
def softplus(x):return log(1+exp(x))
def swish(x):return x*sigmoid(x)
def mish(x):return x*tanh(softplus(x))
def gelu(x):return 0.5*x*(1+tanh(sqrt(2/PI)*(x+0.044715*x**3)))
def softmax(x):e=exp(x-max(x));return e/sum(e)
def log_softmax(x):return x-log(sum(exp(x)))
def cross_entropy(y_true,y_pred):return-sum(y_true*log(y_pred))
def binary_cross_entropy(y_true,y_pred):return-sum(y_true*log(y_pred)+(1-y_true)*log(1-y_pred))
def categorical_cross_entropy(y_true,y_pred):return cross_entropy(y_true,y_pred)
def mean_squared_error(y_true,y_pred):return sum((y_true-y_pred)**2)/len(y_true)
def mean_absolute_error(y_true,y_pred):return sum(abs(y_true-y_pred))/len(y_true)
def huber_loss(y_true,y_pred,delta=1.0):a=abs(y_true-y_pred);return sum(0.5*a**2 if a<=delta else delta*(a-0.5*delta))
def hinge_loss(y_true,y_pred):return max(0,1-y_true*y_pred)
def squared_hinge_loss(y_true,y_pred):return max(0,1-y_true*y_pred)**2
def kullback_leibler(p,q):return sum(p*log(p/q))
def jensen_shannon(p,q):m=0.5*(p+q);return 0.5*kullback_leibler(p,m)+0.5*kullback_leibler(q,m)
def wasserstein_distance(p,q):return sum(abs(p-q))
def bhattacharyya_distance(p,q):return-log(sum(sqrt(p*q)))
def hellinger_distance(p,q):return sqrt(0.5*sum((sqrt(p)-sqrt(q))**2))
def total_variation_distance(p,q):return 0.5*sum(abs(p-q))
def chi_squared_distance(p,q):return sum((p-q)**2/q)
def cosine_similarity(a,b):return dot(a,b)/(norm(a)*norm(b))
def euclidean_distance(a,b):return sqrt(sum((a-b)**2))
def manhattan_distance(a,b):return sum(abs(a-b))
def chebyshev_distance(a,b):return max(abs(a-b))
def minkowski_distance(a,b,p=2):return sum(abs(a-b)**p)**(1/p)
def hamming_distance(a,b):return sum(1 for x,y in zip(a,b)if x!=y)
def jaccard_similarity(a,b):return len(set(a)&set(b))/len(set(a)|set(b))
def dice_coefficient(a,b):return 2*len(set(a)&set(b))/(len(set(a))+len(set(b)))
def overlap_coefficient(a,b):return len(set(a)&set(b))/min(len(set(a)),len(set(b)))
def pearson_correlation(x,y):mx,my=sum(x)/len(x),sum(y)/len(y);return sum((x-mx)*(y-my))/sqrt(sum((x-mx)**2)*sum((y-my)**2))
def spearman_correlation(x,y):rx=[sorted(x).index(v)+1 for v in x];ry=[sorted(y).index(v)+1 for v in y];return pearson_correlation(rx,ry)
def kendall_tau(x,y):n=len(x);c=d=0;[c:=c+1 if(x[i]-x[j])*(y[i]-y[j])>0 else d:=d+1 for i in range(n)for j in range(i+1,n)];return(c-d)/(0.5*n*(n-1))
def covariance(x,y):mx,my=sum(x)/len(x),sum(y)/len(y);return sum((x-mx)*(y-my))/(len(x)-1)
def variance(x):m=sum(x)/len(x);return sum((x-m)**2)/(len(x)-1)
def standard_deviation(x):return sqrt(variance(x))
def skewness(x):m=sum(x)/len(x);s=standard_deviation(x);return sum(((x-m)/s)**3)/len(x)
def kurtosis(x):m=sum(x)/len(x);s=standard_deviation(x);return sum(((x-m)/s)**4)/len(x)-3
def median(x):s=sorted(x);n=len(s);return s[n//2]if n%2 else(s[n//2-1]+s[n//2])/2
def mode(x):from collections import Counter;c=Counter(x);return c.most_common(1)[0][0]
def quartiles(x):s=sorted(x);n=len(s);return(s[n//4],median(x),s[3*n//4])
def interquartile_range(x):q1,_,q3=quartiles(x);return q3-q1
def percentile(x,p):s=sorted(x);k=(len(s)-1)*p/100;f,c=int(k),int(k)+1;return s[f]+(s[c]-s[f])*(k-f)if c<len(s)else s[f]
def z_score(x,mu=None,sigma=None):mu=mu or sum(x)/len(x);sigma=sigma or standard_deviation(x);return(x-mu)/sigma
def t_statistic(x,y):mx,my=sum(x)/len(x),sum(y)/len(y);sx,sy=standard_deviation(x),standard_deviation(y);return(mx-my)/sqrt(sx**2/len(x)+sy**2/len(y))
def chi_squared_test(observed,expected):return sum((o-e)**2/e for o,e in zip(observed,expected))
def f_statistic(x,y):return variance(x)/variance(y)
def anova(groups):k=len(groups);n=sum(len(g)for g in groups);grand_mean=sum(sum(g)for g in groups)/n;ssb=sum(len(g)*(sum(g)/len(g)-grand_mean)**2 for g in groups);ssw=sum(sum((x-sum(g)/len(g))**2 for x in g)for g in groups);msb,msw=ssb/(k-1),ssw/(n-k);return msb/msw
def linear_regression(x,y):n=len(x);sx,sy=sum(x),sum(y);sxx,sxy=sum(xi**2 for xi in x),sum(xi*yi for xi,yi in zip(x,y));b=(n*sxy-sx*sy)/(n*sxx-sx**2);a=(sy-b*sx)/n;return a,b
def polynomial_regression(x,y,degree):X=[[xi**d for d in range(degree+1)]for xi in x];XTX=matrix_mult(matrix_transpose(X),X);XTy=[sum(X[i][j]*y[i]for i in range(len(X)))for j in range(len(X[0]))];return gauss_elim(XTX,XTy)
def logistic_regression(x,y,lr=0.01,epochs=1000):w=[0]*len(x[0]);b=0;[[[w:=[w[j]-lr*(sigmoid(sum(w[j]*x[i][j]for j in range(len(w)))+b)-y[i])*x[i][j]for j in range(len(w))],b:=b-lr*(sigmoid(sum(w[j]*x[i][j]for j in range(len(w)))+b)-y[i])]for i in range(len(x))]for _ in range(epochs)];return w,b
def ridge_regression(x,y,alpha=1.0):X=matrix_transpose(x);I=[[alpha if i==j else 0 for j in range(len(X))]for i in range(len(X))];A=matrix_add(matrix_mult(X,x),I);b=[sum(X[i][j]*y[j]for j in range(len(y)))for i in range(len(X))];return gauss_elim(A,b)
def lasso_regression(x,y,alpha=1.0,lr=0.01,epochs=1000):w=[0]*len(x[0]);[[w:=[w[j]-lr*(sum((sum(w[k]*x[i][k]for k in range(len(w)))-y[i])*x[i][j]for i in range(len(x)))+alpha*(-1 if w[j]<0 else 1))for j in range(len(w))]for _ in range(epochs)]];return w
def elastic_net(x,y,alpha=1.0,l1_ratio=0.5,lr=0.01,epochs=1000):w=[0]*len(x[0]);l1,l2=alpha*l1_ratio,alpha*(1-l1_ratio);[[w:=[w[j]-lr*(sum((sum(w[k]*x[i][k]for k in range(len(w)))-y[i])*x[i][j]for i in range(len(x)))+l1*(-1 if w[j]<0 else 1)+l2*w[j])for j in range(len(w))]for _ in range(epochs)]];return w
def k_nearest_neighbors(x_train,y_train,x_test,k=3):dists=[(dist(x_test,x),y)for x,y in zip(x_train,y_train)];neighbors=sorted(dists,key=lambda d:d[0])[:k];return sum(n[1]for n in neighbors)/k
def naive_bayes(x_train,y_train,x_test):classes=set(y_train);probs={};[probs.__setitem__(c,sum(1 for y in y_train if y==c)/len(y_train)*__import__('functools').reduce(lambda a,b:a*b,[1/(sqrt(2*PI)*standard_deviation([x[i]for x,y in zip(x_train,y_train)if y==c]))*exp(-((x_test[i]-sum(x[i]for x,y in zip(x_train,y_train)if y==c)/sum(1 for y in y_train if y==c))**2)/(2*variance([x[i]for x,y in zip(x_train,y_train)if y==c])))for i in range(len(x_test))],1))for c in classes];return max(probs,key=probs.get)
def decision_tree_gini(y):n=len(y);return 1-sum((sum(1 for yi in y if yi==c)/n)**2 for c in set(y))
def decision_tree_entropy(y):n=len(y);return-sum((sum(1 for yi in y if yi==c)/n)*log(sum(1 for yi in y if yi==c)/n)for c in set(y)if sum(1 for yi in y if yi==c)>0)
def decision_tree_split(x,y,feature):values=set(x[i][feature]for i in range(len(x)));return min([(v,sum([decision_tree_gini([y[j]for j in range(len(y))if x[j][feature]==v])*sum(1 for j in range(len(y))if x[j][feature]==v)/len(y),decision_tree_gini([y[j]for j in range(len(y))if x[j][feature]!=v])*sum(1 for j in range(len(y))if x[j][feature]!=v)/len(y)]))for v in values],key=lambda s:s[1])
def random_forest(x_train,y_train,x_test,n_trees=10):predictions=[];[predictions.append(k_nearest_neighbors(x_train,y_train,x_test,k=3))for _ in range(n_trees)];return sum(predictions)/len(predictions)
def adaboost(x_train,y_train,x_test,n_estimators=50):w=[1/len(y_train)]*len(y_train);alphas=[];stumps=[];[[alphas.append(0.5*log((1-e)/e)if(e:=sum(w[i]for i in range(len(y_train))if y_train[i]!=(s:=k_nearest_neighbors(x_train,y_train,x_train[i],k=1))))>0 and e<1 else 1),stumps.append(s),w:=[w[i]*exp(-alphas[-1]*y_train[i]*s)if y_train[i]==s else w[i]*exp(alphas[-1]*y_train[i]*s)for i in range(len(w))],w:=[w[i]/sum(w)for i in range(len(w))]]for _ in range(n_estimators)];return sum(alphas[i]*k_nearest_neighbors(x_train,y_train,x_test,k=1)for i in range(len(alphas)))
def gradient_boosting(x_train,y_train,x_test,n_estimators=100,lr=0.1):pred=[sum(y_train)/len(y_train)]*len(y_train);[[residuals:=[y_train[i]-pred[i]for i in range(len(y_train))],model:=linear_regression([i for i in range(len(residuals))],residuals),pred:=[pred[i]+lr*model[1]*i for i in range(len(pred))]]for _ in range(n_estimators)];return sum(y_train)/len(y_train)+sum([lr*linear_regression([i for i in range(len(y_train))],[y_train[i]-sum(y_train)/len(y_train)for i in range(len(y_train))])[1]*len(x_train)for _ in range(n_estimators)])
def xgboost_approx(x_train,y_train,x_test,n_estimators=100,lr=0.1,max_depth=3):pred=[sum(y_train)/len(y_train)]*len(y_train);trees=[];[[residuals:=[y_train[i]-pred[i]for i in range(len(y_train))],tree:=linear_regression([i for i in range(len(residuals))],residuals),trees.append(tree),pred:=[pred[i]+lr*tree[1]*i for i in range(len(pred))]]for _ in range(n_estimators)];return sum(y_train)/len(y_train)+sum([lr*trees[i][1]*len(x_train)for i in range(len(trees))])
def support_vector_machine(x_train,y_train,x_test,C=1.0,lr=0.01,epochs=1000):w=[0]*len(x_train[0]);b=0;[[[[w:=[w[j]-lr*(w[j]-C*y_train[i]*x_train[i][j])for j in range(len(w))],b:=b-lr*(-C*y_train[i])]if y_train[i]*(sum(w[j]*x_train[i][j]for j in range(len(w)))+b)<1 else[w:=[w[j]-lr*w[j]for j in range(len(w))],b:=b]]for i in range(len(x_train))]for _ in range(epochs)]];return 1 if sum(w[i]*x_test[i]for i in range(len(w)))+b>=0 else-1
def k_means(x,k=3,max_iter=100):import random;centroids=random.sample(x,k);[[clusters:=[[xi for xi in x if min(range(k),key=lambda j:dist(xi,centroids[j]))==i]for i in range(k)],centroids:=[[sum(p[d]for p in clusters[i])/len(clusters[i])if clusters[i]else centroids[i][d]for d in range(len(x[0]))]for i in range(k)]]for _ in range(max_iter)];return centroids
def dbscan(x,eps=0.5,min_samples=5):labels=[-1]*len(x);cluster_id=0;[[[[labels.__setitem__(i,cluster_id),queue:=[i],[[labels.__setitem__(n,cluster_id),queue.append(n)]if labels[n]==-1 else None for n in[j for j in range(len(x))if dist(x[i],x[j])<eps]while queue and(i:=queue.pop(0))]]]if labels[i]==-1 and len([j for j in range(len(x))if dist(x[i],x[j])<eps])>=min_samples else None,cluster_id:=cluster_id+1 if labels[i]!=-1 else cluster_id]for i in range(len(x))]];return labels
def hierarchical_clustering(x,n_clusters=3):dists=[[dist(x[i],x[j])if i!=j else float('inf')for j in range(len(x))]for i in range(len(x))];clusters=[[i]for i in range(len(x))];[[[i,j:=min([(i,j)for i in range(len(clusters))for j in range(i+1,len(clusters))],key=lambda p:min(dists[ci][cj]for ci in clusters[p[0]]for cj in clusters[p[1]])),clusters:=[clusters[i]+clusters[j]if k==i else clusters[k]if k<j else clusters[k+1]for k in range(len(clusters))if k!=j]]for _ in range(len(x)-n_clusters)]];return clusters
def gaussian_mixture(x,k=3,max_iter=100):import random;means=random.sample(x,k);covs=[1]*k;weights=[1/k]*k;[[resp:=[[weights[j]*exp(-0.5*sum((x[i][d]-means[j][d])**2/covs[j]for d in range(len(x[0]))))/sum(weights[l]*exp(-0.5*sum((x[i][d]-means[l][d])**2/covs[l]for d in range(len(x[0]))))for l in range(k))for j in range(k)]for i in range(len(x))],weights:=[sum(resp[i][j]for i in range(len(x)))/len(x)for j in range(k)],means:=[[sum(resp[i][j]*x[i][d]for i in range(len(x)))/sum(resp[i][j]for i in range(len(x)))for d in range(len(x[0]))]for j in range(k)],covs:=[sum(resp[i][j]*sum((x[i][d]-means[j][d])**2 for d in range(len(x[0])))for i in range(len(x)))/sum(resp[i][j]for i in range(len(x)))for j in range(k)]]for _ in range(max_iter)];return means,covs,weights
def pca(x,n_components=2):mean=[sum(x[i][j]for i in range(len(x)))/len(x)for j in range(len(x[0]))];centered=[[x[i][j]-mean[j]for j in range(len(x[0]))]for i in range(len(x))];cov=[[sum(centered[k][i]*centered[k][j]for k in range(len(centered)))/(len(centered)-1)for j in range(len(x[0]))]for i in range(len(x[0]))];eigvals=eigenvalues(cov);indices=sorted(range(len(eigvals)),key=lambda i:eigvals[i],reverse=True)[:n_components];return[[centered[i][j]for j in indices]for i in range(len(centered))]
def lda(x,y,n_components=1):classes=list(set(y));means={c:[sum(x[i][j]for i in range(len(x))if y[i]==c)/sum(1 for yi in y if yi==c)for j in range(len(x[0]))]for c in classes};overall_mean=[sum(x[i][j]for i in range(len(x)))/len(x)for j in range(len(x[0]))];Sb=sum([sum(1 for yi in y if yi==c)*sum((means[c][i]-overall_mean[i])*(means[c][j]-overall_mean[j])for i in range(len(x[0])))for c in classes]for j in range(len(x[0])));Sw=sum([sum((x[i][j]-means[y[i]][j])**2 for i in range(len(x))if y[i]==c)for c in classes]for j in range(len(x[0])));return Sb/Sw
def tsne(x,n_components=2,perplexity=30,lr=200,n_iter=1000):import random;y=[[random.gauss(0,1)for _ in range(n_components)]for _ in range(len(x))];[[dy:=[[0]*n_components for _ in range(len(y))],[[dy.__setitem__(i,[dy[i][d]+4*(x_dist:=exp(-dist(x[i],x[j])))*(y_dist:=1/(1+dist(y[i],y[j])**2))*(y[i][d]-y[j][d])for d in range(n_components)])for j in range(len(y))if i!=j]for i in range(len(y))],y:=[[y[i][d]+lr*dy[i][d]for d in range(n_components)]for i in range(len(y))]]for _ in range(n_iter)];return y
def autoencoder(x,hidden_dim=2,lr=0.01,epochs=1000):import random;We=[random.gauss(0,0.1)for _ in range(len(x[0])*hidden_dim)];Wd=[random.gauss(0,0.1)for _ in range(hidden_dim*len(x[0]))];[[h:=sigmoid(sum(We[i*hidden_dim+j]*x[k][i]for i in range(len(x[0])))for j in range(hidden_dim)),xr:=[sigmoid(sum(Wd[j*len(x[0])+i]*h[j]for j in range(hidden_dim)))for i in range(len(x[0]))],We:=[We[i]-lr*sum((xr[i//hidden_dim]-x[k][i//hidden_dim])*h[i%hidden_dim]*x[k][i//hidden_dim]for k in range(len(x)))for i in range(len(We))],Wd:=[Wd[i]-lr*sum((xr[i%len(x[0])]-x[k][i%len(x[0])])*h[i//len(x[0])]for k in range(len(x)))for i in range(len(Wd))]]for _ in range(epochs)];return We,Wd
def restricted_boltzmann(visible,hidden=10,lr=0.01,epochs=100):import random;W=[[random.gauss(0,0.01)for _ in range(hidden)]for _ in range(len(visible[0]))];[[h:=[sigmoid(sum(W[i][j]*visible[k][i]for i in range(len(visible[0]))))for j in range(hidden)],v:=[sigmoid(sum(W[i][j]*h[j]for j in range(hidden)))for i in range(len(visible[0]))],W:=[[W[i][j]+lr*(visible[k][i]*h[j]-v[i]*h[j])for j in range(hidden)]for i in range(len(visible[0]))]]for k in range(len(visible))for _ in range(epochs)];return W
def deep_belief_network(x,layers=[10,5],lr=0.01,epochs=100):rbms=[];[[rbms.append(restricted_boltzmann(x if i==0 else h,layers[i],lr,epochs)),h:=[sigmoid(sum(rbms[-1][j][k]*x[l][j]for j in range(len(rbms[-1]))))for k in range(layers[i])for l in range(len(x))]]for i in range(len(layers))];return rbms
def convolutional_layer(input,kernel,stride=1):h,w=len(input),len(input[0]);kh,kw=len(kernel),len(kernel[0]);oh,ow=(h-kh)//stride+1,(w-kw)//stride+1;return[[sum(input[i+ki][j+kj]*kernel[ki][kj]for ki in range(kh)for kj in range(kw))for j in range(0,w-kw+1,stride)]for i in range(0,h-kh+1,stride)]
def max_pooling(input,size=2,stride=2):h,w=len(input),len(input[0]);return[[max(input[i+di][j+dj]for di in range(size)for dj in range(size))for j in range(0,w,stride)]for i in range(0,h,stride)]
def avg_pooling(input,size=2,stride=2):h,w=len(input),len(input[0]);return[[sum(input[i+di][j+dj]for di in range(size)for dj in range(size))/(size*size)for j in range(0,w,stride)]for i in range(0,h,stride)]
def batch_normalization(x,eps=1e-5):mean=sum(x)/len(x);var=sum((xi-mean)**2 for xi in x)/len(x);return[(xi-mean)/sqrt(var+eps)for xi in x]
def layer_normalization(x,eps=1e-5):mean=[sum(x[i])/len(x[i])for i in range(len(x))];var=[sum((x[i][j]-mean[i])**2 for j in range(len(x[i])))/len(x[i])for i in range(len(x))];return[[(x[i][j]-mean[i])/sqrt(var[i]+eps)for j in range(len(x[i]))]for i in range(len(x))]
def dropout(x,rate=0.5):import random;return[xi if random.random()>rate else 0 for xi in x]
def attention(q,k,v):scores=[dot(q,ki)/sqrt(len(q))for ki in k];weights=softmax(scores);return sum(w*vi for w,vi in zip(weights,v))
def multi_head_attention(q,k,v,n_heads=8):d=len(q)//n_heads;heads=[attention(q[i*d:(i+1)*d],k,v)for i in range(n_heads)];return sum(heads)/n_heads
def positional_encoding(seq_len,d_model):return[[sin(pos/(10000**(2*i/d_model)))if i%2==0 else cos(pos/(10000**(2*(i-1)/d_model)))for i in range(d_model)]for pos in range(seq_len)]
def transformer_block(x,n_heads=8):attn=multi_head_attention(x,x,x,n_heads);x_norm=layer_normalization([xi+ai for xi,ai in zip(x,attn)]);ffn=[relu(sum(wi*xi for wi,xi in zip([1]*len(x_norm[0]),xn)))for xn in x_norm];return layer_normalization([xn+fn for xn,fn in zip(x_norm,ffn)])
def lstm_cell(x,h,c,Wf,Wi,Wo,Wc):ft=sigmoid(sum(Wf[i]*x[i]for i in range(len(x)))+sum(Wf[i+len(x)]*h[i]for i in range(len(h))));it=sigmoid(sum(Wi[i]*x[i]for i in range(len(x)))+sum(Wi[i+len(x)]*h[i]for i in range(len(h))));ot=sigmoid(sum(Wo[i]*x[i]for i in range(len(x)))+sum(Wo[i+len(x)]*h[i]for i in range(len(h))));ct=tanh(sum(Wc[i]*x[i]for i in range(len(x)))+sum(Wc[i+len(x)]*h[i]for i in range(len(h))));c_new=[ft*c[i]+it*ct for i in range(len(c))];h_new=[ot*tanh(c_new[i])for i in range(len(c_new))];return h_new,c_new
def gru_cell(x,h,Wz,Wr,Wh):zt=sigmoid(sum(Wz[i]*x[i]for i in range(len(x)))+sum(Wz[i+len(x)]*h[i]for i in range(len(h))));rt=sigmoid(sum(Wr[i]*x[i]for i in range(len(x)))+sum(Wr[i+len(x)]*h[i]for i in range(len(h))));ht=tanh(sum(Wh[i]*x[i]for i in range(len(x)))+sum(Wh[i+len(x)]*rt*h[i]for i in range(len(h))));return[(1-zt)*h[i]+zt*ht for i in range(len(h))]
def rnn_cell(x,h,Wx,Wh):return tanh([sum(Wx[i]*x[i]for i in range(len(x)))+sum(Wh[i]*h[i]for i in range(len(h)))])
def bidirectional_rnn(x,hidden_dim=10):import random;Wxf=[random.gauss(0,0.1)for _ in range(len(x[0])*hidden_dim)];Whf=[random.gauss(0,0.1)for _ in range(hidden_dim*hidden_dim)];Wxb=[random.gauss(0,0.1)for _ in range(len(x[0])*hidden_dim)];Whb=[random.gauss(0,0.1)for _ in range(hidden_dim*hidden_dim)];hf=[0]*hidden_dim;hb=[0]*hidden_dim;forward=[rnn_cell(xi,hf,Wxf,Whf)for xi in x];backward=[rnn_cell(xi,hb,Wxb,Whb)for xi in reversed(x)];return[f+b for f,b in zip(forward,reversed(backward))]
def seq2seq(encoder_input,decoder_input,hidden_dim=10):import random;We=[random.gauss(0,0.1)for _ in range(len(encoder_input[0])*hidden_dim)];Wd=[random.gauss(0,0.1)for _ in range(len(decoder_input[0])*hidden_dim)];h=[0]*hidden_dim;[[h:=rnn_cell(xi,h,We,[0]*hidden_dim*hidden_dim)]for xi in encoder_input];output=[];[[output.append(rnn_cell(xi,h,Wd,[0]*hidden_dim*hidden_dim)),h:=output[-1]]for xi in decoder_input];return output
def beam_search(scores,beam_width=3):import heapq;beams=[(0,[])];[[beams:=heapq.nsmallest(beam_width,[(score+s,path+[i])for score,path in beams for i,s in enumerate(row)])for row in scores]];return min(beams,key=lambda b:b[0])
def viterbi(obs,states,start_p,trans_p,emit_p):V=[{s:start_p[s]*emit_p[s][obs[0]]for s in states}];path={s:[s]for s in states};[[V.append({s:max(V[t-1][s0]*trans_p[s0][s]*emit_p[s][obs[t]]for s0 in states)for s in states}),path.__setitem__(s,path[max(states,key=lambda s0:V[t-1][s0]*trans_p[s0][s])]+[s])]for t in range(1,len(obs))for s in states];return path[max(states,key=lambda s:V[-1][s])]
def forward_algorithm(obs,states,start_p,trans_p,emit_p):fwd=[{s:start_p[s]*emit_p[s][obs[0]]for s in states}];[[fwd.append({s:sum(fwd[t-1][s0]*trans_p[s0][s]*emit_p[s][obs[t]]for s0 in states)for s in states})]for t in range(1,len(obs))];return sum(fwd[-1].values())
def backward_algorithm(obs,states,trans_p,emit_p):bwd=[{s:1 for s in states}for _ in range(len(obs))];[[bwd[t-1].__setitem__(s,sum(trans_p[s][s1]*emit_p[s1][obs[t]]*bwd[t][s1]for s1 in states))]for t in range(len(obs)-1,0,-1)for s in states];return bwd
def baum_welch(obs,states,start_p,trans_p,emit_p,n_iter=100):[[xi:=[{(s0,s1):fwd[t][s0]*trans_p[s0][s1]*emit_p[s1][obs[t+1]]*bwd[t+1][s1]/sum(fwd[t][s]*trans_p[s][s2]*emit_p[s2][obs[t+1]]*bwd[t+1][s2]for s in states for s2 in states)for s0 in states for s1 in states}for t in range(len(obs)-1)],gamma:=[{s:sum(xi[t][(s,s1)]for s1 in states)if t<len(xi)else fwd[t][s]*bwd[t][s]/sum(fwd[t][s2]*bwd[t][s2]for s2 in states)for s in states}for t in range(len(obs))],start_p:={s:gamma[0][s]for s in states},trans_p:={s0:{s1:sum(xi[t][(s0,s1)]for t in range(len(xi)))/sum(gamma[t][s0]for t in range(len(gamma)-1))for s1 in states}for s0 in states},emit_p:={s:{o:sum(gamma[t][s]for t in range(len(obs))if obs[t]==o)/sum(gamma[t][s]for t in range(len(obs)))for o in set(obs)}for s in states}]for _ in range(n_iter)for fwd in[forward_algorithm(obs,states,start_p,trans_p,emit_p)]for bwd in[backward_algorithm(obs,states,trans_p,emit_p)]];return start_p,trans_p,emit_p

def q_learning(states,actions,rewards,alpha=0.1,gamma=0.9,episodes=1000):import random;Q={(s,a):0 for s in states for a in actions};[[s:=random.choice(states),[[a:=random.choice(actions),r:=rewards.get((s,a),0),s_next:=random.choice(states),Q.__setitem__((s,a),Q[(s,a)]+alpha*(r+gamma*max(Q[(s_next,a2)]for a2 in actions)-Q[(s,a)])),s:=s_next]for _ in range(100)]]for _ in range(episodes)];return Q
def sarsa(states,actions,rewards,alpha=0.1,gamma=0.9,episodes=1000):import random;Q={(s,a):0 for s in states for a in actions};[[s:=random.choice(states),a:=random.choice(actions),[[r:=rewards.get((s,a),0),s_next:=random.choice(states),a_next:=random.choice(actions),Q.__setitem__((s,a),Q[(s,a)]+alpha*(r+gamma*Q[(s_next,a_next)]-Q[(s,a)])),s,a:=s_next,a_next]for _ in range(100)]]for _ in range(episodes)];return Q
def dqn(states,actions,alpha=0.001,gamma=0.9,episodes=1000):import random;Q={s:[random.random()for _ in actions]for s in states};[[s:=random.choice(states),a:=random.randint(0,len(actions)-1),r:=random.random(),s_next:=random.choice(states),Q[s].__setitem__(a,Q[s][a]+alpha*(r+gamma*max(Q[s_next])-Q[s][a]))]for _ in range(episodes)];return Q
def policy_gradient(states,actions,alpha=0.001,gamma=0.9,episodes=1000):import random;policy={s:[1/len(actions)]*len(actions)for s in states};[[s:=random.choice(states),a:=random.choices(range(len(actions)),weights=policy[s])[0],r:=random.random(),grad:=[0]*len(actions),grad.__setitem__(a,1),policy[s]:=[policy[s][i]+alpha*grad[i]*r for i in range(len(actions))],policy[s]:=[p/sum(policy[s])for p in policy[s]]]for _ in range(episodes)];return policy
def actor_critic(states,actions,alpha_actor=0.001,alpha_critic=0.001,gamma=0.9,episodes=1000):import random;actor={s:[1/len(actions)]*len(actions)for s in states};critic={s:0 for s in states};[[s:=random.choice(states),a:=random.choices(range(len(actions)),weights=actor[s])[0],r:=random.random(),s_next:=random.choice(states),td_error:=r+gamma*critic[s_next]-critic[s],critic.__setitem__(s,critic[s]+alpha_critic*td_error),grad:=[0]*len(actions),grad.__setitem__(a,1),actor[s]:=[actor[s][i]+alpha_actor*grad[i]*td_error for i in range(len(actions))],actor[s]:=[p/sum(actor[s])for p in actor[s]]]for _ in range(episodes)];return actor,critic
def a3c(states,actions,alpha=0.001,gamma=0.9,episodes=1000):import random;global_policy={s:[1/len(actions)]*len(actions)for s in states};global_value={s:0 for s in states};[[local_policy:={s:[1/len(actions)]*len(actions)for s in states},local_value:={s:0 for s in states},[[s:=random.choice(states),a:=random.choices(range(len(actions)),weights=local_policy[s])[0],r:=random.random(),s_next:=random.choice(states),td_error:=r+gamma*local_value[s_next]-local_value[s],local_value.__setitem__(s,local_value[s]+alpha*td_error),grad:=[0]*len(actions),grad.__setitem__(a,1),local_policy[s]:=[local_policy[s][i]+alpha*grad[i]*td_error for i in range(len(actions))],local_policy[s]:=[p/sum(local_policy[s])for p in local_policy[s]]]for _ in range(10)],global_policy:={s:[(global_policy[s][i]+local_policy[s][i])/2 for i in range(len(actions))]for s in states},global_value:={s:(global_value[s]+local_value[s])/2 for s in states}]for _ in range(episodes)];return global_policy,global_value
def ddpg(states,actions,alpha_actor=0.001,alpha_critic=0.001,gamma=0.9,tau=0.001,episodes=1000):import random;actor={s:[random.random()for _ in actions]for s in states};critic={(s,a):random.random()for s in states for a in actions};target_actor={s:actor[s][:]for s in states};target_critic={(s,a):critic[(s,a)]for s in states for a in actions};[[s:=random.choice(states),a:=max(range(len(actions)),key=lambda i:actor[s][i]),r:=random.random(),s_next:=random.choice(states),a_next:=max(range(len(actions)),key=lambda i:target_actor[s_next][i]),td_error:=r+gamma*target_critic[(s_next,actions[a_next])]-critic[(s,actions[a])],critic.__setitem__((s,actions[a]),critic[(s,actions[a])]+alpha_critic*td_error),actor[s]:=[actor[s][i]+alpha_actor*(1 if i==a else 0)*td_error for i in range(len(actions))],target_actor:={s:[(1-tau)*target_actor[s][i]+tau*actor[s][i]for i in range(len(actions))]for s in states},target_critic:={(s,a):(1-tau)*target_critic[(s,a)]+tau*critic[(s,a)]for s in states for a in actions}]for _ in range(episodes)];return actor,critic
def td3(states,actions,alpha=0.001,gamma=0.9,tau=0.001,policy_delay=2,episodes=1000):import random;actor={s:[random.random()for _ in actions]for s in states};critic1={(s,a):random.random()for s in states for a in actions};critic2={(s,a):random.random()for s in states for a in actions};target_actor={s:actor[s][:]for s in states};target_critic1={(s,a):critic1[(s,a)]for s in states for a in actions};target_critic2={(s,a):critic2[(s,a)]for s in states for a in actions};[[s:=random.choice(states),a:=max(range(len(actions)),key=lambda i:actor[s][i]),r:=random.random(),s_next:=random.choice(states),a_next:=max(range(len(actions)),key=lambda i:target_actor[s_next][i]),q_target:=r+gamma*min(target_critic1[(s_next,actions[a_next])],target_critic2[(s_next,actions[a_next])]),critic1.__setitem__((s,actions[a]),critic1[(s,actions[a])]+alpha*(q_target-critic1[(s,actions[a])])),critic2.__setitem__((s,actions[a]),critic2[(s,actions[a])]+alpha*(q_target-critic2[(s,actions[a])])),[[actor[s]:=[actor[s][i]+alpha*(1 if i==a else 0)*critic1[(s,actions[a])]for i in range(len(actions))],target_actor:={s:[(1-tau)*target_actor[s][i]+tau*actor[s][i]for i in range(len(actions))]for s in states},target_critic1:={(s,a):(1-tau)*target_critic1[(s,a)]+tau*critic1[(s,a)]for s in states for a in actions},target_critic2:={(s,a):(1-tau)*target_critic2[(s,a)]+tau*critic2[(s,a)]for s in states for a in actions}]if _%policy_delay==0 else None]for _ in range(episodes)];return actor,critic1,critic2
def sac(states,actions,alpha=0.001,gamma=0.9,tau=0.001,entropy_coef=0.2,episodes=1000):import random;actor={s:[random.random()for _ in actions]for s in states};critic1={(s,a):random.random()for s in states for a in actions};critic2={(s,a):random.random()for s in states for a in actions};target_critic1={(s,a):critic1[(s,a)]for s in states for a in actions};target_critic2={(s,a):critic2[(s,a)]for s in states for a in actions};[[s:=random.choice(states),a:=random.choices(range(len(actions)),weights=softmax(actor[s]))[0],r:=random.random(),s_next:=random.choice(states),a_next:=random.choices(range(len(actions)),weights=softmax(actor[s_next]))[0],q_target:=r+gamma*(min(target_critic1[(s_next,actions[a_next])],target_critic2[(s_next,actions[a_next])])-entropy_coef*log(softmax(actor[s_next])[a_next])),critic1.__setitem__((s,actions[a]),critic1[(s,actions[a])]+alpha*(q_target-critic1[(s,actions[a])])),critic2.__setitem__((s,actions[a]),critic2[(s,actions[a])]+alpha*(q_target-critic2[(s,actions[a])])),actor[s]:=[actor[s][i]+alpha*(min(critic1[(s,actions[i])],critic2[(s,actions[i])])+entropy_coef*log(softmax(actor[s])[i])if i==a else 0)for i in range(len(actions))],target_critic1:={(s,a):(1-tau)*target_critic1[(s,a)]+tau*critic1[(s,a)]for s in states for a in actions},target_critic2:={(s,a):(1-tau)*target_critic2[(s,a)]+tau*critic2[(s,a)]for s in states for a in actions}]for _ in range(episodes)];return actor,critic1,critic2
def ppo(states,actions,alpha=0.001,gamma=0.9,clip_ratio=0.2,epochs=10,episodes=1000):import random;policy={s:[1/len(actions)]*len(actions)for s in states};value={s:0 for s in states};[[trajectories:=[],[[s:=random.choice(states),a:=random.choices(range(len(actions)),weights=policy[s])[0],r:=random.random(),s_next:=random.choice(states),trajectories.append((s,a,r,s_next))]for _ in range(100)],[[old_prob:=policy[traj[0]][traj[1]],advantage:=traj[2]+gamma*value[traj[3]]-value[traj[0]],ratio:=policy[traj[0]][traj[1]]/old_prob if old_prob>0 else 1,clipped_ratio:=min(max(ratio,1-clip_ratio),1+clip_ratio),policy_loss:=-min(ratio*advantage,clipped_ratio*advantage),value_loss:=(traj[2]+gamma*value[traj[3]]-value[traj[0]])**2,policy[traj[0]]:=[policy[traj[0]][i]+alpha*(-policy_loss if i==traj[1]else 0)for i in range(len(actions))],policy[traj[0]]:=[p/sum(policy[traj[0]])for p in policy[traj[0]]],value.__setitem__(traj[0],value[traj[0]]+alpha*(-value_loss))]for traj in trajectories]for _ in range(epochs)]for _ in range(episodes)];return policy,value
def trpo(states,actions,alpha=0.001,gamma=0.9,delta=0.01,episodes=1000):import random;policy={s:[1/len(actions)]*len(actions)for s in states};value={s:0 for s in states};[[trajectories:=[],[[s:=random.choice(states),a:=random.choices(range(len(actions)),weights=policy[s])[0],r:=random.random(),s_next:=random.choice(states),trajectories.append((s,a,r,s_next))]for _ in range(100)],advantages:=[traj[2]+gamma*value[traj[3]]-value[traj[0]]for traj in trajectories],kl_div:=sum(sum(policy[traj[0]][i]*log(policy[traj[0]][i]/([1/len(actions)]*len(actions))[i])if policy[traj[0]][i]>0 else 0 for i in range(len(actions)))for traj in trajectories),[[policy[traj[0]]:=[policy[traj[0]][i]+alpha*advantages[j]*(1 if i==traj[1]else 0)for i in range(len(actions))],policy[traj[0]]:=[p/sum(policy[traj[0]])for p in policy[traj[0]]]]if kl_div<delta else None for j,traj in enumerate(trajectories)]]for _ in range(episodes)];return policy,value
def rainbow_dqn(states,actions,alpha=0.001,gamma=0.9,episodes=1000):import random;Q={s:[random.random()for _ in actions]for s in states};target_Q={s:Q[s][:]for s in states};priorities={s:1 for s in states};[[s:=random.choice(states,weights=[priorities[s]for s in states]),a:=max(range(len(actions)),key=lambda i:Q[s][i]),r:=random.random(),s_next:=random.choice(states),td_error:=r+gamma*max(target_Q[s_next])-Q[s][a],Q[s].__setitem__(a,Q[s][a]+alpha*td_error),priorities.__setitem__(s,abs(td_error)),target_Q:={s:[(1-0.001)*target_Q[s][i]+0.001*Q[s][i]for i in range(len(actions))]for s in states}if _%10==0 else target_Q]for _ in range(episodes)];return Q
def alphazero(states,actions,mcts_sims=100,episodes=1000):import random;policy={s:[1/len(actions)]*len(actions)for s in states};value={s:0 for s in states};[[s:=random.choice(states),tree:={s:{'N':0,'W':0,'Q':0,'P':policy[s],'children':{}}},[[node:=tree[s],a:=random.choices(range(len(actions)),weights=node['P'])[0]if random.random()>0.5 else max(range(len(actions)),key=lambda i:node['Q'][i]if'Q'in node and isinstance(node['Q'],list)else 0),s_next:=random.choice(states),r:=random.random(),tree.__setitem__(s_next,{'N':0,'W':0,'Q':0,'P':policy.get(s_next,[1/len(actions)]*len(actions)),'children':{}}),node['children'].__setitem__(a,tree[s_next]),v:=r+0.9*value.get(s_next,0),[[node.__setitem__('N',node['N']+1),node.__setitem__('W',node['W']+v),node.__setitem__('Q',node['W']/node['N']if node['N']>0 else 0)]for node in[tree[s]]]]for _ in range(mcts_sims)],policy:={s:[tree[s]['children'].get(a,{'N':0})['N']/tree[s]['N']if tree[s]['N']>0 else 1/len(actions)for a in range(len(actions))]if s in tree else[1/len(actions)]*len(actions)for s in states},value:={s:tree[s]['Q']if s in tree else 0 for s in states}]for _ in range(episodes)];return policy,value
def muzero(states,actions,episodes=1000):import random;model={'dynamics':{},'prediction':{},'representation':{}};[[s:=random.choice(states),h:=model['representation'].get(s,random.random()),[[a:=random.choice(actions),r,h_next:=random.random(),random.random(),model['dynamics'].__setitem__((h,a),(r,h_next)),p,v:=softmax([random.random()for _ in actions]),random.random(),model['prediction'].__setitem__(h_next,(p,v)),h:=h_next]for _ in range(10)]]for _ in range(episodes)];return model
def world_models(states,actions,latent_dim=32,episodes=1000):import random;vae={'encoder':{},'decoder':{}};mdnrnn={'hidden':{},'output':{}};controller={};[[s:=random.choice(states),z:=[random.gauss(0,1)for _ in range(latent_dim)],vae['encoder'].__setitem__(s,z),[[a:=random.choice(actions),s_next:=random.choice(states),r:=random.random(),z_next:=[random.gauss(0,1)for _ in range(latent_dim)],mdnrnn['hidden'].__setitem__((z,a),z_next),mdnrnn['output'].__setitem__((z,a),(r,s_next)),controller.__setitem__(z,a),z:=z_next]for _ in range(10)]]for _ in range(episodes)];return vae,mdnrnn,controller
def dreamer(states,actions,episodes=1000):import random;world_model={'encoder':{},'decoder':{},'dynamics':{},'reward':{}};policy={};value={};[[s:=random.choice(states),h:=[random.random()for _ in range(10)],world_model['encoder'].__setitem__(s,h),[[a:=random.choice(actions),s_next:=random.choice(states),r:=random.random(),h_next:=[random.random()for _ in range(10)],world_model['dynamics'].__setitem__((h,a),h_next),world_model['reward'].__setitem__((h,a),r),world_model['decoder'].__setitem__(h,s),policy.__setitem__(h,a),value.__setitem__(h,r),h:=h_next]for _ in range(10)]]for _ in range(episodes)];return world_model,policy,value
def curiosity_driven(states,actions,alpha=0.001,episodes=1000):import random;forward_model={};inverse_model={};policy={s:[1/len(actions)]*len(actions)for s in states};[[s:=random.choice(states),a:=random.choices(range(len(actions)),weights=policy[s])[0],s_next:=random.choice(states),forward_pred:=forward_model.get((s,a),s_next),inverse_pred:=inverse_model.get((s,s_next),a),forward_error:=1 if forward_pred!=s_next else 0,inverse_error:=1 if inverse_pred!=a else 0,intrinsic_reward:=forward_error,forward_model.__setitem__((s,a),s_next),inverse_model.__setitem__((s,s_next),a),policy[s]:=[policy[s][i]+alpha*intrinsic_reward*(1 if i==a else 0)for i in range(len(actions))],policy[s]:=[p/sum(policy[s])for p in policy[s]]]for _ in range(episodes)];return forward_model,inverse_model,policy
def hindsight_experience(states,actions,goals,alpha=0.001,episodes=1000):import random;Q={(s,g,a):0 for s in states for g in goals for a in actions};[[s:=random.choice(states),g:=random.choice(goals),trajectory:=[],[[a:=random.choice(actions),s_next:=random.choice(states),r:=1 if s_next==g else 0,trajectory.append((s,a,r,s_next)),Q.__setitem__((s,g,a),Q[(s,g,a)]+alpha*(r+0.9*max(Q[(s_next,g,a2)]for a2 in actions)-Q[(s,g,a)])),s:=s_next]for _ in range(10)],achieved_goal:=trajectory[-1][3],[[Q.__setitem__((traj[0],achieved_goal,traj[1]),Q[(traj[0],achieved_goal,traj[1])]+alpha*(1+0.9*max(Q[(traj[3],achieved_goal,a)]for a in actions)-Q[(traj[0],achieved_goal,traj[1])]))]for traj in trajectory]]for _ in range(episodes)];return Q
def meta_learning(tasks,alpha=0.001,beta=0.01,episodes=1000):import random;theta={};[[task:=random.choice(tasks),theta_task:=theta.copy(),[[x,y:=random.random(),random.random(),loss:=(theta_task.get('w',0)*x-y)**2,theta_task.__setitem__('w',theta_task.get('w',0)-alpha*2*(theta_task.get('w',0)*x-y)*x)]for _ in range(10)],theta.__setitem__('w',theta.get('w',0)-beta*sum((theta_task.get('w',0)-theta.get('w',0))**2 for _ in range(1)))]for _ in range(episodes)];return theta
def few_shot_learning(support_set,query_set,n_way=5,k_shot=1):import random;prototypes={};[[class_samples:=[x for x,y in support_set if y==c],prototypes.__setitem__(c,sum(s for s in class_samples)/len(class_samples)if class_samples else 0)]for c in set(y for _,y in support_set)];predictions=[];[[x,_:=q,distances:={c:abs(x-prototypes[c])for c in prototypes},predictions.append(min(distances,key=distances.get))]for q in query_set];return predictions
def contrastive_learning(x,temperature=0.07):import random;similarities=[[dot(xi,xj)/(norm(xi)*norm(xj))for xj in x]for xi in x];loss=0;[[loss:=loss-log(exp(similarities[i][j]/temperature)/sum(exp(similarities[i][k]/temperature)for k in range(len(x))if k!=i))]for i in range(len(x))for j in range(len(x))if i!=j];return loss
def self_supervised_learning(x,mask_ratio=0.15):import random;masked=[];labels=[];[[idx:=random.sample(range(len(xi)),int(len(xi)*mask_ratio)),masked.append([xi[i]if i not in idx else 0 for i in range(len(xi))]),labels.append([xi[i]if i in idx else 0 for i in range(len(xi))])]for xi in x];return masked,labels
def gan_generator(z,hidden_dim=100):import random;W1=[random.gauss(0,0.1)for _ in range(len(z)*hidden_dim)];W2=[random.gauss(0,0.1)for _ in range(hidden_dim*len(z))];h=[relu(sum(W1[i*hidden_dim+j]*z[i]for i in range(len(z))))for j in range(hidden_dim)];return[tanh(sum(W2[j*len(z)+i]*h[j]for j in range(hidden_dim)))for i in range(len(z))]
def gan_discriminator(x,hidden_dim=100):import random;W1=[random.gauss(0,0.1)for _ in range(len(x)*hidden_dim)];W2=[random.gauss(0,0.1)for _ in range(hidden_dim)];h=[relu(sum(W1[i*hidden_dim+j]*x[i]for i in range(len(x))))for j in range(hidden_dim)];return sigmoid(sum(W2[j]*h[j]for j in range(hidden_dim)))
def wgan(x_real,z_dim=10,epochs=1000):import random;G_W=[[random.gauss(0,0.1)for _ in range(100)]for _ in range(z_dim)];D_W=[[random.gauss(0,0.1)for _ in range(100)]for _ in range(len(x_real[0]))];[[z:=[random.gauss(0,1)for _ in range(z_dim)],x_fake:=gan_generator(z),d_real:=gan_discriminator(x_real[_%len(x_real)]),d_fake:=gan_discriminator(x_fake),d_loss:=-(d_real-d_fake),g_loss:=-d_fake,D_W:=[[D_W[i][j]-0.001*d_loss for j in range(len(D_W[0]))]for i in range(len(D_W))],G_W:=[[G_W[i][j]-0.001*g_loss for j in range(len(G_W[0]))]for i in range(len(G_W))]]for _ in range(epochs)];return G_W,D_W
def vae_encoder(x,latent_dim=10):import random;W_mean=[random.gauss(0,0.1)for _ in range(len(x)*latent_dim)];W_logvar=[random.gauss(0,0.1)for _ in range(len(x)*latent_dim)];mean=[sum(W_mean[i*latent_dim+j]*x[i]for i in range(len(x)))for j in range(latent_dim)];logvar=[sum(W_logvar[i*latent_dim+j]*x[i]for i in range(len(x)))for j in range(latent_dim)];return mean,logvar
def vae_decoder(z):import random;W=[random.gauss(0,0.1)for _ in range(len(z)*len(z))];return[sigmoid(sum(W[i*len(z)+j]*z[i]for i in range(len(z))))for j in range(len(z))]
def vae_reparameterize(mean,logvar):import random;std=[exp(0.5*lv)for lv in logvar];eps=[random.gauss(0,1)for _ in range(len(mean))];return[m+e*s for m,e,s in zip(mean,eps,std)]
def diffusion_forward(x,t,beta_start=0.0001,beta_end=0.02,T=1000):beta=beta_start+(beta_end-beta_start)*t/T;alpha=1-beta;import random;noise=[random.gauss(0,1)for _ in range(len(x))];return[sqrt(alpha)*xi+sqrt(1-alpha)*ni for xi,ni in zip(x,noise)]
def diffusion_reverse(x_t,t,model,beta_start=0.0001,beta_end=0.02,T=1000):beta=beta_start+(beta_end-beta_start)*t/T;alpha=1-beta;noise_pred=model.get(x_t,[0]*len(x_t));return[(x_t[i]-sqrt(1-alpha)*noise_pred[i])/sqrt(alpha)for i in range(len(x_t))]
def stable_diffusion(prompt,steps=50):import random;latent=[random.gauss(0,1)for _ in range(64)];[[latent:=diffusion_reverse(latent,t,{})]for t in range(steps,0,-1)];return latent
def neural_ode(x,t,f):h=0.01;xt=x;[[xt:=[xt[i]+h*f(xt,ti)[i]for i in range(len(xt))]]for ti in[t+i*h for i in range(int((t+1-t)/h))]];return xt
def graph_conv(adj,x,W):return[[sum(adj[i][j]*sum(W[k][l]*x[j][k]for k in range(len(x[0])))for j in range(len(adj[0])))for l in range(len(W[0]))]for i in range(len(adj))]
def graph_attention(adj,x,W_q,W_k,W_v):Q=[[sum(W_q[j][k]*x[i][j]for j in range(len(x[0])))for k in range(len(W_q[0]))]for i in range(len(x))];K=[[sum(W_k[j][k]*x[i][j]for j in range(len(x[0])))for k in range(len(W_k[0]))]for i in range(len(x))];V=[[sum(W_v[j][k]*x[i][j]for j in range(len(x[0])))for k in range(len(W_v[0]))]for i in range(len(x))];scores=[[dot(Q[i],K[j])/sqrt(len(Q[0]))if adj[i][j]else float('-inf')for j in range(len(K))]for i in range(len(Q))];attn=[[exp(scores[i][j])/sum(exp(scores[i][k])for k in range(len(scores[0]))if scores[i][k]!=float('-inf'))if scores[i][j]!=float('-inf')else 0 for j in range(len(scores[0]))]for i in range(len(scores))];return[[sum(attn[i][j]*V[j][k]for j in range(len(V)))for k in range(len(V[0]))]for i in range(len(attn))]
def message_passing(edges,node_features,W):messages={i:[0]*len(W[0])for i in range(len(node_features))};[[messages.__setitem__(j,[messages[j][k]+sum(W[l][k]*node_features[i][l]for l in range(len(node_features[0])))for k in range(len(W[0]))])for i,j in edges]];return[[node_features[i][j]+messages[i][j]for j in range(len(node_features[0]))]for i in range(len(node_features))]
def graph_isomorphism(adj1,adj2):if len(adj1)!=len(adj2):return False;import random;colors1=[hash(tuple(adj1[i]))for i in range(len(adj1))];colors2=[hash(tuple(adj2[i]))for i in range(len(adj2))];[[new_colors1:=[hash(tuple(sorted([colors1[j]for j in range(len(adj1))if adj1[i][j]])))for i in range(len(adj1))],new_colors2:=[hash(tuple(sorted([colors2[j]for j in range(len(adj2))if adj2[i][j]])))for i in range(len(adj2))],colors1,colors2:=new_colors1,new_colors2]for _ in range(10)];return sorted(colors1)==sorted(colors2)
def pagerank(adj,damping=0.85,max_iter=100):n=len(adj);ranks=[1/n]*n;[[new_ranks:=[(1-damping)/n+damping*sum(ranks[j]*adj[j][i]/sum(adj[j])if sum(adj[j])>0 else ranks[j]/n for j in range(n))for i in range(n)],ranks:=new_ranks]for _ in range(max_iter)];return ranks
def hits(adj,max_iter=100):n=len(adj);auth=[1]*n;hub=[1]*n;[[auth:=[sum(hub[j]*adj[j][i]for j in range(n))for i in range(n)],auth:=[a/sqrt(sum(x**2 for x in auth))for a in auth],hub:=[sum(auth[j]*adj[i][j]for j in range(n))for i in range(n)],hub:=[h/sqrt(sum(x**2 for x in hub))for h in hub]]for _ in range(max_iter)];return auth,hub
def community_detection(adj):n=len(adj);communities=[i for i in range(n)];modularity=0;[[best_merge:=None,best_delta:=float('-inf'),[[delta:=sum(adj[i][j]for i in range(n)for j in range(n)if communities[i]==c1 and communities[j]==c2)/(sum(sum(adj[i])for i in range(n)if communities[i]==c1)*sum(sum(adj[j])for j in range(n)if communities[j]==c2)),best_merge,best_delta:=(c1,c2),delta if delta>best_delta else(best_merge,best_delta)]for c1 in set(communities)for c2 in set(communities)if c1<c2],communities:=[best_merge[0]if c==best_merge[1]else c for c in communities]if best_merge else communities]for _ in range(n)];return communities
def node2vec(adj,dim=128,walk_length=80,num_walks=10,p=1,q=1):import random;walks=[];[[start:=i,walk:=[start],[[neighbors:=[j for j in range(len(adj))if adj[walk[-1]][j]],walk.append(random.choice(neighbors)if neighbors else walk[-1])]for _ in range(walk_length-1)],walks.append(walk)]for i in range(len(adj))for _ in range(num_walks)];embeddings={};return embeddings
def knowledge_graph_embedding(triples,dim=50,epochs=1000):import random;entities=set([t[0]for t in triples]+[t[2]for t in triples]);relations=set([t[1]for t in triples]);entity_emb={e:[random.gauss(0,0.1)for _ in range(dim)]for e in entities};relation_emb={r:[random.gauss(0,0.1)for _ in range(dim)]for r in relations};[[h,r,t:=random.choice(triples),score:=sum((entity_emb[h][i]+relation_emb[r][i]-entity_emb[t][i])**2 for i in range(dim)),h_neg:=random.choice(list(entities)),score_neg:=sum((entity_emb[h_neg][i]+relation_emb[r][i]-entity_emb[t][i])**2 for i in range(dim)),loss:=max(0,1+score-score_neg),entity_emb[h]:=[entity_emb[h][i]-0.01*2*(entity_emb[h][i]+relation_emb[r][i]-entity_emb[t][i])for i in range(dim)],relation_emb[r]:=[relation_emb[r][i]-0.01*2*(entity_emb[h][i]+relation_emb[r][i]-entity_emb[t][i])for i in range(dim)],entity_emb[t]:=[entity_emb[t][i]+0.01*2*(entity_emb[h][i]+relation_emb[r][i]-entity_emb[t][i])for i in range(dim)]]for _ in range(epochs)];return entity_emb,relation_emb
def federated_learning(clients_data,global_model,rounds=100):import random;[[local_models:=[],[[local_model:=global_model.copy(),[[loss:=sum((local_model.get('w',0)*x-y)**2 for x,y in client_data),local_model.__setitem__('w',local_model.get('w',0)-0.01*sum(2*(local_model.get('w',0)*x-y)*x for x,y in client_data))]for _ in range(10)],local_models.append(local_model)]for client_data in clients_data],global_model:={'w':sum(m.get('w',0)for m in local_models)/len(local_models)if local_models else 0}]for _ in range(rounds)];return global_model
def differential_privacy(x,epsilon=1.0,delta=1e-5):import random;sensitivity=1;noise_scale=sensitivity*sqrt(2*log(1.25/delta))/epsilon;return[xi+random.gauss(0,noise_scale)for xi in x]
def homomorphic_encryption(x,public_key=17):return[(xi*public_key)%100 for xi in x]
def homomorphic_decryption(c,private_key=3):return[(ci*private_key)%100 for ci in c]
def secure_multiparty_computation(x1,x2):import random;r1=[random.randint(0,100)for _ in range(len(x1))];r2=[(x1[i]+x2[i]-r1[i])%100 for i in range(len(x1))];result=[(r1[i]+r2[i])%100 for i in range(len(r1))];return result
def zero_knowledge_proof(secret,challenge):import random;commitment=hash((secret,random.random()));response=(secret+challenge)%100;return commitment,response
def blockchain_hash(data,prev_hash,nonce):return hash((data,prev_hash,nonce))
def proof_of_work(data,prev_hash,difficulty=4):nonce=0;[nonce:=nonce+1 for _ in range(1000000)if not str(blockchain_hash(data,prev_hash,nonce)).startswith('0'*difficulty)];return nonce
def merkle_root(transactions):if not transactions:return None;if len(transactions)==1:return hash(transactions[0]);hashes=[hash(t)for t in transactions];[hashes:=[hash((hashes[i],hashes[i+1]))for i in range(0,len(hashes)-1,2)]+([hashes[-1]]if len(hashes)%2 else[])for _ in range(100)if len(hashes)>1];return hashes[0]

def func1(x):return x*1+1
def func2(x):return x*2+2
def func3(x):return x*3+3
def func4(x):return x*4+4
def func5(x):return x*5+5
def func6(x):return x*6+6
def func7(x):return x*7+7
def func8(x):return x*8+8
def func9(x):return x*9+9
def func10(x):return x*10+0
def func11(x):return x*11+1
def func12(x):return x*12+2
def func13(x):return x*13+3
def func14(x):return x*14+4
def func15(x):return x*15+5
def func16(x):return x*16+6
def func17(x):return x*17+7
def func18(x):return x*18+8
def func19(x):return x*19+9
def func20(x):return x*20+0
def func21(x):return x*21+1
def func22(x):return x*22+2
def func23(x):return x*23+3
def func24(x):return x*24+4
def func25(x):return x*25+5
def func26(x):return x*26+6
def func27(x):return x*27+7
def func28(x):return x*28+8
def func29(x):return x*29+9
def func30(x):return x*30+0
def func31(x):return x*31+1
def func32(x):return x*32+2
def func33(x):return x*33+3
def func34(x):return x*34+4
def func35(x):return x*35+5
def func36(x):return x*36+6
def func37(x):return x*37+7
def func38(x):return x*38+8
def func39(x):return x*39+9
def func40(x):return x*40+0
def func41(x):return x*41+1
def func42(x):return x*42+2
def func43(x):return x*43+3
def func44(x):return x*44+4
def func45(x):return x*45+5
def func46(x):return x*46+6
def func47(x):return x*47+7
def func48(x):return x*48+8
def func49(x):return x*49+9
def func50(x):return x*50+0
def func51(x):return x*51+1
def func52(x):return x*52+2
def func53(x):return x*53+3
def func54(x):return x*54+4
def func55(x):return x*55+5
def func56(x):return x*56+6
def func57(x):return x*57+7
def func58(x):return x*58+8
def func59(x):return x*59+9
def func60(x):return x*60+0
def func61(x):return x*61+1
def func62(x):return x*62+2
def func63(x):return x*63+3
def func64(x):return x*64+4
def func65(x):return x*65+5
def func66(x):return x*66+6
def func67(x):return x*67+7
def func68(x):return x*68+8
def func69(x):return x*69+9
def func70(x):return x*70+0
def func71(x):return x*71+1
def func72(x):return x*72+2
def func73(x):return x*73+3
def func74(x):return x*74+4
def func75(x):return x*75+5
def func76(x):return x*76+6
def func77(x):return x*77+7
def func78(x):return x*78+8
def func79(x):return x*79+9
def func80(x):return x*80+0
def func81(x):return x*81+1
def func82(x):return x*82+2
def func83(x):return x*83+3
def func84(x):return x*84+4
def func85(x):return x*85+5
def func86(x):return x*86+6
def func87(x):return x*87+7
def func88(x):return x*88+8
def func89(x):return x*89+9
def func90(x):return x*90+0
def func91(x):return x*91+1
def func92(x):return x*92+2
def func93(x):return x*93+3
def func94(x):return x*94+4
def func95(x):return x*95+5
def func96(x):return x*96+6
def func97(x):return x*97+7
def func98(x):return x*98+8
def func99(x):return x*99+9
def func100(x):return x*100+0
def func101(x):return x*101+1
def func102(x):return x*102+2
def func103(x):return x*103+3
def func104(x):return x*104+4
def func105(x):return x*105+5
def func106(x):return x*106+6
def func107(x):return x*107+7
def func108(x):return x*108+8
def func109(x):return x*109+9
def func110(x):return x*110+0
def func111(x):return x*111+1
def func112(x):return x*112+2
def func113(x):return x*113+3
def func114(x):return x*114+4
def func115(x):return x*115+5
def func116(x):return x*116+6
def func117(x):return x*117+7
def func118(x):return x*118+8
def func119(x):return x*119+9
def func120(x):return x*120+0
def func121(x):return x*121+1
def func122(x):return x*122+2
def func123(x):return x*123+3
def func124(x):return x*124+4
def func125(x):return x*125+5
def func126(x):return x*126+6
def func127(x):return x*127+7
def func128(x):return x*128+8
def func129(x):return x*129+9
def func130(x):return x*130+0
def func131(x):return x*131+1
def func132(x):return x*132+2
def func133(x):return x*133+3
def func134(x):return x*134+4
def func135(x):return x*135+5
def func136(x):return x*136+6
def func137(x):return x*137+7
def func138(x):return x*138+8
def func139(x):return x*139+9
def func140(x):return x*140+0
def func141(x):return x*141+1
def func142(x):return x*142+2
def func143(x):return x*143+3
def func144(x):return x*144+4
def func145(x):return x*145+5
def func146(x):return x*146+6
def func147(x):return x*147+7
def func148(x):return x*148+8
def func149(x):return x*149+9
def func150(x):return x*150+0
def func151(x):return x*151+1
def func152(x):return x*152+2
def func153(x):return x*153+3
def func154(x):return x*154+4
def func155(x):return x*155+5
def func156(x):return x*156+6
def func157(x):return x*157+7
def func158(x):return x*158+8
def func159(x):return x*159+9
def func160(x):return x*160+0
def func161(x):return x*161+1
def func162(x):return x*162+2
def func163(x):return x*163+3
def func164(x):return x*164+4
def func165(x):return x*165+5
def func166(x):return x*166+6
def func167(x):return x*167+7
def func168(x):return x*168+8
def func169(x):return x*169+9
def func170(x):return x*170+0
def func171(x):return x*171+1
def func172(x):return x*172+2
def func173(x):return x*173+3
def func174(x):return x*174+4
def func175(x):return x*175+5
def func176(x):return x*176+6
def func177(x):return x*177+7
def func178(x):return x*178+8
def func179(x):return x*179+9
def func180(x):return x*180+0
def func181(x):return x*181+1
def func182(x):return x*182+2
def func183(x):return x*183+3
def func184(x):return x*184+4
def func185(x):return x*185+5
def func186(x):return x*186+6
def func187(x):return x*187+7
def func188(x):return x*188+8
def func189(x):return x*189+9
def func190(x):return x*190+0
def func191(x):return x*191+1
def func192(x):return x*192+2
def func193(x):return x*193+3
def func194(x):return x*194+4
def func195(x):return x*195+5
def func196(x):return x*196+6
def func197(x):return x*197+7
def func198(x):return x*198+8
def func199(x):return x*199+9
def func200(x):return x*200+0
def func201(x):return x*201+1
def func202(x):return x*202+2
def func203(x):return x*203+3
def func204(x):return x*204+4
def func205(x):return x*205+5
def func206(x):return x*206+6
def func207(x):return x*207+7
def func208(x):return x*208+8
def func209(x):return x*209+9
def func210(x):return x*210+0
def func211(x):return x*211+1
def func212(x):return x*212+2
def func213(x):return x*213+3
def func214(x):return x*214+4
def func215(x):return x*215+5
def func216(x):return x*216+6
def func217(x):return x*217+7
def func218(x):return x*218+8
def func219(x):return x*219+9
def func220(x):return x*220+0
def func221(x):return x*221+1
def func222(x):return x*222+2
def func223(x):return x*223+3
def func224(x):return x*224+4
def func225(x):return x*225+5
def func226(x):return x*226+6
def func227(x):return x*227+7
def func228(x):return x*228+8
def func229(x):return x*229+9
def func230(x):return x*230+0
def func231(x):return x*231+1
def func232(x):return x*232+2
def func233(x):return x*233+3
def func234(x):return x*234+4
def func235(x):return x*235+5
def func236(x):return x*236+6
def func237(x):return x*237+7
def func238(x):return x*238+8
def func239(x):return x*239+9
def func240(x):return x*240+0
def func241(x):return x*241+1
def func242(x):return x*242+2
def func243(x):return x*243+3
def func244(x):return x*244+4
def func245(x):return x*245+5
def func246(x):return x*246+6
def func247(x):return x*247+7
def func248(x):return x*248+8
def func249(x):return x*249+9
def func250(x):return x*250+0
def func251(x):return x*251+1
def func252(x):return x*252+2
def func253(x):return x*253+3
def func254(x):return x*254+4
def func255(x):return x*255+5
def func256(x):return x*256+6
def func257(x):return x*257+7
def func258(x):return x*258+8
def func259(x):return x*259+9
def func260(x):return x*260+0
def func261(x):return x*261+1
def func262(x):return x*262+2
def func263(x):return x*263+3
def func264(x):return x*264+4
def func265(x):return x*265+5
def func266(x):return x*266+6
def func267(x):return x*267+7
def func268(x):return x*268+8
def func269(x):return x*269+9
def func270(x):return x*270+0
def func271(x):return x*271+1
def func272(x):return x*272+2
def func273(x):return x*273+3
def func274(x):return x*274+4
def func275(x):return x*275+5
def func276(x):return x*276+6
def func277(x):return x*277+7
def func278(x):return x*278+8
def func279(x):return x*279+9
def func280(x):return x*280+0
def func281(x):return x*281+1
def func282(x):return x*282+2
def func283(x):return x*283+3
def func284(x):return x*284+4
def func285(x):return x*285+5
def func286(x):return x*286+6
def func287(x):return x*287+7
def func288(x):return x*288+8
def func289(x):return x*289+9
def func290(x):return x*290+0
def func291(x):return x*291+1
def func292(x):return x*292+2
def func293(x):return x*293+3
def func294(x):return x*294+4
def func295(x):return x*295+5
def func296(x):return x*296+6
def func297(x):return x*297+7
def func298(x):return x*298+8
def func299(x):return x*299+9
def func300(x):return x*300+0
def func301(x):return x*301+1
def func302(x):return x*302+2
def func303(x):return x*303+3
def func304(x):return x*304+4
def func305(x):return x*305+5
def func306(x):return x*306+6
def func307(x):return x*307+7
def func308(x):return x*308+8
def func309(x):return x*309+9
def func310(x):return x*310+0
def func311(x):return x*311+1
def func312(x):return x*312+2
def func313(x):return x*313+3
def func314(x):return x*314+4
def func315(x):return x*315+5
def func316(x):return x*316+6
def func317(x):return x*317+7
def func318(x):return x*318+8
def func319(x):return x*319+9
def func320(x):return x*320+0
def func321(x):return x*321+1
def func322(x):return x*322+2
def func323(x):return x*323+3
def func324(x):return x*324+4
def func325(x):return x*325+5
def func326(x):return x*326+6
def func327(x):return x*327+7
def func328(x):return x*328+8
def func329(x):return x*329+9
def func330(x):return x*330+0
def func331(x):return x*331+1
def func332(x):return x*332+2
def func333(x):return x*333+3
def func334(x):return x*334+4
def func335(x):return x*335+5
def func336(x):return x*336+6
def func337(x):return x*337+7
def func338(x):return x*338+8
def func339(x):return x*339+9
def func340(x):return x*340+0
def func341(x):return x*341+1
def func342(x):return x*342+2
def func343(x):return x*343+3
def func344(x):return x*344+4
def func345(x):return x*345+5
def func346(x):return x*346+6
def func347(x):return x*347+7
def func348(x):return x*348+8
def func349(x):return x*349+9
def func350(x):return x*350+0
def func351(x):return x*351+1
def func352(x):return x*352+2
def func353(x):return x*353+3
def func354(x):return x*354+4
def func355(x):return x*355+5
def func356(x):return x*356+6
def func357(x):return x*357+7
def func358(x):return x*358+8
def func359(x):return x*359+9
def func360(x):return x*360+0
def func361(x):return x*361+1
def func362(x):return x*362+2
def func363(x):return x*363+3
def func364(x):return x*364+4
def func365(x):return x*365+5
def func366(x):return x*366+6
def func367(x):return x*367+7
def func368(x):return x*368+8
def func369(x):return x*369+9
def func370(x):return x*370+0
def func371(x):return x*371+1
def func372(x):return x*372+2
def func373(x):return x*373+3
def func374(x):return x*374+4
def func375(x):return x*375+5
def func376(x):return x*376+6
def func377(x):return x*377+7
def func378(x):return x*378+8
def func379(x):return x*379+9
def func380(x):return x*380+0
def func381(x):return x*381+1
def func382(x):return x*382+2
def func383(x):return x*383+3
def func384(x):return x*384+4
def func385(x):return x*385+5
def func386(x):return x*386+6
def func387(x):return x*387+7
def func388(x):return x*388+8
def func389(x):return x*389+9
def func390(x):return x*390+0
def func391(x):return x*391+1
def func392(x):return x*392+2
def func393(x):return x*393+3
def func394(x):return x*394+4
def func395(x):return x*395+5
def func396(x):return x*396+6
def func397(x):return x*397+7
def func398(x):return x*398+8
def func399(x):return x*399+9
def func400(x):return x*400+0
def func401(x):return x*401+1
def func402(x):return x*402+2
def func403(x):return x*403+3
def func404(x):return x*404+4
def func405(x):return x*405+5
def func406(x):return x*406+6
def func407(x):return x*407+7
def func408(x):return x*408+8
def func409(x):return x*409+9
def func410(x):return x*410+0
def func411(x):return x*411+1
def func412(x):return x*412+2
def func413(x):return x*413+3
def func414(x):return x*414+4
def func415(x):return x*415+5
def func416(x):return x*416+6
def func417(x):return x*417+7
def func418(x):return x*418+8
def func419(x):return x*419+9
def func420(x):return x*420+0
def func421(x):return x*421+1
def func422(x):return x*422+2
def func423(x):return x*423+3
def func424(x):return x*424+4
def func425(x):return x*425+5
def func426(x):return x*426+6
def func427(x):return x*427+7
def func428(x):return x*428+8
def func429(x):return x*429+9
def func430(x):return x*430+0
def func431(x):return x*431+1
def func432(x):return x*432+2
def func433(x):return x*433+3
def func434(x):return x*434+4
def func435(x):return x*435+5
def func436(x):return x*436+6
def func437(x):return x*437+7
def func438(x):return x*438+8
def func439(x):return x*439+9
def func440(x):return x*440+0
def func441(x):return x*441+1
def func442(x):return x*442+2
def func443(x):return x*443+3
def func444(x):return x*444+4
def func445(x):return x*445+5
def func446(x):return x*446+6
def func447(x):return x*447+7
def func448(x):return x*448+8
def func449(x):return x*449+9
def func450(x):return x*450+0
def func451(x):return x*451+1
def func452(x):return x*452+2
def func453(x):return x*453+3
def func454(x):return x*454+4
def func455(x):return x*455+5
def func456(x):return x*456+6
def func457(x):return x*457+7
def func458(x):return x*458+8
def func459(x):return x*459+9
def func460(x):return x*460+0
def func461(x):return x*461+1
def func462(x):return x*462+2
def func463(x):return x*463+3
def func464(x):return x*464+4
def func465(x):return x*465+5
def func466(x):return x*466+6
def func467(x):return x*467+7
def func468(x):return x*468+8
def func469(x):return x*469+9
def func470(x):return x*470+0
def func471(x):return x*471+1
def func472(x):return x*472+2
def func473(x):return x*473+3
def func474(x):return x*474+4
def func475(x):return x*475+5
def func476(x):return x*476+6
def func477(x):return x*477+7
def func478(x):return x*478+8
def func479(x):return x*479+9
def func480(x):return x*480+0
def func481(x):return x*481+1
def func482(x):return x*482+2
def func483(x):return x*483+3
def func484(x):return x*484+4
def func485(x):return x*485+5
def func486(x):return x*486+6
def func487(x):return x*487+7
def func488(x):return x*488+8
def func489(x):return x*489+9
def func490(x):return x*490+0
def func491(x):return x*491+1
def func492(x):return x*492+2
def func493(x):return x*493+3
def func494(x):return x*494+4
def func495(x):return x*495+5
def func496(x):return x*496+6
def func497(x):return x*497+7
def func498(x):return x*498+8
def func499(x):return x*499+9
def func500(x):return x*500+0
