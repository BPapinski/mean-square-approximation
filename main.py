import numpy as np
import matplotlib.pyplot as plt

def rysuj_wielomian_z_punktami(a, x, y, xmin=None, xmax=None, num_points=500):
    """
    Rysuje wykres wielomianu na podstawie współczynników `a`
    oraz zaznacza punkty z tablic `x` i `y`.
    """
    x = np.array(x)
    y = np.array(y)

    if xmin is None:
        xmin = min(x) - 1
    if xmax is None:
        xmax = max(x) + 1

    x_plot = np.linspace(xmin, xmax, num_points)
    y_plot = np.zeros_like(x_plot)

    for i, coeff in enumerate(a):
        y_plot += coeff * x_plot**i

    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, y_plot, label='Wielomian aproksymujący', color='blue')
    plt.scatter(x, y, color='red', label='Punkty dane', zorder=5)
    plt.title('Wielomian i punkty aproksymowane')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='gray', lw=1)
    plt.axvline(0, color='gray', lw=1)
    plt.grid(True)
    plt.legend()
    plt.show()

def cholesky_decomposition(A):
    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1):
            sum_ = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = np.sqrt(A[i][i] - sum_)
            else:
                L[i][j] = (A[i][j] - sum_) / L[j][j]
    return L

def forward_substitution(L, b):
    n = len(b)
    y = np.zeros_like(b)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i][i]
    return y

def backward_substitution(U, y):
    n = len(y)
    x = np.zeros_like(y)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i][i]
    return x

def rozwiaz_uklad_cholesky(Ata, Y):
    # Rozkład Cholesky'ego: Ata = L * L^T
    L = cholesky_decomposition(Ata)

    # Rozwiąż L * z = Y (układ dolnotrójkątny)
    z = forward_substitution(L, Y)

    # Rozwiąż L^T * a = z (układ górnotrójkątny)
    LT = L.T
    a = backward_substitution(LT, z)

    return a

def Papinski_Bartosc_MNK(X, Y, n):
    array = []

    for x in X:
        array.append([1, x, x**2])

    A = np.array(array)

    print("macierz A:")
    print(A)

    At = A.transpose()
    print("transponowana macierz A:")
    print(At)

    print("At * A")

    Ata = At.dot(A)
    print(Ata)
    # Ata - jest dobrze


    # prawa strona rownania
    Yarray = []

    #zamiast tej petli mozna zrobic transpozycje jakos, ale chyba sie nie da bo tablica wejsciowa to Y = [1, 2 ,3] a nie [[1, 2, 3]]
    for y in Y:
        Yarray.append([y])

    Prawa = At.dot(Yarray)

    print("prawa strona rownania")
    print(Prawa)

    a = rozwiaz_uklad_cholesky(Ata, Prawa)
    print("Rozwiązanie a:", a)
    return a


x = [-1, 0, 1, 2]
y = [4, -1, 0, 7]
a = Papinski_Bartosc_MNK(x, y, 3)
print("Współczynniki wielomianu:", a)
rysuj_wielomian_z_punktami(a, x, y)


x = [-1, 2, 4]
y = [-1, 2, -16]
a = Papinski_Bartosc_MNK(x, y, 2)
print("Współczynniki wielomianu:", a)
rysuj_wielomian_z_punktami(a, x, y)