import numpy as np
import matplotlib.pyplot as plt

def wykres(a, x, y, xmin=None, xmax=None, num_points=500): # rysuje wykres wielomianu, a - wpolczynniki, x - wartosci x, y - wartosci y
    x = np.array(x)
    y = np.array(y)

    x_plot = np.linspace(xmin, xmax, num_points)
    y_plot = np.zeros_like(x_plot)

    power = 0
    for c in a:
        y_plot += c * x_plot**power
        power+=1

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

def Cholesky(A): # zwraca dolna macierz trojkatna, rozkladu macierzy symetrycznej A
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)

    L[0][0] = np.sqrt(A[0][0])
    for i in range(1, n):
        L[i][0] = A[i][0] / L[0][0]
    # mamy ustawiona pierwsza kolumne

    # Uzupełniamy resztę macierzy
    for i in range(1, n):
        sum_sq = sum(L[i][k] ** 2 for k in range(i))
        L[i][i] = np.sqrt(A[i][i] - sum_sq)
        for j in range(i + 1, n):
            sum_prod = sum(L[j][k] * L[i][k] for k in range(i))
            L[j][i] = (A[j][i] - sum_prod) / L[i][i]
    return L


def forward_substitution(L, b): # rozwiazuje układ dolnotrójkątny
    n = len(b)
    y = np.zeros_like(b)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i][i]
    return y

def backward_substitution(U, y): # rozwiazuje uklad gornotrokatny
    n = len(y)
    x = np.zeros_like(y)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i][i]
    return x

def rozwiaz_uklad_cholesky(Ata, Y):
    # Rozkład Cholesky'ego: Ata = L * L^T
    L = Cholesky(Ata)

    # Rozwiąż L * z = Y (układ dolnotrójkątny)
    z = forward_substitution(L, Y)

    # Rozwiąż L^T * a = z (układ górnotrójkątny)
    LT = L.transpose() # transpozycja L
    a = backward_substitution(LT, z)

    return a

def Papinski_Bartosc_MNK(X, Y, n):
    array = []

    for x in X:
        row = [x ** i for i in range(n + 1)]
        array.append(row)
    A = np.array(array)

    print("macierz A:")
    print(A)

    At = A.transpose()
    print("transponowana macierz A:")
    print(At)

    Ata = At.dot(A)
    print("At * A")
    print(Ata)

    # prawa strona rownania

    Prawa = At.dot(np.array(Y))

    print("prawa strona rownania")
    print(Prawa)

    a = rozwiaz_uklad_cholesky(Ata, Prawa)
    return a



# wielomian stopnia 2 f(x) = 2x^2 - 3x + 1

x = [-2, -1, 0, 1, 2]
y = [15, 6, 1, 0, 1]

a = Papinski_Bartosc_MNK(x, y, 2)
print("Współczynniki wielomianu:", a)
wykres(a, x, y, xmin=-3, xmax=3)


# wielomian stopnia 3 f(x) = 2x^3 - x^2 + 3x + 1

x = [-2, -1, 0, 1, 2]
y = [-25, -5, 1, 5, 19]

a = Papinski_Bartosc_MNK(x, y, 3)
print("Współczynniki wielomianu:", a)
wykres(a, x, y, xmin=-3, xmax=3)

# wielomian stopnia 4

x = [-2, -1, 0, 1, 2]
y = [-25, -5, 1, 5, 19]

a = Papinski_Bartosc_MNK(x, y, 4)
print("Współczynniki wielomianu:", a)
wykres(a, x, y, xmin=-3, xmax=4)


# wielomian 5 stopnia

x = [-3, -2, -1, 0, 1, 2]
y = [-448, -81, -8, -1, 0, 7]

a = Papinski_Bartosc_MNK(x, y, 5)
print("Współczynniki wielomianu:", a)
wykres(a, x, y, xmin=-4, xmax=3)

