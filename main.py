import numpy as np
from numpy import prod
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon

# Wczytanie danych
data = np.loadtxt("136684.dat", delimiter="\t")
x, y, z = data[:, 0], data[:, 1], data[:, 2]

# Tworzenie siatki
x_grid, y_grid = np.meshgrid(np.unique(x), np.unique(y))
z_grid = z.reshape(x_grid.shape)

# Wizualizacja 2D
plt.figure(figsize=(8, 8))
plt.imshow(z_grid, cmap='viridis', extent=(x.min(), x.max(), y.min(), y.max()), origin='lower')
plt.tripcolor(x, y, z, cmap='viridis')
plt.xticks(np.arange(0, 2.01, 0.1))
plt.yticks(np.arange(0, 1.01, 0.1))
plt.colorbar(label='Z')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mapa 2D', fontdict={'fontname': 'monospace', 'fontsize': 20})
plt.show()

# Wizualizacja 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Mapa 3D', fontdict={'fontname': 'monospace', 'fontsize': 20})
ax.set_box_aspect([2, 1, 1])
ax.auto_scale_xyz([x.min(), x.max()], [y.min(), y.max()], [z.min(), z.max()])
plt.show()

# Statystyki dla Y
import numpy as np

def select_with_Y(data, y_level):
    mask = data[:, 1] == y_level
    return data[mask, 2]  # Pobranie wartości Z dla danego Y

for i in range(11):
    y_level = y[i * 21] 
    Z = select_with_Y(data, y_level)

    if Z.size == 0:  # Obsługa przypadku braku danych
        continue

    total_sum = np.sum(Z)
    mean = np.mean(Z)
    median = np.median(Z)
    standard_deviation = np.std(Z, ddof=0)  # Odchylenie dla całej populacji

    print("\nDla y =", y_level)
    print("Średnia:", mean)
    print("Mediana:", median)
    print("Odchylenie standardowe:", standard_deviation)

# Funkcje interpolacyjne (Lagrange'a)
def select_with_Y(data, y_level, start_index):
    mask = data[:, 1] == y_level
    filtered_data = data[mask]
    
    if start_index + 6 > filtered_data.shape[0]:
        raise ValueError("Indeks wykracza poza dostępne punkty.")

    return filtered_data[start_index:start_index+6, 0], filtered_data[start_index:start_index+6, 2]

def lagrange_value(k, X, X_t):
    return prod(X_t - np.delete(X, k))

def denominator(k, X):
    return prod(X[k] - np.delete(X, k))

def lagrange_interp(X, Y):
    A = Y / np.array([denominator(i, X) for i in range(len(X))])
    X_t = np.linspace(X.min(), X.max(), 100)

    Y_t = np.array([
        sum(A[j] * lagrange_value(j, X, xt) for j in range(len(X)))
        for xt in X_t
    ])
    return X_t, Y_t

def interpolate_segments():
    plt.figure(figsize=(8, 5)) 

    colors = ['#9B59B6', '#ff7f0e', '#2ca02c', '#d62728']
    y_level = 0.8
    
    for n in range(4):
        start_index = max(0, n * 6 - n)
        try:
            X, Y = select_with_Y(data, y_level, start_index)
        except ValueError:
            continue

        lagrange_x, lagrange_y = lagrange_interp(X, Y)

        plt.plot(lagrange_x, lagrange_y, label=f'Przedział {n+1}', linewidth=1.2, color=colors[n])
        plt.scatter(X, Y, color=colors[n], s=20, alpha=0.7)

    plt.title(f'Interpolacja Lagrange’a dla y = {y_level}', fontsize=13, fontweight='medium')
    plt.xticks(np.arange(0, 2.01, 0.2))
    plt.xlabel('X', fontsize=11)
    plt.ylabel('Z', fontsize=11)
    plt.grid(True, linestyle=':', linewidth=0.6, alpha=0.6)
    plt.legend(fontsize=9, frameon=False)
    plt.show()
    
y_level = 0.8

interpolate_segments()
# Funkcje interpolacyjne (Trygonometryczna)

def select_data_by_y_for_tryg(data, y_level, start_index):
    mask = data[:, 1] == y_level
    filtered_data = data[mask]

    if start_index + 21 > filtered_data.shape[0]:
        raise ValueError("Indeks wykracza poza dostępne punkty.")

    return filtered_data[start_index:start_index + 21, 0], filtered_data[start_index:start_index + 21, 2]

def transpose_tryg(mA):
    result = np.zeros((mA.shape[1], mA.shape[0]))
    for x in range(mA.shape[0]):
        for y in range(mA.shape[1]):
            result[y][x] = mA[x][y]
    return result

def trig_interpolation(X, Y, points_in_segment, segment):
    n = X.shape[0]
    x = np.zeros((n))
    mX = np.zeros((n, n))

    for i in range(n):
        x[i] = (2 * i * np.pi) / n

    y = np.copy(Y) * x[1] * 10

    for i in range(n):
        mX[i][0] = 1 / np.sqrt(2)
    for i in range(n):
        for j in range(1, n, 2):
            mX[i][j] = np.sin(x[i] * ((j + 1) / 2))
            mX[i][j + 1] = np.cos(x[i] * ((j + 1) / 2))

    mX *= 2 / n
    mR = transpose_tryg(mX)

    A = np.zeros((n))
    for i in range(n):
        for j in range(n):
            A[i] += mR[i][j] * y[j]

    xt = np.linspace(0, x[n - 1], points_in_segment)
    yt = np.zeros(xt.shape[0])

    for i in range(xt.shape[0]):
        yt[i] = A[0] / np.sqrt(2)
        for j in range(1, n, 2):
            t = int((j + 1) / 2)
            yt[i] += A[j] * np.sin(xt[i] * t) + A[j + 1] * np.cos(xt[i] * t)

    text = ""
    for i in range(A.shape[0]):
        text += "a{}={}		".format(i, A[i])

    return yt / (x[1] * 10), np.linspace(X.min(), X.max(), points_in_segment)

def plot_trig_interpolation():
    plt.figure(figsize=(10, 6))

    try:
        X, Y = select_data_by_y_for_tryg(data, y_level, 0)
    except ValueError:
        print("Nieprawidłowe dane lub indeks.")
        return

    y_t, x_t = trig_interpolation(X, Y, 100, 1)

    plt.plot(X, Y, 'o', label='Dane', markersize=6, color='#9B59B6')
    plt.plot(x_t, y_t, '-', label='Interpolacja trygonometryczna', linewidth=2, color='#1ABC9C')

    plt.xlabel('x', fontsize=12)
    plt.ylabel('z', fontsize=12)
    plt.title(f'Interpolacja trygonometryczna dla y = {y_level}', fontdict={'fontname': 'monospace', 'fontsize': 18})
    plt.legend(fontsize=10, frameon=False)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.show()
    
y_level = 0.8

plot_trig_interpolation()

# Funkcje aproksymacyjne (Liniowa)

def select_data_by_y(data, y_level, start_index):
    mask = data[:, 1] == y_level
    filtered_data = data[mask]

    if start_index + 21 > filtered_data.shape[0]:
        raise ValueError("Indeks wykracza poza dostępne punkty.")

    return filtered_data[start_index:start_index + 21, 0], filtered_data[start_index:start_index + 21, 2]

def linear_least_squares(x, y):
    s1 = np.sum(x)
    s2 = np.sum(y)
    s3 = np.sum(x ** 2)
    s4 = np.sum(x * y)

    n = len(x)
    w = n * s3 - s1 ** 2
    p0 = (s2 * s3 - s1 * s4) / w
    p1 = (n * s4 - s1 * s2) / w

    return p0, p1

def plot_linear_approximation():
    for n in range(1):
        try:
            X, Y = select_data_by_y(data, y_level, n * 21 - n)
        except ValueError:
            continue

        p0, p1 = linear_least_squares(X, Y)

        approx_x = np.linspace(X[0], X[-1], 100)
        approx_y = p0 + p1 * approx_x

        plt.plot(X, Y, 'o', label='Dane', markersize=6, color='#9B59B6')
        plt.plot(approx_x, approx_y, '-', label=f'Aproksymacja: z = {p0:.4f} + {p1:.4f}x', linewidth=2, color='#1ABC9C')

    plt.xlabel('x', fontsize=12)
    plt.ylabel('z', fontsize=12)
    plt.title(f'Aproksymacja liniowa dla y = {y_level}', fontdict={'fontname': 'monospace', 'fontsize': 18})
    plt.legend(fontsize=10, frameon=False)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.show()

def linear_approximation_for_all_segments():
    plt.figure(figsize=(10, 6))

    try:
        X, Y = select_data_by_y(data, y_level, 0)
    except ValueError:
        print("Nieprawidłowe dane lub indeks.")
        return

    p0, p1 = linear_least_squares(X, Y)

    approx_x = np.linspace(X[0], X[-1], 100)
    approx_y = p0 + p1 * approx_x

    plt.plot(X, Y, 'o', label='Dane', markersize=6, color='#9B59B6')
    plt.plot(approx_x, approx_y, '-', label=f'Aproksymacja: z = {p0:.4f} + {p1:.4f}x', linewidth=2, color='#1ABC9C')

    plt.xlabel('x', fontsize=12)
    plt.ylabel('z', fontsize=12)
    plt.title(f'Aproksymacja liniowa dla y = {y_level}', fontdict={'fontname': 'monospace', 'fontsize': 18})
    plt.legend(fontsize=10, frameon=False)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.show()

y_level = 0.8

linear_approximation_for_all_segments()

# Funkcje aproksymacyjne (Kwadratowa)

def select_data_by_y(data, y_level, start_index):
    mask = data[:, 1] == y_level
    filtered_data = data[mask]

    if start_index + 21 > filtered_data.shape[0]:
        raise ValueError("Indeks wykracza poza dostępne punkty.")

    return filtered_data[start_index:start_index + 21, 0], filtered_data[start_index:start_index + 21, 2]

def metoda_eliminacji_gaussa(mA):
    mI = np.zeros((mA.shape[0], mA.shape[1]))
    mI[:, :] = mA[:, :]
    n = mA.shape[1] - 1

    for s in range(n - 1):
        for i in range(s + 1, n):
            for j in range(s + 1, n + 1):
                mI[i][j] += (-(mI[i][s] / mI[s][s]) * mI[s][j])

    x = np.zeros((n))
    for i in range(n):
        x[i] = mI[i][i + 1] / mI[i][i]

    for i in range(n - 1, -1, -1):
        suma = 0
        for s in range(i + 1, n):
            suma += mI[i][s] * x[s]
        x[i] = (mI[i][n] - suma) / mI[i][i]

    return x

def aproksymacja_f1zmKwadrat(X, Y, points_in_segment, segment):
    m = np.zeros((X.shape[0], 2))
    for j in range(X.shape[0]):
        m[j][0] = X[j]
        m[j][1] = Y[j]

    mX = np.zeros((3, 4))
    c = 0
    for i in range(3):
        for j in range(3):
            for k in range(m.shape[0]):
                mX[i][j] += np.power(m[k][0], c)
            c += 1
        c -= 2

    mX[0][0] = m.shape[0]
    mX[0][3] = np.sum(m[:, 1])
    mX[1][3] = np.sum(m[:, 0] * m[:, 1])
    mX[2][3] = np.sum(np.power(m[:, 0], 2) * m[:, 1])

    a_org = metoda_eliminacji_gaussa(mX)
    a = np.around(a_org, 10)

    x_f = np.linspace(X.min(), X.max(), points_in_segment)
    y_f = a_org[2] * (x_f ** 2) + a_org[1] * x_f + a_org[0]

    return y_f, x_f

def quadratic_approximation_for_all_segments():
    plt.figure(figsize=(10, 6))

    try:
        X, Y = select_data_by_y(data, y_level, 0)
    except ValueError:
        print("Nieprawidłowe dane lub indeks.")
        return

    y_f, x_f = aproksymacja_f1zmKwadrat(X, Y, 100, 1)

    plt.plot(X, Y, 'o', label='Dane', markersize=6, color='#9B59B6')
    plt.plot(x_f, y_f, '-', label='Aproksymacja kwadratowa', linewidth=2, color='#1ABC9C')

    plt.xlabel('x', fontsize=12)
    plt.ylabel('z', fontsize=12)
    plt.title(f'Aproksymacja kwadratowa dla y = {y_level}', fontdict={'fontname': 'monospace', 'fontsize': 18})
    plt.legend(fontsize=10, frameon=False)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.show()

y_level = 0.8

quadratic_approximation_for_all_segments()

# Pole powierzchni funkcji

def calculate_triangle_area(point1, point2, point3, point4):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    x3, y3, z3 = point3
    x4, y4, z4 = point4
    ab = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** (1 / 2)
    bc = ((x2 - x3) ** 2 + (y2 - y3) ** 2 + (z2 - z3) ** 2) ** (1 / 2)
    cd = ((x3 - x4) ** 2 + (y3 - y4) ** 2 + (z3 - z4) ** 2) ** (1 / 2)
    da = ((x4 - x1) ** 2 + (y4 - y1) ** 2 + (z4 - z1) ** 2) ** (1 / 2)
    bd = ((x2 - x4) ** 2 + (y2 - y4) ** 2 + (z2 - z4) ** 2) ** (1 / 2)
    p1 = (ab + bd + da) / 2
    p2 = (bc + cd + bd) / 2
    area = (p1 * (p1 - ab) * (p1 - bd) * (p1 - da)) ** (1 / 2) + (p2 * (p2 - bc) * (p2 - cd) * (p2 - bd)) ** (1 / 2)
    return area

def calculate_surface_area(points):
    surface_area = 0.0
    num_triangles = len(points) - 22

    for i in range(num_triangles):
        point1 = points[i]
        point2 = points[i + 1]
        point3 = points[i + 21]
        point4 = points[i + 22]
        triangle_area = calculate_triangle_area(point1, point2, point3, point4)
        surface_area += triangle_area

    return surface_area

points = []
for i in range(len(x)):
    points.append((x[i], y[i], z[i]))
surface_area = calculate_surface_area(points)
print("\nPole powierzchni funkcji: \n", surface_area)

# Funkcje całkowania z funkcji interpolacyjnych i aproksymacyjnych

def rectangle_method(Y, dx):
    integral = 0 
    for i in range(len(Y)-1):
        integral += Y[i] * dx
    return integral

def trapezoid_method(Y, dx):
    integral = 0
    for i in range(len(Y)-1):
        integral += ((Y[i] + Y[i+1]) / 2) * dx
    return integral

def plot_intervals(data, y_level, segments=5):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    integral_rect_interp_lag = 0
    integral_trapz_interp_lag = 0

    for n in range(4):
        start_index = max(0, n * 6 - n)
        try:
            X, Y = select_with_Y(data, y_level, start_index)
        except ValueError:
            continue
        
        dx = (X.max() - X.min()) / segments
        x_interp_lag, y_interp_lag = lagrange_interp(X, Y)
        
        # Rysowanie prostokątów
        for i in range(segments):
            rect_x = X.min() + i * dx
            rect_y = np.interp(rect_x, x_interp_lag, y_interp_lag)
            rect = Polygon([[rect_x, 0], 
                            [rect_x + dx, 0], 
                            [rect_x + dx, rect_y], 
                            [rect_x, rect_y]], 
                            closed=True, facecolor='#2ECC71', alpha=0.5)
            axs[0].add_patch(rect)
            integral_rect_interp_lag += rect_y * dx
        
        axs[0].plot(x_interp_lag, y_interp_lag, 'r-', label=f'Przedział {n+1}')
        axs[0].scatter(X, Y, s=20, alpha=0.7, color='#9B59B6')
        axs[0].set_title('Całka metodą prostokątów (Interpolacja Lagrange’a)', fontsize=13, fontweight='medium')
        axs[0].set_xlabel('X', fontsize=11)
        axs[0].set_ylabel('Z', fontsize=11)
        axs[0].legend(fontsize=9, frameon=False)
        axs[0].grid(True, linestyle=':', linewidth=0.6, alpha=0.6)
        
        # Rysowanie trapezów
        trapezoids_segments = segments
        for i in range(trapezoids_segments):
            trap_x1 = X.min() + i * dx
            trap_x2 = trap_x1 + dx
            trap_y1 = np.interp(trap_x1, x_interp_lag, y_interp_lag)
            trap_y2 = np.interp(trap_x2, x_interp_lag, y_interp_lag)
            trap = Polygon([[trap_x1, 0], 
                            [trap_x2, 0], 
                            [trap_x2, trap_y2], 
                            [trap_x1, trap_y1]], 
                            closed=True, facecolor='#2ECC71', alpha=0.5)
            axs[1].add_patch(trap)
            integral_trapz_interp_lag += ((trap_y1 + trap_y2) / 2) * dx
            
        axs[1].plot(x_interp_lag, y_interp_lag, 'b-', label=f'Przedział {n+1}')
        axs[1].scatter(X, Y, s=20, alpha=0.7, color='#9B59B6')
        axs[1].set_title('Całka metodą trapezów (Interpolacja Lagrange’a)', fontsize=13, fontweight='medium')
        axs[1].set_xlabel('X', fontsize=11)
        axs[1].set_ylabel('Z', fontsize=11)
        axs[1].legend(fontsize=9, frameon=False)
        axs[1].grid(True, linestyle=':', linewidth=0.6, alpha=0.6)    

    plt.tight_layout()
    plt.show()
    
    print("\nCałka oznaczona dla interpolacji Lagrange’a (metoda prostokątów):", integral_rect_interp_lag)
    print("Całka oznaczona dla interpolacji Lagrange’a (metoda trapezów):", integral_trapz_interp_lag)

y_level = 0.8
plot_intervals(data, y_level)

def plot_intervals_trig(data, y_level, segments=20):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    integral_rect_interp_trig = 0
    integral_trapz_interp_trig = 0

    try:
        X, Y = select_data_by_y_for_tryg(data, y_level, 0)
    except ValueError:
        print("Nieprawidłowe dane lub indeks.")
        return

    dx = (X.max() - X.min()) / segments
    y_interp_trig, x_interp_trig = trig_interpolation(X, Y, 100, 1)

    # Rysowanie prostokątów
    for i in range(segments):
        rect_x = X.min() + i * dx
        rect_y = np.interp(rect_x, x_interp_trig, y_interp_trig)
        rect = Polygon([[rect_x, 0], 
                        [rect_x + dx, 0], 
                        [rect_x + dx, rect_y], 
                        [rect_x, rect_y]], 
                        closed=True, facecolor='#2ECC71', alpha=0.5)
        axs[0].add_patch(rect)
        integral_rect_interp_trig += rect_y * dx

    axs[0].plot(x_interp_trig, y_interp_trig, 'r-', label='Interpolacja trygonometryczna')
    axs[0].scatter(X, Y, s=20, alpha=0.7, color='#9B59B6')
    axs[0].set_title('Całka metodą prostokątów (Interpolacja trygonometryczna)', fontsize=13, fontweight='medium')
    axs[0].set_xlabel('X', fontsize=11)
    axs[0].set_ylabel('Z', fontsize=11)
    axs[0].legend(fontsize=9, frameon=False)
    axs[0].grid(True, linestyle=':', linewidth=0.6, alpha=0.6)

    # Rysowanie trapezów
    for i in range(segments):
        trap_x1 = X.min() + i * dx
        trap_x2 = trap_x1 + dx
        trap_y1 = np.interp(trap_x1, x_interp_trig, y_interp_trig)
        trap_y2 = np.interp(trap_x2, x_interp_trig, y_interp_trig)
        trap = Polygon([[trap_x1, 0], 
                        [trap_x2, 0], 
                        [trap_x2, trap_y2], 
                        [trap_x1, trap_y1]], 
                        closed=True, facecolor='#2ECC71', alpha=0.5)
        axs[1].add_patch(trap)
        integral_trapz_interp_trig += ((trap_y1 + trap_y2) / 2) * dx

    axs[1].plot(x_interp_trig, y_interp_trig, 'b-', label='Interpolacja trygonometryczna')
    axs[1].scatter(X, Y, s=20, alpha=0.7, color='#9B59B6')
    axs[1].set_title('Całka metodą trapezów (Interpolacja trygonometryczna)', fontsize=13, fontweight='medium')
    axs[1].set_xlabel('X', fontsize=11)
    axs[1].set_ylabel('Z', fontsize=11)
    axs[1].legend(fontsize=9, frameon=False)
    axs[1].grid(True, linestyle=':', linewidth=0.6, alpha=0.6)

    plt.tight_layout()
    plt.show()

    print("\nCałka oznaczona dla interpolacji trygonometrycznej (metoda prostokątów):", integral_rect_interp_trig)
    print("Całka oznaczona dla interpolacji trygonometrycznej (metoda trapezów):", integral_trapz_interp_trig)

y_level = 0.8
plot_intervals_trig(data, y_level)

def plot_intervals_linear(data, y_level, segments=20):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    integral_rect_approx_linear = 0
    integral_trapz_approx_linear = 0

    try:
        X, Y = select_data_by_y(data, y_level, 0)
    except ValueError:
        print("Nieprawidłowe dane lub indeks.")
        return

    dx = (X.max() - X.min()) / segments
    p0, p1 = linear_least_squares(X, Y)

    approx_x = np.linspace(X[0], X[-1], 100)
    approx_y = p0 + p1 * approx_x

    # Rysowanie prostokątów
    for i in range(segments):
        rect_x = X.min() + i * dx
        rect_y = p0 + p1 * rect_x
        rect = Polygon([[rect_x, 0], 
                        [rect_x + dx, 0], 
                        [rect_x + dx, rect_y], 
                        [rect_x, rect_y]], 
                        closed=True, facecolor='#2ECC71', alpha=0.5)
        axs[0].add_patch(rect)
        integral_rect_approx_linear += rect_y * dx

    axs[0].plot(approx_x, approx_y, 'r-', label='Aproksymacja liniowa')
    axs[0].scatter(X, Y, s=20, alpha=0.7, color='#9B59B6')
    axs[0].set_title('Całka metodą prostokątów (Aproksymacja liniowa)', fontsize=13, fontweight='medium')
    axs[0].set_xlabel('X', fontsize=11)
    axs[0].set_ylabel('Z', fontsize=11)
    axs[0].legend(fontsize=9, frameon=False)
    axs[0].grid(True, linestyle=':', linewidth=0.6, alpha=0.6)

    # Rysowanie trapezów
    for i in range(segments):
        trap_x1 = X.min() + i * dx
        trap_x2 = trap_x1 + dx
        trap_y1 = p0 + p1 * trap_x1
        trap_y2 = p0 + p1 * trap_x2
        trap = Polygon([[trap_x1, 0], 
                        [trap_x2, 0], 
                        [trap_x2, trap_y2], 
                        [trap_x1, trap_y1]], 
                        closed=True, facecolor='#2ECC71', alpha=0.5)
        axs[1].add_patch(trap)
        integral_trapz_approx_linear += ((trap_y1 + trap_y2) / 2) * dx

    axs[1].plot(approx_x, approx_y, 'b-', label='Aproksymacja liniowa')
    axs[1].scatter(X, Y, s=20, alpha=0.7, color='#9B59B6')
    axs[1].set_title('Całka metodą trapezów (Aproksymacja liniowa)', fontsize=13, fontweight='medium')
    axs[1].set_xlabel('X', fontsize=11)
    axs[1].set_ylabel('Z', fontsize=11)
    axs[1].legend(fontsize=9, frameon=False)
    axs[1].grid(True, linestyle=':', linewidth=0.6, alpha=0.6)

    plt.tight_layout()
    plt.show()

    print("\nCałka oznaczona dla aproksymacji liniowej (metoda prostokątów):", integral_rect_approx_linear)
    print("Całka oznaczona dla aproksymacji liniowej (metoda trapezów):", integral_trapz_approx_linear)

y_level = 0.8
plot_intervals_linear(data, y_level)

def plot_intervals_quadratic(data, y_level, segments=20):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    integral_rect_approx_quadratic = 0
    integral_trapz_approx_quadratic = 0

    try:
        X, Y = select_data_by_y(data, y_level, 0)
    except ValueError:
        print("Nieprawidłowe dane lub indeks.")
        return

    dx = (X.max() - X.min()) / segments
    y_f, x_f = aproksymacja_f1zmKwadrat(X, Y, 100, 1)

    # Rysowanie prostokątów
    for i in range(segments):
        rect_x = X.min() + i * dx
        rect_y = np.interp(rect_x, x_f, y_f)
        rect = Polygon([[rect_x, 0], 
                        [rect_x + dx, 0], 
                        [rect_x + dx, rect_y], 
                        [rect_x, rect_y]], 
                        closed=True, facecolor='#2ECC71', alpha=0.5)
        axs[0].add_patch(rect)
        integral_rect_approx_quadratic += rect_y * dx

    axs[0].plot(x_f, y_f, 'r-', label='Aproksymacja kwadratowa')
    axs[0].scatter(X, Y, s=20, alpha=0.7, color='#9B59B6')
    axs[0].set_title('Całka metodą prostokątów (Aproksymacja kwadratowa)', fontsize=13, fontweight='medium')
    axs[0].set_xlabel('X', fontsize=11)
    axs[0].set_ylabel('Z', fontsize=11)
    axs[0].legend(fontsize=9, frameon=False)
    axs[0].grid(True, linestyle=':', linewidth=0.6, alpha=0.6)

    # Rysowanie trapezów
    for i in range(segments):
        trap_x1 = X.min() + i * dx
        trap_x2 = trap_x1 + dx
        trap_y1 = np.interp(trap_x1, x_f, y_f)
        trap_y2 = np.interp(trap_x2, x_f, y_f)
        trap = Polygon([[trap_x1, 0], 
                        [trap_x2, 0], 
                        [trap_x2, trap_y2], 
                        [trap_x1, trap_y1]], 
                        closed=True, facecolor='#2ECC71', alpha=0.5)
        axs[1].add_patch(trap)
        integral_trapz_approx_quadratic += ((trap_y1 + trap_y2) / 2) * dx

    axs[1].plot(x_f, y_f, 'b-', label='Aproksymacja kwadratowa')
    axs[1].scatter(X, Y, s=20, alpha=0.7, color='#9B59B6')
    axs[1].set_title('Całka metodą trapezów (Aproksymacja kwadratowa)', fontsize=13, fontweight='medium')
    axs[1].set_xlabel('X', fontsize=11)
    axs[1].set_ylabel('Z', fontsize=11)
    axs[1].legend(fontsize=9, frameon=False)
    axs[1].grid(True, linestyle=':', linewidth=0.6, alpha=0.6)

    plt.tight_layout()
    plt.show()

    print("\nCałka oznaczona dla aproksymacji kwadratowej (metoda prostokątów):", integral_rect_approx_quadratic)
    print("Całka oznaczona dla aproksymacji kwadratowej (metoda trapezów):", integral_trapz_approx_quadratic)

y_level = 0.8
plot_intervals_quadratic(data, y_level)

#Pochodne cząstkowe dla wynego wiersza 

def select_data_by_y(data, y_level, start_index, n):
    X = np.zeros((n))
    Y = np.zeros((n))
    for i in range(data.shape[0]):
        if data[i][1] == y_level:
            for j in range(n):
                X[j] = data[i + j + start_index][0]
                Y[j] = data[i + j + start_index][2]
            break
    return X, Y

def compute_first_derivative(X, Y):
    n = len(Y)
    zxN = np.zeros_like(Y)
    for j in range(n):
        if j == 0:
            zxN[j] = (Y[j + 1] - Y[j]) / (X[j + 1] - X[j])
        elif j == n - 1:
            zxN[j] = (Y[j] - Y[j - 1]) / (X[j] - X[j - 1])
        else:
            zxN[j] = (Y[j + 1] - Y[j - 1]) / (X[j + 1] - X[j - 1])
    return zxN

def compute_second_derivative(X, zxN):
    n = len(zxN)
    zxxN = np.zeros_like(zxN)
    for j in range(n):
        if j == 0:
            zxxN[j] = (zxN[j + 1] - zxN[j]) / (X[j + 1] - X[j])
        elif j == n - 1:
            zxxN[j] = (zxN[j] - zxN[j - 1]) / (X[j] - X[j - 1])
        else:
            zxxN[j] = (zxN[j + 1] - zxN[j - 1]) / (X[j + 1] - X[j - 1])
    return zxxN

y_level = 0.8
n = 21
start_index = 0

X, Y = select_data_by_y(data, y_level, start_index, n)

zxN = compute_first_derivative(X, Y)
zxxN = compute_second_derivative(X, zxN)

# Wizualizacja pierwszej pochodnej
plt.figure(figsize=(8, 6))
plt.plot(X, Y, 'ro', label='Punkty', color='#9B59B6')
plt.plot(X, zxN, 'b-', label='Pierwsza pochodna', color='#1ABC9C')
plt.xlabel('x')
plt.ylabel("z'")
plt.title('Pierwsza pochodna dla y = {}'.format(y_level))
plt.legend()
plt.grid(True)
plt.show()

# Wizualizacja drugiej pochodnej
plt.figure(figsize=(8, 6))
plt.plot(X, Y, 'ro', label='Punkty', color='#9B59B6')
plt.plot(X, zxxN, 'r-', label='Druga pochodna', color='#1ABC9C')
plt.xlabel('x')
plt.ylabel("z''")
plt.title('Druga pochodna dla y = {}'.format(y_level))
plt.legend()
plt.grid(True)
plt.show()

# Funkcje analizy monotoniczności

def analyze_monotonicity(data, y_level, start_index, n):
    try:
        X, Y = select_data_by_y(data, y_level, start_index, n)
    except ValueError:
        print("Nieprawidłowe dane lub indeks.")
        return

    # Sprawdzanie monotoniczności
    is_increasing = all(Y[i] <= Y[i + 1] for i in range(len(Y) - 1))
    is_decreasing = all(Y[i] >= Y[i + 1] for i in range(len(Y) - 1))

    print("\nCzy rosnąca:", is_increasing)
    print("\nCzy malejąca:", is_decreasing)

    # Wykres
    plt.figure(figsize=(8, 6))
    plt.plot(X, Y, 'o-', label='Punkty', color='#1ABC9C' )
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(f'Monotoniczność dla y = {y_level}', fontdict={'fontname': 'monospace', 'fontsize': 18})

    if is_increasing:
        plt.text(0.1, 0.1, 'Funkcja jest rosnąca', transform=plt.gca().transAxes)
    elif is_decreasing:
        plt.text(0.1, 0.1, 'Funkcja jest malejąca', transform=plt.gca().transAxes)
    else:
        plt.text(0.1, 0.1, 'Funkcja nie jest monotoniczna', transform=plt.gca().transAxes)

    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.show()

# Użycie funkcji analizy monotoniczności
y_level = 0.8
start_index = 0
n = 21  # Ustal wartość n, aby wskazać segment danych do analizy
analyze_monotonicity(data, y_level, start_index, n)

