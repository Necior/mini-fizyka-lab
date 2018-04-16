import pandas
import matplotlib.pyplot as plt
import numpy as np
import math

data_1 = pandas.read_csv('pomiary/1-malus.csv')
data_2 = pandas.read_csv('pomiary/2-snell.csv')
data_3 = pandas.read_csv('pomiary/3-graniczny.csv')
data_4 = pandas.read_csv('pomiary/4-brewster.csv')
data_5 = pandas.read_csv('pomiary/5-polaryzacja.csv')

print(80*'-')
print(data_1.to_latex())
print(80*'-')
print(data_2.to_latex())

std_uncert_device = data_1['zakres(uA)'] / (100 * math.sqrt(3))
std_uncert_device.name = 'niepewność std klasa miernika (uA)'

std_uncert_disp = data_1['zakres(uA)'].transform(
        lambda x:  
            x / (60 * math.sqrt(3)) if x % 3 == 0 \
            else x / (50 * math.sqrt(3))
        )
std_uncert_disp.name = 'niepewność std podziałka (uA)'
std_uncert_read = std_uncert_disp / 2
std_uncert_read.name = 'niepewność std odczyt (uA)'

std_uncert_total = np.sqrt(np.power(std_uncert_device, float(2)) + np.power(std_uncert_disp, float(2)) + np.power(std_uncert_read, float(2)))
std_uncert_total.name = 'niepewność standartowa typu B (uA)'

# print(std_uncert_device)
# print(std_uncert_disp)
# print(std_uncert_read)
print(std_uncert_total)

table = data_1[[
    'kąt(deg)',
    'I (uA)'
    ]]
table.columns = [
    'kąt obrotu analizatora (deg)',
    'natężenie prądu (uA)'
    ]

#table['niepewność pomiarowa (uA)'] = std_uncert_total

print(table)
table.plot(
        title='Natężenie prądu fali docierającej do detektora od kąta analizatora',
        kind='scatter',
        s=8,
        x='kąt obrotu analizatora (deg)',
        y='natężenie prądu (uA)',
        xerr=np.divide(np.array([2, 2, 2, 2, 2, 2, 2, 2]), math.sqrt(3)),
        yerr=std_uncert_total,
        figsize=(6,8),
        label='pomiary'
)

x = np.linspace(60, 190, 100)
plt.plot(x, 262*np.power(np.cos(np.deg2rad(x+2)), 2), 'r-', label='262 * cos^2(x+2)')
plt.plot(x, 266*np.power(np.cos(np.deg2rad(x+9)), 2), 'g--', label='266 * cos^2(x+9)')
plt.legend(loc=2)
plt.savefig('malus.png')



print(data_2)
std_uncert = {'niepewność': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]}
std_uncert = pandas.DataFrame(std_uncert)
std_uncert = std_uncert.transform(lambda x: np.deg2rad(x)/math.sqrt(3))
std_uncert['niepewność kąta (rad)'] = std_uncert['niepewność']
std_uncert = std_uncert.drop(columns=['niepewność'])
data_2['alfa (rad)'] = data_2['alfa (deg)'].transform(lambda x: np.deg2rad(x))
data_2['beta (rad)'] = data_2['beta (deg)'].transform(lambda x: np.deg2rad(x))
data_2['sin(alfa)'] = data_2['alfa (rad)'].transform(lambda x: np.sin(x))
data_2['sin(beta)'] = data_2['beta (rad)'].transform(lambda x: np.sin(x))
data_2['niepewność sin(alfa)'] = np.abs(np.multiply(np.cos(data_2['alfa (rad)']), std_uncert['niepewność kąta (rad)']))
data_2['niepewność sin(beta)'] = np.abs(np.multiply(np.cos(data_2['beta (rad)']), std_uncert['niepewność kąta (rad)']))
x = data_2['sin(alfa)']
y = data_2['sin(beta)']
A = np.array([x, np.ones(7)])
a, b = np.linalg.lstsq(A.T, y, rcond=None)[0]
print(f"a = {a}, b = {b}")
data_2.plot(
        title='Sinus kąta odbicia w zależności od sinusa kąta padania',
        kind='scatter',
        s=10,
        x='sin(alfa)',
        y='sin(beta)',
        figsize=(8,8),
        xerr='niepewność sin(alfa)',
        yerr='niepewność sin(beta)'
        )
plt.plot(np.linspace(0,1.2,100), a*np.linspace(0, 1.2, 100) + b, 'r', label='Metoda najmniejszych kwadratów')
plt.legend(loc=2)
plt.axis((0,1,0,0.7))
plt.text(0.1, 0.6, f"a = {round(a, 6)}\nb = {round(b, 6)}", bbox=dict(facecolor='red', alpha=0.5))
plt.savefig('snell.png')
print(data_2.to_latex())
print(std_uncert.to_latex())

# Counting least squares fit w/ uncertainties by hand
print("Counting least squares fit by hand:")
x = np.array([0.087156,0.258819,0.5,0.707107,0.866025,0.939693,0.965926])
y = np.array([0.061049,0.182236,0.333807,0.469472,0.587785,0.629320,0.649448])
n = 7
xy = np.multiply(x, y)
a = (n*xy.sum() - x.sum() * y.sum()) / (n * np.power(x, 2).sum() - x.sum() **2 )
b = 1/n * (y.sum() - a*x.sum())
u_a = np.sqrt(n/(n-2) * (np.power(y, 2).sum() - a*xy.sum() - b*y.sum())/(n * np.power(x, 2).sum() - x.sum() ** 2))
u_b = u_a * np.sqrt(np.power(x, 2).sum()/n)
print(f"a={a}\nb={b}\nu_a={u_a}\nu_b={u_b}")

# Preparing summary plot
x = np.array([1, 2, 3])
y = np.array([1.500, 1.466, 1.483])
x_ticks = ['Prawo Snella', 'Kąt graniczny', 'Kąt Brewstera']
yerr = np.multiply(np.array([0.013, 0.035, 0.048]), 1)
x_exact = np.linspace(0, 4, 100)
y_exact = np.ones(100) * 1.4917

plt.figure()
plt.xticks(x, x_ticks)
plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=5)
plt.plot(x_exact, y_exact, 'g--')
plt.axis((0,4,1.42, 1.54))
plt.text(0.2, 1.495, '1.4917', color='g')
plt.savefig('wnioski.png')
