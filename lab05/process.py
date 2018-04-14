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
        yerr=std_uncert_total,
        figsize=(6,8)
)
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
data_2['niepewność sin(alfa)'] = np.multiply(np.abs(np.cos(data_2['alfa (rad)'])), std_uncert['niepewność kąta (rad)'])
data_2['niepewność sin(beta)'] = np.multiply(np.abs(np.cos(data_2['beta (rad)'])), std_uncert['niepewność kąta (rad)'])
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
plt.plot(x, a*x + b, 'r', label='Metoda najmniejszych kwadratów')
plt.legend(loc=2)
plt.text(0.1, 0.6, f"a = {round(a, 6)}\nb = {round(b, 6)}", bbox=dict(facecolor='red', alpha=0.5))
plt.savefig('snell.png')

