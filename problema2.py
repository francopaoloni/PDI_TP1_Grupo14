import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Problema 2 - Validación de planilla de calificaciones

Para validar cada campo, se divide el problema en 3 partes:
    1) Detectar las líneas horizontales y verticales de la tabla, para obtener las celdas.
    2) Recortar cada registro (fila de la tabla) y cada uno de sus campos.
    3) Analizar cada campo: contar caracteres y palabras, y verificar las restricciones.
"""

CAMPOS = ['Legajo', 'Nombre y apellido', 'Parcial 1', 'Parcial 2', 'Parcial 3', 'Condición Final']


def inicio_fin(v):
    # v : Vector booleano con "pulsos" de valores TRUE.
    # Retorna una matriz donde cada fila contiene el [inicio, fin] de cada pulso.
    x = np.diff(v)
    idxs = np.argwhere(x)
    ii = np.arange(0, len(idxs), 2)  
    idxs[ii] += 1                       
    return idxs.reshape((-1, 2))    


def detectar_lineas(img_th):
    img_rows = np.sum(img_th, 1)
    img_cols = np.sum(img_th, 0)
    th_row = 0.5 * img_rows.max()
    th_col = 0.5 * img_cols.max()
    img_rows_th = img_rows > th_row
    img_cols_th = img_cols > th_col
    lineas_h = inicio_fin(img_rows_th)
    lineas_v = inicio_fin(img_cols_th)
    return lineas_h, lineas_v


def obtener_registros(img_th):
    lineas_h, lineas_v = detectar_lineas(img_th)

    registros = []
    for ir in range(1, len(lineas_h) - 1):
        y0 = lineas_h[ir][1] + 1       
        y1 = lineas_h[ir + 1][0] 
        campos = []
        cord_campos = []
        for ic in range(1, len(lineas_v) - 1):
            x0 = lineas_v[ic][1] + 1
            x1 = lineas_v[ic + 1][0]
            campos.append(img_th[y0:y1, x0:x1])
            cord_campos.append([y0, y1, x0, x1])
        registros.append({
            "id": ir,
            "cord": [y0, y1],
            "campos": campos,
            "cord_campos": cord_campos
        })
    return registros


def contar_caracteres(celda, th_area=2):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(celda.astype(np.uint8), 8, cv2.CV_32S)
    stats = stats[1:, :]             
    ix_area = stats[:, -1] > th_area
    stats = stats[ix_area, :]
    return stats.shape[0]


def contar_palabras(celda, th_espacio=6):
    col_zeros = celda.any(axis=0)
    if not col_zeros.any():
        return 0
    letras_indxs = inicio_fin(col_zeros)
    separaciones = letras_indxs[1:, 0] - letras_indxs[:-1, 1] - 1
    return 1 + np.sum(separaciones > th_espacio)


def validar_registro(campos):
    resultados = []
    for nombre, celda in zip(CAMPOS, campos):
        n_car = contar_caracteres(celda)
        n_pal = contar_palabras(celda)
        if nombre == 'Legajo':
            ok = n_car == 8 and n_pal == 1
        elif nombre == 'Nombre y apellido':
            ok = n_pal >= 2 and n_car <= 12
        elif nombre.startswith('Parcial'):
            ok = 1 <= n_car <= 2 and n_pal == 1
        else:  
            ok = n_car == 1
        resultados.append(ok)
    return resultados

def validar_planilla(img, th=150):
    img_th = img < th
    registros = obtener_registros(img_th)

    for registro in registros:
        if not any(celda.any() for celda in registro["campos"]):
            registro["resultados"] = None
            continue
        registro["resultados"] = validar_registro(registro["campos"])
        print(f'> Registro {registro["id"]}:')
        for nombre, ok in zip(CAMPOS, registro["resultados"]):
            print(f'>   {nombre}: {"OK" if ok else "MAL"}')
        print('>')
    return registros

def clasificar_condicion(celda, th_area=2):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(celda.astype(np.uint8), 8, cv2.CV_32S)
    ix_area = np.argwhere(stats[1:, -1] > th_area).flatten() + 1   # Índices de las componentes (sin el fondo)
    if len(ix_area) != 1:
        return None
    x, y, w, h, area = stats[ix_area[0]]
    letra = labels[y:y + h, x:x + w] == ix_area[0]   # Sub-imagen que contiene sólo la letra

    fondo = cv2.copyMakeBorder((~letra).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    nf, labels_f, stats_f, centroids_f = cv2.connectedComponentsWithStats(fondo, 4, cv2.CV_32S)
    agujeros = centroids_f[2:, :]                  

    col_izq_llena = letra[:, 0].all()
    if col_izq_llena and len(agujeros) == 0 and not letra[0, w // 2:].any():
        return 'L'
    if col_izq_llena and len(agujeros) == 1 and agujeros[0, 1] < (h + 2) / 2 and letra[-1, -1]:
        return 'R'
    return None


def generar_imagen_no_aprobados(img, registros, archivo_salida):
    colores = {'R': (0, 140, 255), 'L': (0, 0, 255)}   # BGR
    filas = []
    for registro in registros:
        if registro["resultados"] is None or not all(registro["resultados"]):
            continue
        condicion = clasificar_condicion(registro["campos"][5])
        if condicion not in colores:
            continue
        y0, y1, x0, x1 = registro["cord_campos"][1]
        nombre = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
        indicador = np.full((nombre.shape[0], 40, 3), colores[condicion], dtype=np.uint8)
        cv2.putText(indicador, condicion, (12, nombre.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        fila = np.hstack((indicador, nombre))
        fila = cv2.copyMakeBorder(fila, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        filas.append(fila)

    if len(filas) == 0:
        img_salida = np.full((30, 300, 3), 255, dtype=np.uint8)
        cv2.putText(img_salida, 'Sin alumnos desaprobados', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    else:
        img_salida = np.vstack(filas)
    cv2.imwrite(archivo_salida, img_salida)
    return img_salida


def generar_csv(registros, archivo_salida):
    with open(archivo_salida, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['ID'] + CAMPOS)
        for registro in registros:
            if registro["resultados"] is None:      # Registros vacíos
                continue
            writer.writerow([registro["id"]] + ['OK' if ok else 'MAL' for ok in registro["resultados"]])


"""
Se aplica el algoritmo, de forma cíclica, sobre las 4 planillas.
Por cada planilla se genera:
    * La validación de cada registro (por pantalla).
    * La imagen con los alumnos que no aprobaron (no_aprobados_<id>.png).
    * El archivo CSV con los resultados (resultados_<id>.csv).
"""
for id_planilla in range(1, 5):
    print(f'========== Planilla grade_sheet_{id_planilla}.png ==========')
    img = cv2.imread(f'grade_sheet_{id_planilla}.png', cv2.IMREAD_GRAYSCALE)
    registros = validar_planilla(img)
    img_salida = generar_imagen_no_aprobados(img, registros, f'no_aprobados_{id_planilla}.png')
    generar_csv(registros, f'resultados_{id_planilla}.csv')

    plt.figure(), plt.imshow(cv2.cvtColor(img_salida, cv2.COLOR_BGR2RGB))
    plt.title(f'grade_sheet_{id_planilla}.png - Alumnos que no aprobaron'), plt.axis('off')
plt.show()
