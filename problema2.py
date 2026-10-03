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


# --- Inicio y fin de pulsos ----------------------------------------------------
def inicio_fin(v):
    # v : Vector booleano con "pulsos" de valores TRUE.
    # Retorna una matriz donde cada fila contiene el [inicio, fin] de cada pulso.
    # (Misma metodología que en el problema "Letras").
    x = np.diff(v)
    idxs = np.argwhere(x)
    ii = np.arange(0, len(idxs), 2)     # Los inicios están en los índices pares...
    idxs[ii] += 1                       # ... y se les suma 1 para que coincidan.
    return idxs.reshape((-1, 2))        # Cada fila contiene inicio y fin de un pulso


# -----------------------------------------------------------------------------
# --- PARTE 1: Detección de líneas de la tabla --------------------------------
# -----------------------------------------------------------------------------
def detectar_lineas(img_th):
    # img_th : Imagen umbralizada (TRUE donde hay píxeles oscuros).
    # Las líneas de la tabla tienen muchos más píxeles oscuros que el resto del formulario,
    # por lo que se suman los píxeles de cada fila/columna y se umbraliza.
    # Como las líneas pueden tener más de un píxel de ancho, se obtiene inicio y fin de cada una.
    img_rows = np.sum(img_th, 1)
    img_cols = np.sum(img_th, 0)
    th_row = 0.5 * img_rows.max()
    th_col = 0.5 * img_cols.max()
    img_rows_th = img_rows > th_row
    img_cols_th = img_cols > th_col
    lineas_h = inicio_fin(img_rows_th)
    lineas_v = inicio_fin(img_cols_th)
    return lineas_h, lineas_v


# -----------------------------------------------------------------------------
# --- PARTE 2: Recorte de registros y campos ----------------------------------
# -----------------------------------------------------------------------------
def obtener_registros(img_th):
    # img_th : Imagen umbralizada (TRUE donde hay píxeles oscuros).
    # Retorna una lista de diccionarios, uno por registro, con:
    #   * id     : índice del registro (orden en la planilla).
    #   * cord   : filas de inicio y fin del registro.
    #   * campos : lista con las sub-imágenes de los 6 campos.
    #   * cord_campos : lista con las coordenadas [y0, y1, x0, x1] de cada campo.
    lineas_h, lineas_v = detectar_lineas(img_th)

    registros = []
    # lineas_h[0] y lineas_h[1] delimitan el encabezado, luego vienen los registros.
    for ir in range(1, len(lineas_h) - 1):
        y0 = lineas_h[ir][1] + 1          # Fila siguiente al fin de la línea superior
        y1 = lineas_h[ir + 1][0]          # Fila de inicio de la línea inferior
        campos = []
        cord_campos = []
        # La primera columna (Nro.) no se analiza.
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


# -----------------------------------------------------------------------------
# --- PARTE 3: Análisis de cada campo -----------------------------------------
# -----------------------------------------------------------------------------
def contar_caracteres(celda, th_area=2):
    # celda   : Sub-imagen umbralizada del campo.
    # th_area : Área mínima de una componente para ser considerada caracter
    #           (elimina restos de las líneas divisorias de la tabla).
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(celda.astype(np.uint8), 8, cv2.CV_32S)
    stats = stats[1:, :]                # Se descarta el fondo (componente 0)
    ix_area = stats[:, -1] > th_area
    stats = stats[ix_area, :]
    return stats.shape[0]


def contar_palabras(celda, th_espacio=6):
    # celda      : Sub-imagen umbralizada del campo.
    # th_espacio : Cantidad mínima de columnas vacías entre dos caracteres para
    #              considerar que hay un espacio entre palabras.
    # Se analizan las columnas (como en el problema "Letras"): cada pulso es un caracter
    # y la separación entre pulsos consecutivos indica si hay un espacio.
    col_zeros = celda.any(axis=0)
    if not col_zeros.any():
        return 0
    letras_indxs = inicio_fin(col_zeros)
    separaciones = letras_indxs[1:, 0] - letras_indxs[:-1, 1] - 1
    return 1 + np.sum(separaciones > th_espacio)


def validar_registro(campos):
    # campos : Lista con las sub-imágenes de los 6 campos de un registro.
    # Retorna una lista con True (OK) / False (MAL) para cada campo.
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
        else:  # Condición Final
            ok = n_car == 1
        resultados.append(ok)
    return resultados


# --- a) Validación de una planilla -------------------------------------------
def validar_planilla(img, th=150):
    # img : Imagen de la planilla en escala de grises.
    # th  : Umbral para separar los píxeles oscuros (líneas y texto) del fondo.
    # Imprime, por cada registro, si cada campo es correcto (OK) o incorrecto (MAL).
    # Los registros vacíos (sin ningún caracter) no se informan.
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


# --- b) Imagen de salida con los alumnos que no aprobaron -------------------
def clasificar_condicion(celda, th_area=2):
    # celda : Sub-imagen umbralizada del campo Condición Final (con un único caracter).
    # Retorna 'L', 'R' u otro caracter (None), según la forma de la letra:
    #   * L : columna izquierda llena, sin agujeros y esquina superior derecha vacía.
    #   * R : columna izquierda llena, un agujero en la mitad superior y
    #         la "pata" llega a la esquina inferior derecha.
    #   (La 'A' no tiene la columna izquierda llena).
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(celda.astype(np.uint8), 8, cv2.CV_32S)
    ix_area = np.argwhere(stats[1:, -1] > th_area).flatten() + 1   # Índices de las componentes (sin el fondo)
    if len(ix_area) != 1:
        return None
    x, y, w, h, area = stats[ix_area[0]]
    letra = labels[y:y + h, x:x + w] == ix_area[0]   # Sub-imagen que contiene sólo la letra

    # Agujeros: componentes del fondo que no tocan el borde de la letra.
    # Se agrega un borde de fondo para que todo el exterior quede en una única componente.
    fondo = cv2.copyMakeBorder((~letra).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    nf, labels_f, stats_f, centroids_f = cv2.connectedComponentsWithStats(fondo, 4, cv2.CV_32S)
    agujeros = centroids_f[2:, :]                    # 0: la letra, 1: el exterior, 2...: agujeros

    col_izq_llena = letra[:, 0].all()
    if col_izq_llena and len(agujeros) == 0 and not letra[0, w // 2:].any():
        return 'L'
    if col_izq_llena and len(agujeros) == 1 and agujeros[0, 1] < (h + 2) / 2 and letra[-1, -1]:
        return 'R'
    return None


def generar_imagen_no_aprobados(img, registros, archivo_salida):
    # img            : Imagen de la planilla en escala de grises.
    # registros      : Registros obtenidos con validar_planilla().
    # archivo_salida : Nombre del archivo de la imagen de salida.
    # Genera una única imagen con el crop del campo Nombre y Apellido de los alumnos
    # que no aprobaron (Condición Final 'L' o 'R'), considerando sólo los registros
    # cargados correctamente. A la izquierda de cada nombre se agrega un indicador:
    #   * R (recupera) : recuadro naranja.
    #   * L (libre)    : recuadro rojo.
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


img = cv2.imread('grade_sheet_1.png', cv2.IMREAD_GRAYSCALE)
registros = validar_planilla(img)
img_salida = generar_imagen_no_aprobados(img, registros, 'no_aprobados.png')

plt.figure(), plt.imshow(cv2.cvtColor(img_salida, cv2.COLOR_BGR2RGB)), plt.title('Alumnos que no aprobaron'), plt.axis('off'), plt.show()
