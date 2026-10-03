import cv2
import numpy as np
import matplotlib.pyplot as plt


def ecualizacion_local(img, M, N):
    # img : Imagen de entrada en escala de grises (2D), formato uint8.
    # M   : Cantidad de filas de la ventana de procesamiento (entero positivo).
    # N   : Cantidad de columnas de la ventana de procesamiento (entero positivo).
    # Se desplaza la ventana de MxN píxel a píxel. En cada posición se calcula el
    # histograma de la ventana, su cdf, y se transforma únicamente el píxel central.
    if M <= 0 or N <= 0:
        raise ValueError("M y N deben ser enteros positivos.")

    # Se agregan bordes replicados para que la ventana siempre quede completa.
    # Si M o N son pares, el centro queda desplazado hacia arriba/izquierda.
    top, bottom = (M - 1) // 2, M // 2
    left, right = (N - 1) // 2, N // 2
    img_pad = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)

    img_out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ventana = img_pad[i:i + M, j:j + N]
            hist = cv2.calcHist([ventana], [0], None, [256], [0, 256])
            histn = hist.flatten().astype(np.double) / ventana.size
            cdf = histn.cumsum()
            img_out[i, j] = np.uint8(np.round(255 * cdf[img[i, j]]))   # Transformación aplicada al píxel central
    return img_out


# --- b) Análisis de la imagen con detalles ocultos -------------------------
img = cv2.imread('Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)
img_leq = ecualizacion_local(img, 15, 15)

ax1 = plt.subplot(121)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen Original')
plt.subplot(122, sharex=ax1, sharey=ax1)
plt.imshow(img_leq, cmap='gray', vmin=0, vmax=255)
plt.title('Ecualización local (15x15)')
plt.show()

# Los cuadrados negros (intensidad ~0) esconden detalles con intensidades muy
# cercanas a la del fondo local (entre 2 y 10). Con la ecualización local se observan:
#   - Arriba a la izquierda: un cuadrado pequeño.
#   - Arriba a la derecha:   una línea diagonal.
#   - Centro:                la letra "a".
#   - Abajo a la izquierda:  cuatro líneas horizontales.
#   - Abajo a la derecha:    un círculo.


# --- c) Influencia del tamaño de la ventana --------------------------------
tamanios = [3, 7, 15, 31, 51, 101]

plt.figure()
for k, t in enumerate(tamanios):
    img_leq = ecualizacion_local(img, t, t)
    if k == 0:
        ax1 = plt.subplot(2, 3, 1)
    else:
        plt.subplot(2, 3, k + 1, sharex=ax1, sharey=ax1)
    plt.imshow(img_leq, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Ventana {t}x{t}')
plt.show()

# - Ventanas chicas (3x3, 7x7): la ventana es más chica que los objetos, por lo que
#   sólo se realzan los bordes de los detalles (se ven "huecos"). También se
#   amplifica mucho el ruido del fondo.
# - Ventanas medianas (15x15, 31x31): los detalles se ven completos y con buen
#   contraste. Aparece un halo cerca de los bordes de los cuadrados, ya que la
#   ventana incluye parte del fondo claro.
# - Ventanas grandes (51x51, 101x101): el histograma de la ventana se parece cada vez
#   más al de toda la imagen y el resultado se acerca a una ecualización global:
#   los detalles pierden contraste y con 101x101 casi no se distinguen.
