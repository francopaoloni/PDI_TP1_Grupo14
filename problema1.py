import cv2
import numpy as np
import matplotlib.pyplot as plt


def ecualizacion_local(img, M, N):
    if M <= 0 or N <= 0:
        raise ValueError("M y N deben ser enteros positivos.")

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
            img_out[i, j] = np.uint8(np.round(255 * cdf[img[i, j]]))
    return img_out


img = cv2.imread('Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)
img_leq = ecualizacion_local(img, 15, 15)

ax1 = plt.subplot(121)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen Original')
plt.subplot(122, sharex=ax1, sharey=ax1)
plt.imshow(img_leq, cmap='gray', vmin=0, vmax=255)
plt.title('Ecualización local (15x15)')
plt.show()


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
