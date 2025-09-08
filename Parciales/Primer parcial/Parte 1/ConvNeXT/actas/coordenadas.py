import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Cargar la imagen
imagen_path = "download.jpeg"  # Cambia por tu imagen
img = mpimg.imread(imagen_path)

coords = []  # Aquí guardaremos las coordenadas

def onclick(event):
    # Solo si el clic es dentro de los ejes
    if event.xdata is not None and event.ydata is not None:
        x, y = event.xdata, event.ydata
        coords.append((x, y))
        print(f"Clic en: x={x:.2f}, y={y:.2f}")
        # Dibuja un punto donde se hizo clic
        plt.plot(x, y, 'ro')
        plt.draw()

# Mostrar la imagen
fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title("Haz clic en las esquinas (izquierda-arriba y derecha-abajo)")
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

print("Coordenadas capturadas:", coords)
