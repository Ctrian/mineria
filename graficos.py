import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("tracking_data.csv")
unique_ids = df["track_id"].nunique()
print(f"Personas únicas detectadas: {unique_ids}")


for pid in df["track_id"].unique():
    person = df[df["track_id"] == pid]
    plt.plot(person["x"], person["y"], label=f"ID {pid}")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Trayectorias de personas")
plt.legend()
plt.gca().invert_yaxis()  # si quieres que coincida con coordenadas de video
plt.show()

import seaborn as sns
import numpy as np

# Creamos un heatmap del área
heatmap = np.zeros((1080, 1920))  # ajusta al tamaño de tu video

for _, row in df.iterrows():
    x, y = int(row["x"]), int(row["y"])
    if 0 <= x < 1920 and 0 <= y < 1080:
        heatmap[y, x] += 1  # cuidado: y primero

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap, cmap="hot", cbar=True)
plt.title("Diagrama de calor de movimiento")
plt.gca().invert_yaxis()
plt.show()