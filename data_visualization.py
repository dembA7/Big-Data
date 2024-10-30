import pandas as pd

# Cargar el dataset
df = pd.read_csv('data/fashion-mnist_test.csv')

# Convertir de wide a long format
df_long = df.melt(id_vars=["label"], 
                  var_name="pixel", 
                  value_name="intensity")

# Extraer las coordenadas x, y a partir del número del píxel
df_long['x'] = df_long['pixel'].str.extract(r'(\d+)').astype(int) % 28
df_long['y'] = df_long['pixel'].str.extract(r'(\d+)').astype(int) // 28

# Guardar en un nuevo CSV
df_long.to_csv('data/fashion_mnist_long.csv', index=False)
