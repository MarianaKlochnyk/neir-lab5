import os
# Встав сюди шлях, який ти бачиш у провіднику Windows
path = r'data/raw/eye_disease/preprocessed_images'
print(f"Чи існує папка: {os.path.exists(path)}")
if os.path.exists(path):
    print(f"Перший файл у папці: {os.listdir(path)[0]}")