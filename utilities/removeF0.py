import os, shutil

ROOT_DIR = r"C:\Users\oriol\Downloads\embryo_dataset_\embryo_dataset"

testing = True # Activem o desactivem el mode testing.

# Contadors
deleted = 0
found = 0

# Es recorren totes les carpetes, subcarpetes incloses
for folder in os.listdir(ROOT_DIR):
    folder_path = os.path.join(ROOT_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    f0_path = os.path.join(folder_path, "F0")

    if os.path.exists(f0_path) and os.path.isdir(f0_path): # detecta carpeta f0
        found += 1

        if testing:
            print(f"[Testing] S'eliminaria: {f0_path}\n")
        else:
            shutil.rmtree(f0_path) # elimina carpeta f0
            print(f"[DELETED] {f0_path}\n")
            deleted += 1

# Informació final
print(f"Carpetes amb F0: {found}")
print(f"Carpetes eliminades: {deleted}")