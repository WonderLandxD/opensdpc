import os
import csv

# data_folder path
folder_path = '..' # your folder  


sdpc_files = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(('.svs', '.sdpc', '.tiff', '.tif', '.ndpi')):
            full_path = os.path.join(root, file) 
            sdpc_files.append(full_path)

csv_file_path = 'process_wsi.csv' # your saved csv file
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for filename in sdpc_files:
        writer.writerow([filename])  

print(f"Saved in {csv_file_path}")

