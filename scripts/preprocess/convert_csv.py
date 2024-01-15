import os
import subprocess

DATASET = 'dataset/KiMoRe'
for directory, subdirectories, files in os.walk(DATASET):
    for file in files:
        filepath = os.path.join(directory, file)
        if filepath.endswith('.xlsx'):

            target_filepath = filepath[:-4] + 'csv'
            print(f'converting {file} to {target_filepath}')

            if not subprocess.run(['xlsx2csv', filepath, target_filepath]):
                print(f'error converting {file}')
