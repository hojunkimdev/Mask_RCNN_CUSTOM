import os
import shutil

root_dir = 't'
merge_path = os.path.join(root_dir, 'merge')

if not os.path.isdir(merge_path):
    os.makedirs(merge_path)
for root, dirs, files in os.walk(root_dir):
    for file in files:
        file_path = os.path.join(root, file)
        print()
        shutil.move(file_path, merge_path)
