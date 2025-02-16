import os
import hashlib


def get_md5(input_string):
    md5_obj = hashlib.md5()
    md5_obj.update(input_string.encode('utf-8'))
    return md5_obj.hexdigest()


def list_files_with_os(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.startswith('.'):
                file_list.append(os.path.join(root, file))
    return file_list
