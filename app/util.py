#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil


def abs_path(p: str) -> str:
    """
    return the absolute path of a relative path
    """
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), p)


def relative_path_files_in_folder(folder_rlv_path: str) -> list:
    """
    return list of file's relative path in a folder
    """
    list_rlv_path = []
    t = os.listdir(folder_rlv_path)
    for i in t:
        list_rlv_path.append(os.path.join(folder_rlv_path, i))
    return list_rlv_path


def remove_all_files_in_folder(folder: str) -> None:
    """
    remove all files in a folder
    """
    abs_path_folder = abs_path(folder)
    for files in os.listdir(abs_path_folder):
        path = os.path.join(abs_path_folder, files)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)


remove_all_files_in_folder('./images/signatures')
