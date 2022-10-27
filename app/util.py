#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import aiofiles
from fastapi import HTTPException, UploadFile, status
from zipfile import ZipFile
from fastapi.logger import logger



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


async def save_uploaded_zipfile(file: UploadFile,rlv_path_folder: str = "./images/", CHUNK_SIZE = 1024 * 1024):
    if file.content_type not in ["application/zip", "application/octet-stream", "application/x-zip-compressed"]:
        logger.error(f'File type of {file.content_type} is not supported')
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type of {file.content_type} is not supported",
        )
    try:
        filepath = os.path.join(rlv_path_folder, os.path.basename(file.filename))
        async with aiofiles.open(filepath, 'wb') as f:
            while chunk := await file.read(CHUNK_SIZE):
                await f.write(chunk)
    except Exception:
        logger.error('There was an error uploading the file')
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail='There was an error uploading the file')
    finally:
        await file.close()

def unzip_file_in_folder(path_to_zip_file, directory_to_extract_to = './images'):
    try:
        with ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
    except Exception:
        logger.error('There was an error unzipping the file')
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail='There was an error unzipping the file')

def remove_file(path_to_file):
    if os.path.exists(path_to_file):
        os.remove(path_to_file)
        logger.info(f'File deleted: {path_to_file}')
    else:
        logger.error("The file does not exist")