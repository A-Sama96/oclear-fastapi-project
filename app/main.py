#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import shutil
import sys
import traceback

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.logger import logger
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from aws import S3_CHECK_FOLDERNAME

import torch
import tensorflow as tf

from model import VGG16
from util import abs_path, relative_path_files_in_folder, remove_all_files_in_folder, remove_file, save_uploaded_zipfile, unzip_file_in_folder
from aws import download_all_files_in_folder_from_s3, S3_BUCKET_NAME, S3_SIGNATURE_FOLDERNAME
from oclear import detector
from predict import predict
from config import CONFIG
from exception_handler import validation_exception_handler, python_exception_handler
from schema import *

CHEKS_FOLDERNAME = "./images/cheks"
SIGNATURES_FOLDERNAME = "./images/signatures"

# Initialize API Server
app = FastAPI(
    title="Oclear Check Reader",
    description="Modèle ML permettant de lire les chèques",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None
)

# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Mount static folder, like demo pages, if any
app.mount("/static", StaticFiles(directory="static/"), name="static")

# Load custom exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, python_exception_handler)


@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """

    logger.info('Running environment: {}'.format(CONFIG['ENV']))
    logger.info('PyTorch using device: {}'.format(CONFIG['DEVICE']))

    # Initialize the pytorch model
    device = torch.device(CONFIG['DEVICE'])
    vgg = VGG16(29)
    vgg.load_state_dict(torch.load('../models/best.pth', map_location=device))

    # add model and other preprocess tools too app state
    app.package = {
        "model": vgg
    }



@app.post('/api/v1/report',
          response_model=InferenceResponse,
          responses={422: {"model": ErrorResponse},
                     500: {"model": ErrorResponse}}
          )
async def do_report(file: UploadFile = File(description="A file read as UploadFile")):
    """
    Perform OCR on input data
    """

    logger.info('API Report called')
    logger.info(f'File name: {file.filename}')
    await save_uploaded_zipfile(file)
    logger.info(f'{file.filename} was uploaded')

    logger.info(f'Start decompress {file.filename}')
    rlv_path_to_file = './images/' + file.filename
    unzip_file_in_folder(rlv_path_to_file)
    logger.info(f'Finish decompress {file.filename}')

    remove_file(rlv_path_to_file)

    # get all relative path of checks and signatures
    if os.path.isdir(abs_path(CHEKS_FOLDERNAME)) and os.path.isdir(abs_path(SIGNATURES_FOLDERNAME)):
        checks_rlv_path = relative_path_files_in_folder(CHEKS_FOLDERNAME)
        signatures_rlv_path = relative_path_files_in_folder(SIGNATURES_FOLDERNAME)
    else:
        logger.error("The cheks or signatures folder does not exist")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='The cheks or signatures folder does not exist')
    results = []
    errorReadCheckName = []
    errorRead = False
    for check in checks_rlv_path:
        d = detector(check, app.package['model'])
        try:
            val_criteria = InferenceResult(
                file_name=check.split('/')[-1],
                is_crossed=d.detect_bar(),
                amount_letter=d.montant_lettre(),
                amount_number=d.montant_chiffre(),
                location=d.pred_place(),
                date=d.pred_date(),
                amounts_compliance=d.conforme(),
                # name_recipient=' '.join(d.name),
                endorsable=True,
                signature_check=d.verif_sign(signatures_rlv_path)
            )
            results.append(val_criteria.dict())
        except:
            errorRead = True
            errorReadCheckName.append(check.split('/')[-1])
            logger.warning('Error reading with check : '+check.split('/')[-1])

    logger.info(f'results: {results}')

    logger.info("Star remove all local cheks and signatures ...")
    shutil.rmtree(CHEKS_FOLDERNAME, ignore_errors=True)
    shutil.rmtree(SIGNATURES_FOLDERNAME, ignore_errors=True)
    logger.info("Done: all local cheks and signatures are deleted !")

    return {
        "error": errorRead,
        "results": results,
        "errorReadChecksName": errorReadCheckName
    }


@app.get('/about')
def show_about():
    """
    Get deployment information, for debugging
    """

    def bash(command):
        output = os.popen(command).read()
        return output

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
        "tf.__version__": tf.__version__,
        "tf.config.list_physical_devices('GPU')": tf.config.list_physical_devices('GPU'),
        "nvidia-smi": bash('nvidia-smi')
    }


# if __name__ == '__main__':
#     # server api
#     uvicorn.run("main:app", host="0.0.0.0", port=8080,
#                 reload=True, debug=True, log_config="log.ini"
#                 )
