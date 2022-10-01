#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import traceback

import uvicorn
from fastapi import FastAPI, Request, status
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
from util import relative_path_files_in_folder, remove_all_files_in_folder
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
        # "scaler": load(CONFIG['SCALAR_PATH']),  # joblib.load
        "model": vgg
    }


# @app.post('/api/v1/predict',
#           response_model=InferenceResponse,
#           responses={422: {"model": ErrorResponse},
#                      500: {"model": ErrorResponse}}
#           )
# def do_predict(request: Request, body: InferenceInput):
#     """
#     Perform prediction on input data
#     """

#     logger.info('API predict called')
#     logger.info(f'input: {body}')

#     # prepare input data
#     X = [body.sepal_length, body.sepal_width,
#          body.petal_length, body.petal_width]

#     # run model inference
#     y = predict(app.package, [X])[0]

#     # generate prediction based on probablity
#     pred = ['setosa', 'versicolor', 'virginica'][y.argmax()]

#     # round probablities for json
#     y = y.tolist()
#     y = list(map(lambda v: round(v, ndigits=CONFIG['ROUND_DIGIT']), y))

#     # prepare json for returning
#     results = {
#         'setosa': y[0],
#         'versicolor': y[1],
#         'virginica': y[2],
#         'pred': pred
#     }

#     logger.info(f'results: {results}')

#     return {
#         "error": False,
#         "results": results
#     }

@app.post('/api/v1/report',
          response_model=InferenceResponse,
          responses={422: {"model": ErrorResponse},
                     500: {"model": ErrorResponse}}
          )
def do_report(request: Request, body: InferenceInput):
    """
    Perform OCR on input data
    """

    logger.info('API Report called')
    logger.info(f'input: {body}')

    # download all checks and signature from AWS s3
    logger.info('Download in local all cheks from S3 Bucket')
    download_all_files_in_folder_from_s3(
        S3_BUCKET_NAME, S3_CHECK_FOLDERNAME, CHEKS_FOLDERNAME)

    logger.info('Download in local all signatures from S3 Bucket')
    download_all_files_in_folder_from_s3(
        S3_BUCKET_NAME, S3_SIGNATURE_FOLDERNAME, SIGNATURES_FOLDERNAME)

    # get all relative path of checks and signatures
    checks_rlv_path = relative_path_files_in_folder(CHEKS_FOLDERNAME)
    signatures_rlv_path = relative_path_files_in_folder(SIGNATURES_FOLDERNAME)
    results = []
    for check in checks_rlv_path:
        d = detector(check, app.package['model'])
        try:
            val_criteria = InferenceResult(
                is_crossed=d.detect_bar(),
                amount_letter=d.montant_lettre(),
                amount_number=d.montant_chiffre(),
                # location=' '.join(d.place),
                # date=' '.join(d.date),
                # name_recipient=' '.join(d.name),
                signature_check=d.verif_sign(signatures_rlv_path)
            )
            results.append(val_criteria.dict())
        except:
            logger.info('Problème avec le check : '+check)
    logger.info(f'results: {results}')

    logger.info("Star remove all local cheks and signatures ...")
    remove_all_files_in_folder(CHEKS_FOLDERNAME)
    remove_all_files_in_folder(SIGNATURES_FOLDERNAME)
    logger.info("Done: all local cheks and signatures are deleted !")

    return {
        "error": False,
        "results": results
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


if __name__ == '__main__':
    # server api
    uvicorn.run("main:app", host="0.0.0.0", port=8080,
                reload=True, debug=True, log_config="log.ini"
                )
