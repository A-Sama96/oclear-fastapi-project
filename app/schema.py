#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

"""
class RequestValidation(BaseModel):
    check_path: str
    signature_path: str


class ValidationCriteria(BaseModel):
    is_crossed: bool
    amount_letter: str
    # location: str
    # date: str
    # name_recipient: str
    signature_check: List[float]
"""


class InferenceInput(BaseModel):
    """
    Input values for model inference
    """
    check_path: List(str) = Field(..., example="['URL1', 'URL2']",
                                  title="List de lien de chèque provenant de AWS S3")
    signature_path: str = Field(..., example="['URL1', 'URL2']",
                                title='List de lien de signature provenant de AWS S3')


class InferenceResult(BaseModel):
    """
    Inference result from the model
    """
    is_crossed: bool = Field(..., example=False,
                             title='Vérifie si le chèque est barré')
    amount_letter: str = Field(..., example='deux cent cinquante milles',
                               title='Montant en lettre')
    amount_number: str = Field(..., example='250.000',
                               title='Montant en chiffre')
    # location: str
    # date: str
    # name_recipient: str
    signature_check: List[float] = Field(..., example='[80.9]',
                                         title='Taux de correspondace de la signature')


class InferenceResponse(BaseModel):
    """
    Output response for model inference
    """
    error: bool = Field(..., example=False, title='Spécifie si il y a erreur')
    results: List(InferenceResult) = ...


class ErrorResponse(BaseModel):
    """
    Error response for the API
    """
    error: bool = Field(..., example=True, title='Spécifie si il y a erreur')
    message: str = Field(..., example='', title='Message d\'erreur')
    traceback: str = Field(
        None, example='', title='Traceback détaillé de l\'erreur')
