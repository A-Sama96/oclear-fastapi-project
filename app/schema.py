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
    Input zip File
    """
    check_path: List[str] = Field(..., example=['URL1', 'URL2'],
                                  title="List de lien de chèque provenant de AWS S3")
    signature_path: List[str] = Field(..., example=['URL1', 'URL2'],
                                      title='List de lien de signature provenant de AWS S3')


class InferenceResult(BaseModel):
    """
    Inference result from the model
    """
    file_name: str = Field(..., example='20190316-SN099SN0991000002190017823951Recto.jpg',
                               title='Nom du fichier image')
    is_crossed: bool = Field(..., example=False,
                             title='Vérifie si le chèque est barré')
    amount_letter: str = Field(..., example='deux cent cinquante milles',
                               title='Montant en lettre')
    amount_number: str = Field(..., example='250.000',
                               title='Montant en chiffre')
    location: str = Field(..., example='Dakar',
                               title='Lieu mentionné sur le chèque')
    date: str = Field(..., example='08/12/2021',
                               title='Date sur le check')
    amounts_compliance: bool = Field(..., example=False,
                              title='Vérifie si les montants en chiffre et en lettre sont conforme')
    # name_recipient: str

    endorsable: bool = Field(..., example=True,
                              title='Spécifie si le chèque est endossable')
    signature_check: List[float] = Field(..., example='[80.9]',
                                         title='Taux de correspondace de la signature')


class InferenceResponse(BaseModel):
    """
    Output response for model inference
    """
    error: bool = Field(..., example=False, title='Spécifie si il y a erreur')
    results: List[InferenceResult] = ...
    errorReadChecksName: list = ...


class ErrorResponse(BaseModel):
    """
    Error response for the API
    """
    error: bool = Field(..., example=True, title='Spécifie si il y a erreur')
    message: str = Field(..., example='', title='Message d\'erreur')
    traceback: str = Field(
        None, example='', title='Traceback détaillé de l\'erreur')
