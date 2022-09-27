#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


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


class InferenceInput(BaseModel):
    """
    Input values for model inference
    """
    sepal_length: float = Field(..., example=3.1,
                                gt=0, title='sepal length (cm)')
    sepal_width: float = Field(..., example=3.5, gt=0,
                               title='sepal width (cm)')
    petal_length: float = Field(..., example=3.4,
                                gt=0, title='petal length (cm)')
    petal_width: float = Field(..., example=3.0, gt=0,
                               title='petal width (cm)')
    check_path: List(str) = Field(..., example='versicolor',
                                  title='Predicted class with highest probablity')
    signature_path: str


class InferenceResult(BaseModel):
    """
    Inference result from the model
    """
    is_crossed: bool = Field(..., example=False,
                             title='Whether there is error')
    amount_letter: str
    amount_number: str
    # location: str
    # date: str
    # name_recipient: str
    signature_check: List[float]


class InferenceResponse(BaseModel):
    """
    Output response for model inference
    """
    error: bool = Field(..., example=False, title='Whether there is error')
    results: InferenceResult = ...


class ErrorResponse(BaseModel):
    """
    Error response for the API
    """
    error: bool = Field(..., example=True, title='Whether there is error')
    message: str = Field(..., example='', title='Error message')
    traceback: str = Field(
        None, example='', title='Detailed traceback of the error')
