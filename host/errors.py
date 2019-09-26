"""
This module contains custom error message for HOS picking algorithm

REFERENCE:
https://stackoverflow.com/questions/1319615/proper-way-to-declare-custom-exceptions-in-modern-python
"""


class Error(Exception):
    """ Base class for other exceptions """
    pass


class BadInstance(Error):
    """ Raised when important instance checks are not respected """
    pass


class BadKeyValue(Error):
    """ Raised when important instance checks are not respected """
    pass


class BadParameterValue(Error):
    """ Raised when a wrong parameter is given to a function/class """
    pass


class MissingVariable(Error):
    """ Raised when a parameter/value is missing """
    pass


class MissingKey(Error):
    """ Raised when a dict parameter/value is missing """
    pass


class MissingAttribute(Error):
    """ Raised when an attribute is not found """
    pass


class PickNotFound(Error):
    """ Raised when a pick is not found or is not possible to evaluate
        the trace
    """
    pass


class Miscellanea(Error):
    """ Raised when the developer is too lazy to think at smth else """
    pass
