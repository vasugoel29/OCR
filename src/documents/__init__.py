"""Documents module."""

from .base import BaseDocumentProcessor
from .invoice import InvoiceProcessor
from .id_document import IDDocumentProcessor

__all__ = ['BaseDocumentProcessor', 'InvoiceProcessor', 'IDDocumentProcessor']
