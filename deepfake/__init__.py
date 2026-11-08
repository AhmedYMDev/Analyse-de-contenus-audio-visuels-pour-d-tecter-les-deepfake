"""Boîte à outils partagée pour la détection de deepfakes (Celeb-DF v2).

Ce package est importé par l'application Flask et par les scripts du dépôt, afin
que le prétraitement, le chargement des modèles et les métriques ne soient
définis qu'à un seul endroit.
"""

from .config import ModelCard, load_model_card

__all__ = ["ModelCard", "load_model_card"]
