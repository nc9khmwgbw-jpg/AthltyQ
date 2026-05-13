from abc import ABC, abstractmethod

class BaseScraper(ABC):
    """
    Interface de base pour tous les scrapers du projet AthlytIQ.
    Définit le contrat que chaque source (SofaScore, Transfermarkt, etc.) doit respecter.
    """
    
    @abstractmethod
    def scrape(self, target_id, **kwargs):
        """
        Méthode principale pour lancer le scrapping.
        target_id: ID du joueur, de l'équipe ou de la ligue.
        """
        pass

    @abstractmethod
    def save(self, data, path):
        """
        Méthode pour sauvegarder les données extraites.
        """
        pass
