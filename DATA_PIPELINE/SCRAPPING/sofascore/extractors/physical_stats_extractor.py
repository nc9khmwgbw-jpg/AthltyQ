class SofaScorePhysicalStatsExtractor:
    """Extracteur spécialisé dans les données de performance (distanceRun, sprints)."""
    
    @staticmethod
    def extract(stats_json):
        """Extrait les données physiques depuis le JSON de statistiques détaillées."""
        # Note: SofaScore utilise souvent des 'groups' pour ces stats
        groups = stats_json.get('statistics', {}).get('groups', [])
        physical_data = {
            'distanceRun': 0,
            'sprints': 0,
            'kpi_work_rate': 0
        }

        for group in groups:
            # On cherche partout, car le nom du groupe peut varier
            for item in group.get('statisticsItems', []):
                if item.get('key') == 'distanceRun':
                    physical_data['distanceRun'] = item.get('value', 0)
                elif item.get('key') == 'sprints':
                    physical_data['sprints'] = item.get('value', 0)
        
        # Calcul du KPI maison (Work Rate)
        if physical_data['distanceRun'] > 0:
            physical_data['kpi_work_rate'] = (physical_data['distanceRun'] / 1000) * (1 + (physical_data['sprints'] / 20))
            
        return physical_data
