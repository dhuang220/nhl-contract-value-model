def infer_contract_type(age_at_signing: int, debut_year: int, offseason_year: int) -> str:
     """Approximate UFA/RFA status: UFA at 27+ or ~7 seasons since NHL debut, else RFA."""
     years_since_debut = offseason_year - debut_year
     if years_since_debut >= 7 or age_at_signing >= 27:
          return 'UFA'
     return 'RFA'