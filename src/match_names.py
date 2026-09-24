from rapidfuzz import process

def match_contracts_to_bios(contract_names: list[str], bios: list[dict], threshold: int = 90) -> dict[str, dict]:
    """Match each contract player name to its bios record by fuzzy name match.

    Returns {contract_name: bios_record}. Names that don't clear `threshold`
    are left out - caller should check for gaps rather than trust every name matched.
    """
    bio_names = [b["skaterFullName"] for b in bios]
    matches = {}

    for name in contract_names:
        result = process.extractOne(name, bio_names)
        matched_name, score, index = result

        if score >= threshold:
            matches[name] = bios[index]

    return matches