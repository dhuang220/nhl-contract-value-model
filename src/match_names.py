from rapidfuzz import fuzz, process


def match_contracts_to_bios(contract_names: list[str], bios: list[dict], threshold: int = 90) -> dict[str, dict]:
    """Match each contract player name to its bios record by fuzzy name match.

    Returns {contract_name: bios_record}. Name-only - can't disambiguate same-name
    players; use match_rows_to_bios when contract positions are available.
    """
    bio_names = [b["skaterFullName"] for b in bios]
    matches = {}
    for name in contract_names:
        matched_name, score, index = process.extractOne(name, bio_names)
        if score >= threshold:
            matches[name] = bios[index]
    return matches


def _contract_is_defense(position) -> bool:
    """True if every listed position token is a defense slot (D/LD/RD)."""
    tokens = [t.strip() for t in str(position).split(",")]
    return bool(tokens) and all(t in {"D", "LD", "RD"} for t in tokens)


def match_rows_to_bios(contracts, bios: list[dict], threshold: int = 88) -> list[dict | None]:
    """Match each contract ROW to a bios record, disambiguating same-name players
    by position (D vs forward). Fixes collisions like the two Sebastian Ahos
    (Carolina forward vs Pittsburgh defenseman) or the two Elias Petterssons.

    Returns a list aligned to `contracts` rows (None where no name clears threshold).
    """
    bio_names = [b["skaterFullName"] for b in bios]
    out = []
    for _, c in contracts.iterrows():
        cands = process.extract(c["player_name"], bio_names, scorer=fuzz.WRatio,
                                limit=6, score_cutoff=threshold)
        if not cands:
            out.append(None)
            continue
        want_d = _contract_is_defense(c["position"])
        pos_match = [idx for _, _, idx in cands if (bios[idx]["positionCode"] == "D") == want_d]
        chosen = pos_match[0] if pos_match else cands[0][2]  # best position-consistent, else best name
        out.append(bios[chosen])
    return out