import hashlib
import re
import unicodedata

def normalize_name(name):
    """Lowercase, strip accents, collapse non-alphanumerics to single spaces."""
    if name is None:
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return s

def make_match_id(date, home, away):
    """Deterministic 16-char id from date + normalized team names (order matters)."""
    key = f"{str(date)[:10]}|{normalize_name(home)}|{normalize_name(away)}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]

class IDMap:
    """Resolve a (source, source_id) or a player/team name to a canonical id."""
    def __init__(self):
        self._by_source = {}
        self._by_name = {}

    def add(self, source, source_id, canonical_id, name=None):
        self._by_source[(source, str(source_id))] = canonical_id
        if name:
            self._by_name[normalize_name(name)] = canonical_id

    def resolve(self, source, source_id):
        return self._by_source.get((source, str(source_id)))

    def resolve_name(self, name):
        return self._by_name.get(normalize_name(name))
