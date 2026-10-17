"""Label derivation helpers used by build_dataset.py.

IMPORTANT: these are *weak* labels.
  * Gatekeeper label comes from star rating (<=2 stars = complaint, >=4 = not).
    3-star reviews are dropped as ambiguous.
  * Router label comes from keyword rules below. A model trained on them will
    partly just learn the keywords. Treat its metrics as optimistic until you
    evaluate on a hand-labelled gold set (build_dataset.py writes one to label).
"""
import re
from typing import Optional

from .config import ROUTER_LABELS

# Order matters only for tie-breaking (earlier wins when match counts are equal).
CATEGORY_PATTERNS = {
    "order_accuracy": [
        r"wrong (order|item|food|drink|sauce|size)", r"missing", r"forgot", r"left out",
        r"incorrect", r"not what i (ordered|asked)", r"order was (wrong|incorrect|messed)",
        r"messed up (my|the|our) order", r"didn'?t (get|receive|include)", r"no (fries|sauce|napkins|utensils|straw)",
        r"asked for .{0,30} (but|and) (got|received)",
    ],
    "wait_time_service_speed": [
        r"\bwait(ed|ing)?\b", r"\bslow\b", r"took (forever|too long|so long|\d+ min)", r"\bforever\b",
        r"\d+ ?(min|minutes|mins)\b", r"long (time|wait)", r"drive.?thr(u|ough).{0,40}(long|slow|wait)",
        r"\bdelay(ed)?\b",
    ],
    "food_quality": [
        r"\bcold\b", r"\bstale\b", r"\bsoggy\b", r"\bdry\b", r"\bburn(t|ed)\b", r"\braw\b", r"undercooked",
        r"\bgreasy\b", r"tasteless|bland|no flavor|flavorless", r"\bgross\b", r"disgusting", 
        r"\brubbery\b", r"\bnasty\b", r"food poisoning|got sick|\bsick\b|stomach", r"\btaste[sd]? (bad|awful|terrible|like)",
        r"\bmushy\b", r"overcooked",
    ],
    "cleanliness_hygiene": [
        r"\bdirty\b", r"\bfilthy\b", r"\bunclean\b", r"\bsticky\b", r"\bmess(y)?\b", r"\bsmell(s|ed)?\b",
        r"\broach(es)?\b", r"\bbugs?\b", r"\bhair\b", r"\bbathroom|restroom\b", r"\btrash\b",
        r"\bgloves?\b", r"unsanitary|hygiene|sanitary", r"\btables? (were|was|are|is) ",
    ],
    "staff_attitude": [
        r"\brude\b", r"\battitude\b", r"\bunfriendly\b", r"\bignored\b", r"\bunprofessional\b", 
        r"\bdisrespect", r"\byell(ed|ing)\b", r"\bcurs(e|ed|ing)\b", r"\bargu(ed|ing|ment)\b", r"\bcustomer service\b",
        r"\bhung up\b", r"\bcould ?n'?t care\b", r"\bsnippy\b", r"\bsarcastic\b", r"\bemployees? (were|was|are)\b",
    ],
    "price_value": [
        r"\bexpensive\b", r"\boverpriced\b", r"\bprice[sd]?\b", r"\brip.?off\b", r"\bovercharg",
        r"\bnot worth\b", r"\bwaste of money\b", r"\bpaid \$?\d+", r"\bportion[s]?\b", r"\btiny\b", r"\bsmall (portion|size|amount)",
    ],
}

_COMPILED = {
    cat: [re.compile(p, re.IGNORECASE) for p in pats] for cat, pats in CATEGORY_PATTERNS.items()
}


def sentiment_from_stars(stars: float) -> Optional[int]:
    """1 = complaint (<=2 stars), 0 = not complaint (>=4 stars), None = dropped (3 stars)."""
    if stars <= 2:
        return 1
    if stars >= 4:
        return 0
    return None


def category_scores(text: str) -> dict:
    return {cat: sum(1 for rx in rxs if rx.search(text)) for cat, rxs in _COMPILED.items()}


def weak_category(text: str) -> str:
    """Pick the category with the most keyword hits; 'other' if no hits."""
    scores = category_scores(text)
    best = max(scores.values())
    if best == 0:
        return "other"
    for cat in ROUTER_LABELS:  # deterministic tie-break by ROUTER_LABELS order
        if scores.get(cat) == best:
            return cat
    return "other"


_BRAND_FIXES = [
    (re.compile(r"^dairy queen.*", re.I), "Dairy Queen"),
    (re.compile(r"^mcdonald'?s.*", re.I), "McDonald's"),
    (re.compile(r"^qdoba.*", re.I), "Qdoba"),
    (re.compile(r"^chipotle.*", re.I), "Chipotle"),
    (re.compile(r"^popeyes.*", re.I), "Popeyes"),
    (re.compile(r"^steak .?n shake.*", re.I), "Steak n Shake"),
    (re.compile(r"^wendy'?s.*", re.I), "Wendy's"),
    (re.compile(r"^arby'?s.*", re.I), "Arby's"),
    (re.compile(r"^domino'?s.*", re.I), "Domino's"),
    (re.compile(r"^papa john'?s.*", re.I), "Papa John's"),
    (re.compile(r"^jimmy john'?s.*", re.I), "Jimmy John's"),
    (re.compile(r"^jersey mike'?s.*", re.I), "Jersey Mike's"),
]


def normalize_brand(name: str) -> str:
    name = (name or "").strip()
    for rx, canon in _BRAND_FIXES:
        if rx.match(name):
            return canon
    return name
