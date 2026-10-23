"""Customer-language → document-vocabulary glossary used ONLY to expand the lexical (BM25)
query. It maps how customers talk ("send it back", "nobody came to collect", "paisa wapas",
"टूटा") to the words the policies use. It contains vocabulary, never answers or policy facts.
"""
from __future__ import annotations

import re

GLOSSARY: list[tuple[str, str]] = [
    (r"send (it )?back|give (it )?back|wapas|wapis|वापस|लौटा|lautan|lauta|return kar", "return returned window"),
    (r"money back|paisa|paise|पैसा|पैसे|रिफ़ंड|रिफंड|refund", "refund"),
    (r"collect|collection|pick ?up|nobody came|koi nahi aaya", "reverse pickup"),
    (r"cancel|कैंसिल|रद्द|stop (it|the order)", "cancel cancellation"),
    (r"broken|टूट|toot|crack|torn|damag|खराब|kharab|defect", "damaged defective"),
    (r"wrong item|galat|गलत|missing|not in the box", "wrong missing item"),
    (r"not (at )?home|ghar (pe|par) nahi|घर पर नहीं|courier comes", "delivery attempts"),
    (r"\blate\b|der se|देर|delay", "late delivery promised date"),
    (r"guarantee|गारंटी|warrant|वारंटी", "warranty assured"),
    (r"cash on delivery|\bcod\b|नकद", "cash on delivery cod"),
    (r"coupon|promo|voucher|कूपन", "coupon"),
    (r"human|person|insaan|इंसान|real agent|colleague", "human handover"),
    (r"kab tak|कब तक|by which date|by when|how soon|how fast|kitne din|कितने दिन|how long", "timeline days business"),
    (r"shuru|started|initiated|शुरू", "initiated"),
    (r"nahi aaya|नहीं आया|hasn'?t arrived|not received|where is", "track order status"),
    (r"on its way|shipped|dispatch|रवाना|शिप", "shipped"),
    (r"opened|open kiya|खोल|tried it", "unopened seal"),
    (r"doesn'?t fit|size|fit nahi", "return exchange size"),
    (r"plus member|zentromart plus|प्लस", "plus membership"),
    (r"swimsuit|swimwear|briefs|innerwear|lingerie|bra\b", "innerwear swimwear lingerie"),
    (r"lipstick|perfume|parfum|serum|face wash|beauty", "beauty personal care"),
    (r"store near|shop|दुकान|dukaan", "store"),
    (r"never placed|didn'?t place|not mine|unauthori[sz]ed|hack", "did not make account activity"),
]
_COMPILED = [(re.compile(p, re.I), add) for p, add in GLOSSARY]


def expand(query: str) -> str:
    extra = [add for rx, add in _COMPILED if rx.search(query)]
    return query + (" " + " ".join(extra) if extra else "")
