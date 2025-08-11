def clean_text(s: str) -> str:
    # fix apostrophes and accented e's (like è)
    return s.replace("\u2018", "'").replace("\u2019", "'").replace("\u00e9", 'e')