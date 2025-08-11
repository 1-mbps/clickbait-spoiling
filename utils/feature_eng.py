import torch

def search_for_phrase(text: list[str], phrase: str) -> int:
    if len(text) <= 1:
        return 0
    phrase = phrase.split()
    for i in range(len(text)-len(phrase)+1):
        if all([text[i+j] == phrase[j] for j in range(len(phrase))]):
            return 1
    return 0

def check_patterns(title: list[str], title_str: str, paragraphs: list[str]) -> tuple[list[int], torch.Tensor]:
    """
    Produce a vector of hand-crafted numeric and categorical features extracted from the clickbait dataset
    """

    # prepare output
    num = [0 for _ in range(2)]
    cat = [0 for _ in range(4)]

    # combine paragraphs into a single string, then split into words
    full_txt = ''.join(paragraphs).split()

    # number of paragraphs
    num[0] = len(paragraphs)
    
    # number of words
    num[1] = len(full_txt)

    # 'multi' and 'passage' have far more titles beginning with "how" or "why"
    cat[0] = int(title[0] == "how" or title[0] == "why")

    # search for the following phrases
    c1 = search_for_phrase(title, "guess who") or search_for_phrase(title, "believe who")
    c2 = search_for_phrase(title, "guess where") or search_for_phrase(title, "believe where")
    c3 = search_for_phrase(title, "guess how much") or search_for_phrase(title, "believe how much")
    cat[1] = int(c1 or c2 or c3)

    # check if first word in title is a number - indicates multipart spoiler
    first_word_lower = title[0].lower()
    cat[2] = int(first_word_lower.isdigit())

    # check if there's a paragraph beginning with #1, 1., or 1), and #2, 2., or 2) - indicates multipart spoiler
    cat[3] = (
        any([par.strip().startswith("#1") or par.strip().startswith("1.") or par.strip().startswith("1)") for par in paragraphs]) and
        any([par.strip().startswith("#2") or par.strip().startswith("2.") or par.strip().startswith("2)") for par in paragraphs])
    )

    return num, torch.tensor(cat, dtype=torch.float32)