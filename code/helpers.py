# =============================================================================
# helper functions
# =============================================================================

import spacy
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

# fix spacy tokenizer so does not split on hyphens

# load large 
nlp = spacy.load("en_core_web_lg")

# modify tokenizer infix patterns
infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            # EDIT: commented out regex that splits on hyphens between letters:
            # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
)

infix_re = compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_re.finditer


def preprocess(texts, to_lower=True, rm_stops=False, rm_punct=True, lemmatize=False):
    # lowercase
    if (to_lower):
        texts = [elem.lower() for elem in texts]

    # convert list of strings to list of docs
    if (rm_stops):
        for num, item in enumerate(texts):
            doc = nlp(item)
            doc_list = []
            for token in doc:
                if (token.is_stop == False):
                    doc_list.append(token.text)
            texts[num] = ' '.join(doc_list)

    # remove punctuation
    if (rm_punct):
        for num, item in enumerate(texts):
            doc = nlp(item)
            doc_list = []
            for token in doc:
                if not token.is_punct:
                    doc_list.append(token.text)
            texts[num] = ' '.join(doc_list)

    # lemmatize
    if (lemmatize):
        for num, item in enumerate(texts):
            doc = nlp(item)
            doc_list = []
            for token in doc:
                doc_list.append(token.lemma_)
            texts[num] = ' '.join(doc_list)

    # split into list of lists
    texts = [item.split() for item in texts]

    return texts


# create and return sbmtm object
def run_sbmtm(text_object, seed=1234):
    # we create an instance of the sbmtm-class
    model = sbmtm()
    # we have to create the word-document network from the corpus
    model.make_graph(text_object)
    # fit the model
    gt.seed_rng(seed)  # seed for graph-tool's random number generator --> same results
    model.fit()
    return model
