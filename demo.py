from deidentify.base import Document, Annotation
from deidentify.taggers import FlairTagger
from deidentify.tokenizer import TokenizerFactory
from deidentify.util import mask_annotations

import spacy
import sys

def anonymize(input_file):

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Wrap text in document
    documents = [
        Document(name='doc_01', text=text)
    ]

    # Select downloaded model
    model = 'models/model_bilstmcrf_ons_fast-v0.1.0/final-model.pt'

    nlp = spacy.load('de_core_news_sm')

    # Instantiate tokenizer
    tokenizer = TokenizerFactory().tokenizer(corpus='germeval', disable=("tagger", "ner"), model=nlp)

    # Load tagger with a downloaded model file and tokenizer
    tagger = FlairTagger(model=model, tokenizer=tokenizer, verbose=False)

    # Annotate your documents
    annotated_doc = tagger.annotate(documents)[0]

    # Spacy NER extraction
    ners = nlp(text)

    filtered_annotations = []

    # Dict for storing SpaCy and deidentify tag correspondences
    tag_dict = {"PER": "Name", "LOC": "Address", "ORG": "Organization_Company", "MISC": "Other"}

    # Add all SpaCy-detected NEs to list
    for ent in ners.ents:
        filtered_annotations.append({"text": ent.text, "start": ent.start_char, "end": ent.end_char,
                                "tag": tag_dict[ent.label_]})

    for ann in annotated_doc.annotations:
        # discard names; they have a high likelihood of false positives since
        # nouns are capitalized in German, unlike in Dutch
        if ann.tag == "Name":
            continue
        # don't add the entity if it overlaps with SpaCy's - SpaCy makes fewer mistakes
        if True in [ent.start_char <= ann.end <= ent.end_char for ent in ners.ents] or \
            True in [ann.start <= ent.end_char <= ann.end for ent in ners.ents]:
            continue
        filtered_annotations.append({"text": ann.text, "start": ann.start, "end": ann.end, 
                                    "tag": ann.tag})

    filtered_annotations.sort(key=lambda x: x["start"])

    masked_output = mask_annotations(annotated_doc.text, filtered_annotations)
    print(masked_output)


if __name__ == "__main__":
    input_file = sys.argv[1]
    anonymize(input_file)

