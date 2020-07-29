from deidentify.base import Document, Annotation
from deidentify.taggers import FlairTagger
from deidentify.tokenizer import TokenizerFactory
import spacy

# Create some text
text = (
    "Dies ist ein Textstück mit dem Namen Jan Jansen. Der Patient J. Jansen (e: "
    "j.jnsen@email.com, t: 06-12345678) ist 64 Jahre alt und lebt in Utrecht. Er war am 10."
    "Oktober von Arzt Peter de Visser aus der UMCU-Klinik entlassen."
)

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

# print(annotated_doc)
# print(annotated_doc.annotations)

# Spacy NER extraction
ners = nlp(text)

filtered_annotations = []

for ent in ners.ents:
    filtered_annotations.append({"text": ent.text, "start": ent.start_char, "end": ent.end_char,
                            "tag": ent.label_})

for ann in annotated_doc.annotations:
    # don't add the entity if it overlaps with SpaCy's - SpaCy makes fewer mistakes
    if True in [ent.start_char <= ann.end <= ent.end_char for ent in ners.ents] or \
        True in [ann.start <= ent.end_char <= ann.end for ent in ners.ents]:
        continue
    filtered_annotations.append({"text": ann.text, "start": ann.start, "end": ann.start, 
                                "tag": ann.tag})
filtered_annotations.sort(key=lambda x: x["start"])

for f in filtered_annotations:
    print(f)

# from deidentify.util import mask_annotations

# masked_doc = mask_annotations(annotated_doc)
# print(masked_doc.text)



