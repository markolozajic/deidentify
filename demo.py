from deidentify.base import Document
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
annotated_docs = tagger.annotate(documents)


from pprint import pprint

first_doc = annotated_docs[0]
pprint(first_doc.annotations)


from deidentify.util import mask_annotations

masked_doc = mask_annotations(first_doc)
print(masked_doc.text)

ners = nlp(text)

for ent in ners.ents:
	print(ent.text, ent.start_char, ent.end_char, ent.label_)
