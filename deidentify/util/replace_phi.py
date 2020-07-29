
def _uppercase_formatter(annotation):
    return '[{}]'.format(annotation["tag"].upper())


def mask_annotations(doc_text, annotations, replacement_formatter = _uppercase_formatter):
    """Utility function to replace sensitive PHI spans with a placeholder."""
    # Amount of characters by which start point of annotation is adjusted
    # Positive shift if replacement is longer than original annotation
    # Negative shift if replacement is shorter
    shift = 0

    original_text_pointer = 0
    text_rewritten = ''
    annotations_rewritten = []

    for annotation in annotations:
        replacement = replacement_formatter(annotation)
        part = doc_text[original_text_pointer:annotation["start"]]

        start = annotation["start"] + shift
        end = start + len(replacement)
        shift += len(replacement) - len(annotation["text"])

        text_rewritten += part + replacement
        original_text_pointer = annotation["end"]

    text_rewritten += doc_text[original_text_pointer:]
    return text_rewritten
