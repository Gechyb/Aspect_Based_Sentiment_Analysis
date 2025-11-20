import xml.etree.ElementTree as ET
import spacy
import json
from preprocess import normalize_text, to_bio_sentiment

nlp = spacy.load("en_core_web_sm")


def convert_xml_to_jsonl(xml_path, jsonl_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    with open(jsonl_path, "w", encoding="utf-8") as out_f:
        for sent in root.findall("sentence"):
            text_el = sent.find("text")
            if text_el is None:
                continue

            text = normalize_text(text_el.text)
            doc = nlp(text)
            tokens = [t.text for t in doc]

            # initialize all tokens as non-aspect
            annotated_tokens = [{"token": t, "aspect": False} for t in tokens]

            # read <aspectTerms>
            aspect_terms_el = sent.find("aspectTerms")
            if aspect_terms_el is not None:
                for term_el in aspect_terms_el.findall("aspectTerm"):
                    term = term_el.attrib["term"]
                    polarity = term_el.attrib["polarity"].upper()
                    polarity = {
                        "positive": "POS",
                        "negative": "NEG",
                        "neutral": "NEU",
                    }.get(polarity.lower(), "NEU")

                    # tokenize the aspect term
                    term_tokens = [t.text for t in nlp(term)]
                    L = len(term_tokens)

                    # align with sentence tokens
                    for i in range(len(tokens) - L + 1):
                        if tokens[i : i + L] == term_tokens:
                            annotated_tokens[i]["aspect"] = True
                            annotated_tokens[i]["sent"] = polarity
                            annotated_tokens[i]["begin"] = True
                            for j in range(1, L):
                                annotated_tokens[i + j]["aspect"] = True
                                annotated_tokens[i + j]["sent"] = polarity
                                annotated_tokens[i + j]["begin"] = False

            # convert to BIO-Sent tags
            labels = to_bio_sentiment(annotated_tokens)

            example = {"tokens": tokens, "labels": labels}
            out_f.write(json.dumps(example) + "\n")

    print("Saved:", jsonl_path)
