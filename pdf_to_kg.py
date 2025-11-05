import fitz  # PyMuPDF
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import os

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text("text")
    return text

def extract_entities_and_relations(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    triples = []
    for sent in doc.sents:
        entities = [ent.text for ent in sent.ents]
        if len(entities) >= 2:
            # naive heuristic: link first and last entities in a sentence
            triples.append((entities[0], "related_to", entities[-1]))
    return triples

def build_knowledge_graph(triples):
    G = nx.DiGraph()
    for subj, rel, obj in triples:
        G.add_edge(subj, obj, label=rel)
    return G

def visualize_graph(G):
    pos = nx.spring_layout(G, k=0.5)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2500, font_size=8, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.savefig("knowledge_graph.png")


if __name__ == "__main__":
    # CHANGE this path to your PDF’s location
    pdf_path = os.path.expanduser("/home/user/Desktop/Get_Started_With_Smallpdf.pdf")
    text = extract_text_from_pdf(pdf_path)
    print("Extracted text length:", len(text))

    triples = extract_entities_and_relations(text)
    print("Extracted triples:", triples[:10])

    G = build_knowledge_graph(triples)
    visualize_graph(G)
