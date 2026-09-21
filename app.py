"""
========================================================================================
 Information Retrieval / Graph Theory Virtual Lab
 Experiment No. 26-30 :  IDENTIFY GRAPH ENTITIES
 Style   : IIT Kharagpur Virtual Labs
           (Introduction -> Pre-Test -> Aim -> Theory -> Procedure -> Simulation ->
            Post-Test -> References -> Feedback), single-file Streamlit app.

 What's new in this version (as suggested by faculty)
 ----------------------------------------------------
 1. Corpus upload      : upload one or many .txt / .md / .csv / .pdf / .docx files.
 2. Hover definitions  : every technical term (Precision@k, PageRank, ...) shows a
                         tooltip with its definition AND its role in this experiment.
 3. Intermediate steps : sentence splitting, candidate extraction trace, co-occurrence
                         matrix, edge-by-edge graph construction, PageRank iterations
                         (worked example), Precision/Recall/AP tables - all shown.
 4. Rich theory        : motivation, diagrams, applications and e-book references.

 Graph is built with NetworkX + Matplotlib (few-node, in-memory). No Neo4j / external DB.

 Install : pip install streamlit networkx pandas numpy matplotlib pypdf python-docx
 Run     : streamlit run graph_entities_vlab.py
========================================================================================
"""

import io
import re
import html
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------------------------------------------------------------
# PAGE CONFIG + STYLE
# ----------------------------------------------------------------------------------
st.set_page_config(page_title="Identify Graph Entities | Virtual Lab",
                   page_icon="🕸️", layout="wide")

PRIMARY = "#8B0000"   # IIT Kharagpur maroon
ACCENT = "#003366"

CSS = """
<style>
.top-banner {background: linear-gradient(90deg, __P__ 0%, __A__ 100%);
    padding: 18px 25px; border-radius: 6px; color: white; margin-bottom: 12px;}
.top-banner h1 {margin: 0; font-size: 26px; color: white;}
.top-banner p  {margin: 2px 0 0 0; font-size: 14px; opacity: 0.92;}
.section-card {background-color: #fafafa; border-left: 5px solid __P__;
    padding: 14px 18px; border-radius: 4px; margin-bottom: 14px; color:#222;}
.hook {background:#fff8e6; border-left:5px solid #f0a500; padding:14px 18px;
    border-radius:4px; margin-bottom:14px; color:#333; font-size:16px;}
.step-badge {display:inline-block; background:__P__; color:#fff; border-radius:12px;
    padding:2px 12px; font-size:13px; margin-right:6px;}
.tip {border-bottom: 1.5px dotted __P__; color: __P__; cursor: help;
    position: relative; font-weight: 600;}
.tip:hover::after {content: attr(data-tip); position: absolute; left: 0; top: 1.7em;
    z-index: 99999; width: 330px; background: #222; color: #fff; padding: 9px 11px;
    border-radius: 6px; font-size: 12.5px; font-weight: 400; line-height: 1.4;
    box-shadow: 0 4px 14px rgba(0,0,0,.35); white-space: normal;}
.ent {padding:2px 6px; border-radius:4px; color:#fff; font-weight:600; margin:0 1px;}
.ent small {font-size:9px; opacity:.9; margin-left:4px; text-transform:uppercase;}
.rankbox {display:inline-block; width:38px; height:38px; line-height:38px; text-align:center;
    border-radius:5px; margin:2px; color:#fff; font-weight:700;}
.small-note {font-size:12.5px; color:#666;}
</style>
""".replace("__P__", PRIMARY).replace("__A__", ACCENT)
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("""
<div class="top-banner">
  <h1>🕸️ Virtual Lab — Information Retrieval &amp; Graph Theory</h1>
  <p>Discipline: Computer Science &amp; Engineering &nbsp;|&nbsp;
     Experiment 26–30 : <b>Identify Graph Entities</b> &nbsp;|&nbsp;
     Modelled on the IIT Kharagpur Virtual Labs format</p>
</div>
""", unsafe_allow_html=True)


def md(text):
    st.markdown(text, unsafe_allow_html=True)


# ----------------------------------------------------------------------------------
# GLOSSARY  ->  hover definitions   (term : (definition, relevance in this lab))
# ----------------------------------------------------------------------------------
GLOSSARY = {
    "NER": ("Named Entity Recognition - locating and classifying real-world objects "
            "(people, organisations, places...) in text.",
            "Step 1 of the simulation: we run a rule-based NER to get graph nodes."),
    "Entity": ("A real-world object or concept mentioned in text, e.g. 'IIT Kharagpur'.",
               "Every entity becomes one node of the graph."),
    "Corpus": ("A collection of text documents used as the data set.",
               "You can upload several files; all of them are analysed together."),
    "Gazetteer": ("A look-up list of known names (e.g. place names).",
                  "We use a small gazetteer to decide whether a phrase is a LOCATION."),
    "Co-occurrence": ("Two items appearing together in the same context (here: sentence).",
                      "Co-occurrence in a sentence creates an edge between two entities."),
    "Node": ("A vertex of a graph.", "One node = one entity."),
    "Edge": ("A link between two nodes of a graph.",
             "An edge means 'these two entities occur in the same sentence'."),
    "Edge weight": ("A number attached to an edge that expresses its strength.",
                    "Weight = number of sentences in which both entities co-occur."),
    "Degree": ("Number of edges attached to a node.",
               "A high-degree entity is connected to many other entities."),
    "Co-occurrence matrix": ("A square table whose cell (i,j) counts how often entity i "
                             "and entity j co-occur.",
                             "It is the adjacency (weight) matrix of our graph."),
    "Knowledge Graph": ("A graph of entities (nodes) and relations (edges) used to store "
                        "knowledge.", "Our small entity graph is a mini knowledge graph."),
    "Frequency": ("How many sentences in the corpus contain the entity (term frequency).",
                  "Baseline ranking: the more frequent, the higher the rank."),
    "Document frequency": ("Number of documents that contain the entity.",
                           "Shows how widespread an entity is across your corpus."),
    "PageRank": ("Algorithm that scores a node by how likely a random surfer is to be "
                 "there; important neighbours give more score.",
                 "Our probabilistic ranking of entities."),
    "Random surfer": ("Imaginary walker who keeps following links at random.",
                      "PageRank score = long-run fraction of time the surfer spends on a node."),
    "Damping factor": ("Probability d that the surfer follows an edge; with probability "
                       "1-d it jumps to a random node. Usually 0.85.",
                       "Controlled by the slider in Step 3."),
    "Transition matrix": ("Matrix M where M[j,i] = probability to move from node i to j.",
                          "Built from edge weights: w(i,j) / sum of i's weights."),
    "Iteration": ("One repetition of the PageRank update formula.",
                  "Scores are refined iteration by iteration until they stop changing."),
    "Convergence": ("State when further iterations no longer change the scores.",
                    "We stop when the total change (L1 delta) is below 1e-6."),
    "L1 delta": ("Sum of absolute differences between successive score vectors.",
                 "Used as the stopping criterion for PageRank."),
    "Dangling node": ("Node with no outgoing edges.",
                      "The surfer teleports uniformly from such a node."),
    "Ground truth": ("The set of items a human judges to be truly relevant.",
                     "You choose it in Step 4 to evaluate both rankings."),
    "Precision@k": ("Fraction of the top-k results that are relevant = hits_in_top_k / k.",
                    "Tells how clean the top of each ranking is."),
    "Recall@k": ("Fraction of all relevant items found in top-k = hits_in_top_k / R.",
                 "Tells how much of the ground truth is captured by the top-k."),
    "Average Precision": ("Mean of Precision@rank taken at every rank where a relevant item "
                          "appears (divided by R). Rewards ranking relevant items early.",
                          "Single number used to compare PageRank vs Frequency ranking."),
    "Sentence segmentation": ("Splitting text into sentences.",
                              "Sentences are the 'context windows' for co-occurrence."),
}


def T(term, label=None):
    """Return an HTML span that shows definition + relevance on hover."""
    d, r = GLOSSARY[term]
    tip = html.escape(f"{term}: {d}  \u25B8 In this lab: {r}", quote=True)
    return f'<span class="tip" data-tip="{tip}">{label or term}</span>'


def plain_help(term):
    """Plain text help string for Streamlit widget / column tooltips."""
    d, r = GLOSSARY[term]
    return f"{term}: {d}\n\nIn this lab: {r}"


# ----------------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------------
st.sidebar.title("📘 Experiment Navigator")
PAGES = ["Introduction", "Pre-Test", "Aim & Objectives", "Theory", "Procedure",
         "Simulation", "Post-Test", "References", "Feedback"]
page = st.sidebar.radio("Go to", PAGES)
st.sidebar.markdown("---")
st.sidebar.info(
    "Extract entities from a **corpus**, connect them into a small "
    "**co-occurrence graph**, rank them with **PageRank**, and compare against a "
    "**frequency** baseline - with every intermediate step visible.")
with st.sidebar.expander("📖 Glossary (hover terms in the lab too)"):
    for k, (d, r) in GLOSSARY.items():
        st.markdown(f"**{k}** - {d}")

COLOR_MAP = {"PERSON": "#1f77b4", "ORGANIZATION": "#ff7f0e",
             "LOCATION": "#2ca02c", "MISC/CONCEPT": "#9467bd"}


def ent_span(text, label):
    return (f'<span class="ent" style="background:{COLOR_MAP[label]}">{text}'
            f'<small>{label}</small></span>')


# ----------------------------------------------------------------------------------
# NLP HELPERS  (rule-based NER)
# ----------------------------------------------------------------------------------
ORG_KW = {"University", "Institute", "Corporation", "Corp", "Company", "Inc", "Ltd", "Bank",
          "Ministry", "Agency", "Organization", "Organisation", "Foundation", "Association",
          "Group", "Council", "Commission", "School", "College", "Nations", "Department",
          "Laboratory", "Academy", "Society"}
LOC_GAZ = {"India", "USA", "America", "United States", "London", "Paris", "Delhi",
           "New Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore", "Kharagpur", "New York",
           "California", "Tokyo", "Beijing", "Germany", "France", "China", "Japan", "Russia",
           "Bengal", "Asia", "Europe", "Africa", "Silicon Valley", "Geneva", "Washington",
           "Cambridge", "Oxford", "Santiniketan", "Boston", "Berlin", "Pune", "Hyderabad"}
TITLES = {"Mr", "Mrs", "Ms", "Dr", "Prof", "Sir", "Shri", "Smt"}
STOP_START = {"The", "This", "That", "These", "Those", "It", "In", "On", "At", "A", "An",
              "He", "She", "They", "We", "I", "You", "His", "Her", "Their", "Its", "As",
              "For", "But", "And", "After", "Before", "During", "When", "While", "If", "Both",
              "Later", "Also", "However", "Then", "There", "Here", "With", "From", "By", "Of"}

ABBR = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|St|Sr|Jr|Inc|Ltd|Corp|vs|etc)\.")
ENT_RE = re.compile(r"\b[A-Z][A-Za-z\-]*\.?(?:\s+(?:(?:of|for)\s+)?[A-Z][A-Za-z\-]*\.?)*")


def split_sentences(text):
    """Sentence segmentation that does not break on 'Dr.', 'Mr.' etc."""
    text = text.replace("\r", "")
    protected = ABBR.sub(lambda m: m.group(1) + "\u00a7", text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", protected)
    return [p.replace("\u00a7", ".").strip() for p in parts if p.strip()]


def classify(tokens, title):
    """Return (type, rule that fired)."""
    ent = " ".join(tokens)
    for t in tokens:
        if t in ORG_KW:
            return "ORGANIZATION", f"contains organisation keyword '{t}'"
    if title:
        return "PERSON", f"preceded by honorific '{title}.'"
    if ent in LOC_GAZ or any(t in LOC_GAZ for t in tokens):
        return "LOCATION", "found in place-name gazetteer"
    if len(tokens) == 2:
        return "PERSON", "two consecutive capitalised words (First Last pattern)"
    return "MISC/CONCEPT", "no rule matched -> domain concept"


def extract_candidates(sentence):
    """Return list of dicts documenting every extraction decision."""
    out = []
    for m in ENT_RE.finditer(sentence):
        raw = m.group(0).strip()
        tokens = [t.rstrip(".") for t in raw.split()]
        first_raw = tokens[0] if tokens else ""
        while tokens and tokens[0] in STOP_START:
            tokens.pop(0)
        title = ""
        if tokens and tokens[0] in TITLES:
            title = tokens.pop(0)
        entity = " ".join(tokens)
        rec = {"raw": raw, "entity": entity, "type": "", "rule": "", "kept": True}
        if not tokens or len(entity) < 2:
            rec.update(kept=False, rule="dropped: only stop-word / title left")
            out.append(rec)
            continue
        typ, rule = classify(tokens, title)
        if (len(tokens) == 1 and m.start() == 0 and tokens[0] == first_raw
                and typ == "MISC/CONCEPT" and not title):
            rec.update(type=typ, kept=False,
                       rule="dropped: capitalised only because it starts the sentence")
        else:
            rec.update(type=typ, rule=rule)
        out.append(rec)
    return out


def analyse_corpus(docs):
    """docs = [(name, text)]. Returns dict with every intermediate artefact."""
    sentences = []
    for name, text in docs:
        for s in split_sentences(text):
            sentences.append({"id": len(sentences) + 1, "doc": name, "text": s})

    freq, occ = Counter(), defaultdict(set)
    docs_of, types, rules = defaultdict(set), {}, {}
    trace = []
    for s in sentences:
        cands = extract_candidates(s["text"])
        kept = set()
        for c in cands:
            trace.append({"Sent #": s["id"], "Raw match": c["raw"], "Cleaned entity": c["entity"],
                          "Type": c["type"], "Rule applied": c["rule"],
                          "Kept?": "✔" if c["kept"] else "✘"})
            if c["kept"]:
                kept.add(c["entity"])
                types[c["entity"]] = c["type"]
                rules[c["entity"]] = c["rule"]
        for e in kept:
            freq[e] += 1
            occ[e].add(s["id"])
            docs_of[e].add(s["doc"])

    rows = [{"Entity": e, "Type": types[e], "Frequency": f, "Doc. freq.": len(docs_of[e]),
             "Sentence IDs": ", ".join(map(str, sorted(occ[e])))} for e, f in freq.items()]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Frequency", "Entity"], ascending=[False, True]).reset_index(drop=True)
    return {"sentences": sentences, "trace": pd.DataFrame(trace), "df": df, "occ": occ}


def build_graph(top_df, occ):
    G = nx.Graph()
    for _, r in top_df.iterrows():
        G.add_node(r["Entity"], type=r["Type"], freq=int(r["Frequency"]))
    edges = []
    for a, b in combinations(list(top_df["Entity"]), 2):
        shared = occ[a] & occ[b]
        if shared:
            G.add_edge(a, b, weight=len(shared))
            edges.append({"Entity A": a, "Entity B": b, "Weight": len(shared),
                          "Shared sentence IDs": ", ".join(map(str, sorted(shared)))})
    edges.sort(key=lambda e: (-e["Weight"], e["Entity A"], e["Entity B"]))
    for i, e in enumerate(edges, 1):
        e["Edge #"] = i
    return G, edges


def draw_graph(G, pos, edge_subset=None, scores=None, figsize=(7.5, 5.5)):
    fig, ax = plt.subplots(figsize=figsize)
    nodes = list(G.nodes)
    colors = [COLOR_MAP.get(G.nodes[n]["type"], "#777") for n in nodes]
    if scores:
        sizes = [500 + 9000 * scores[n] for n in nodes]
    else:
        sizes = [450 + 350 * G.nodes[n]["freq"] for n in nodes]
    edgelist = edge_subset if edge_subset is not None else list(G.edges)
    if edgelist:
        nx.draw_networkx_edges(G, pos, edgelist=edgelist, ax=ax, alpha=0.55,
                               width=[1 + G[u][v]["weight"] for u, v in edgelist])
        nx.draw_networkx_edge_labels(G, pos, ax=ax, font_size=7,
                                     edge_labels={(u, v): G[u][v]["weight"] for u, v in edgelist})
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_size=sizes,
                           edgecolors="black", linewidths=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    ax.axis("off")
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=k, markerfacecolor=v,
                          markersize=10) for k, v in COLOR_MAP.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    return fig


# ----------------------------------------------------------------------------------
# PAGERANK with every intermediate iteration recorded
# ----------------------------------------------------------------------------------
def pagerank_steps(G, d=0.85, tol=1e-6, max_iter=100):
    nodes = list(G.nodes)
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    A = np.zeros((n, n))
    for u, v, data in G.edges(data=True):
        A[idx[u], idx[v]] = A[idx[v], idx[u]] = data["weight"]
    out = A.sum(axis=1)
    M = np.zeros((n, n))                       # M[j,i] = P(i -> j)
    for i in range(n):
        M[:, i] = A[i, :] / out[i] if out[i] > 0 else 1.0 / n
    pr = np.full(n, 1.0 / n)
    history, deltas = [pr.copy()], []
    for _ in range(max_iter):
        new = (1 - d) / n + d * (M @ pr)
        delta = float(np.abs(new - pr).sum())
        history.append(new.copy())
        deltas.append(delta)
        pr = new
        if delta < tol:
            break
    return {"nodes": nodes, "idx": idx, "A": A, "out": out, "M": M, "d": d,
            "history": history, "deltas": deltas}


def worked_example(info, node, t):
    """Contribution table for PR_t(node) computed from PR_{t-1}."""
    nodes, idx, A, out, M = info["nodes"], info["idx"], info["A"], info["out"], info["M"]
    n, d = len(nodes), info["d"]
    prev = info["history"][t - 1]
    i = idx[node]
    rows, total = [], 0.0
    for u in range(n):
        if M[i, u] > 0:
            contrib = prev[u] * M[i, u]
            total += contrib
            rows.append({"Neighbour u": nodes[u], "PR(t-1) of u": prev[u],
                         "w(u,v)": A[u, i] if out[u] > 0 else np.nan,
                         "Σ weights leaving u": out[u] if out[u] > 0 else np.nan,
                         "Share = w / Σ": M[i, u], "Contribution = PR × Share": contrib})
    return pd.DataFrame(rows), total


# ----------------------------------------------------------------------------------
# RETRIEVAL EVALUATION with visible steps
# ----------------------------------------------------------------------------------
def eval_table(ranked, rel):
    hits, precs, rows = 0, [], []
    for r, e in enumerate(ranked, 1):
        is_rel = e in rel
        hits += int(is_rel)
        p = hits / r
        if is_rel:
            precs.append(p)
        rows.append({"Rank": r, "Entity": e, "Relevant?": "✔" if is_rel else "✘",
                     "Hits so far": hits, "Precision@rank": p,
                     "Counts toward AP?": "yes" if is_rel else "-"})
    ap = sum(precs) / len(rel) if rel else 0.0
    return pd.DataFrame(rows), precs, ap


# ----------------------------------------------------------------------------------
# FILE READING  (corpus upload)
# ----------------------------------------------------------------------------------
def read_uploaded(f):
    name, raw = f.name, f.getvalue()
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    try:
        if ext in ("txt", "md", "text"):
            return raw.decode("utf-8", errors="ignore")
        if ext == "csv":
            df = pd.read_csv(io.BytesIO(raw))
            cols = df.select_dtypes(include="object").columns
            return "\n".join(" , ".join(str(v) for v in row if str(v).strip() and str(v) != "nan")
                             for row in df[cols].itertuples(index=False))
        if ext == "pdf":
            from pypdf import PdfReader
            return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(raw)).pages)
        if ext == "docx":
            from docx import Document
            return "\n".join(p.text for p in Document(io.BytesIO(raw)).paragraphs)
    except ImportError as e:
        st.warning(f"Could not read **{name}**: missing library ({e.name}). "
                   f"Install it with `pip install {'pypdf' if ext == 'pdf' else 'python-docx'}`.")
        return ""
    except Exception as e:  # noqa
        st.warning(f"Could not read **{name}**: {e}")
        return ""
    st.warning(f"Unsupported file type: {name}")
    return ""


# ----------------------------------------------------------------------------------
# QUIZ HELPER
# ----------------------------------------------------------------------------------
def run_quiz(prefix, questions):
    answers = []
    for i, q in enumerate(questions):
        answers.append(st.radio(f"{i + 1}. {q['q']}", q["options"], key=f"{prefix}{i}", index=None))
    if st.button("Submit", key=f"{prefix}_submit"):
        score = sum(a == q["answer"] for a, q in zip(answers, questions))
        st.success(f"You scored {score} / {len(questions)}")
        for i, (a, q) in enumerate(zip(answers, questions)):
            if a == q["answer"]:
                st.write(f"✅ Q{i + 1}: Correct. {q['why']}")
            else:
                st.write(f"❌ Q{i + 1}: Correct answer - **{q['answer']}**. {q['why']}")


# ====================================================================================
# PAGE: INTRODUCTION  (motivating)
# ====================================================================================
if page == "Introduction":
    md("""<div class="hook">
    <b>🔎 Ask yourself:</b> when you type <i>"economist who studied at Cambridge and worked in
    Delhi"</i> into a search engine, how does it know that <i>Cambridge</i> is a place,
    <i>Amartya Sen</i> is a person, and that the two are connected?<br><br>
    The answer is <b>entities and graphs</b>. Behind Google's knowledge panels, Amazon's
    product recommendations, fraud-detection systems in banks and even COVID-19 literature
    search tools, there is the same simple idea you will build in this lab:
    <b>find the entities → link them → rank the important ones.</b></div>""")
    c1, c2, c3 = st.columns(3)
    c1.markdown("### 🧩 Identify\nPick out *persons, organisations, locations and concepts* from raw text.")
    c2.markdown("### 🕸️ Connect\nTurn co-occurring entities into a *graph* of nodes and weighted edges.")
    c3.markdown("### 🏆 Rank\nUse *PageRank* to find the most influential entity and *measure* how good the ranking is.")
    st.markdown("---")
    md(f"""In about 20 minutes you will upload your own {T('Corpus')}, watch every intermediate
    computation, and discover why a graph-based ranking can beat a plain word-count.
    <span class="small-note">(Hover over the <span class="tip" data-tip="Like this! Hover
    tooltips explain technical terms wherever you see the dotted maroon underline.">dotted terms</span>
    anywhere in the lab to see definitions.)</span>""")
    st.info("👉 Use the navigator on the left. Suggested order: Pre-Test → Aim → Theory → "
            "Procedure → Simulation → Post-Test.")

# ====================================================================================
# PAGE: PRE-TEST
# ====================================================================================
elif page == "Pre-Test":
    st.header("Pre-Test")
    st.write("Check what you already know. Don't worry about wrong answers - explanations appear after submitting.")
    PRE = [
        {"q": "A graph consists of:", "options": ["Rows and columns only", "Nodes and edges",
         "Only numbers", "Only text"], "answer": "Nodes and edges",
         "why": "G = (V, E): vertices (nodes) and edges."},
        {"q": "Which is a named entity?", "options": ["quickly", "Indian Institute of Technology",
         "because", "running"], "answer": "Indian Institute of Technology",
         "why": "Named entities are proper names of persons, organisations, places etc."},
        {"q": "Search engines rank pages mainly by:", "options": ["Only page colour",
         "Only the number of words", "Link structure and relevance", "Alphabetical order"],
         "answer": "Link structure and relevance", "why": "PageRank uses the link structure."},
        {"q": "If 3 of the top 5 results are relevant, precision@5 is:", "options": ["0.3", "0.6", "0.5", "3"],
         "answer": "0.6", "why": "3/5 = 0.6."},
        {"q": "'Tokenisation' means:", "options": ["Encrypting text", "Splitting text into units such as words or sentences",
         "Translating text", "Deleting stop-words"], "answer": "Splitting text into units such as words or sentences",
         "why": "Tokenisation / segmentation breaks text into smaller units."},
    ]
    run_quiz("pre", PRE)

# ====================================================================================
# PAGE: AIM & OBJECTIVES
# ====================================================================================
elif page == "Aim & Objectives":
    md('<div class="section-card">', )
    st.header("Aim")
    md(f"""To identify graph entities (persons, organisations, locations and domain concepts) from a
    {T('Corpus')} of text, represent their relationships as a small entity
    {T('Co-occurrence', 'co-occurrence')} graph, rank the nodes probabilistically with
    {T('PageRank')}, and compare the retrieval performance with a frequency baseline.""")
    st.markdown("</div>", unsafe_allow_html=True)
    st.header("Learning Objectives")
    md(f"""
1. Understand how {T('NER')} finds {T('Entity', 'entities')} in unstructured text.
2. Build a graph whose {T('Node', 'nodes')} are entities and whose {T('Edge', 'edges')} come from {T('Co-occurrence')}.
3. Understand {T('PageRank')} through the {T('Random surfer')} model and follow its {T('Iteration', 'iterations')} until {T('Convergence')}.
4. Evaluate rankings with {T('Precision@k')}, {T('Recall@k')} and {T('Average Precision')} against a {T('Ground truth')}.
5. Appreciate real-world applications: knowledge graphs, search, biomedical text mining, fraud analytics.
""")

# ====================================================================================
# PAGE: THEORY  (diagrams + applications + e-books)
# ====================================================================================
elif page == "Theory":
    st.header("Theory")
    tabs = st.tabs(["🌟 Why it matters", "1 · NER", "2 · Entity graph", "3 · PageRank",
                    "4 · Evaluation", "🚀 Applications", "📚 Further reading"])

    # ---------------- Why it matters --------------------------------------------------
    with tabs[0]:
        md("""<div class="hook">Roughly <b>80 %</b> of the world's data is unstructured text - news,
        papers, emails, reports. Computers cannot query a paragraph, but they <i>can</i> query a graph.
        Turning text into a graph of entities is the first step to making machines <b>understand who did
        what, where and with whom</b>.</div>""")
        st.markdown("**The pipeline of this experiment**")
        st.graphviz_chart(r"""
        digraph {
          rankdir=LR; nodesep=0.3;
          node [shape=box, style="rounded,filled", fillcolor="#f8e8e8", fontname="Helvetica", fontsize=11];
          A [label="Raw text /\nCorpus files"];
          B [label="Sentence\nsegmentation"];
          C [label="Candidate\nextraction"];
          D [label="Entity typing\n(NER rules)"];
          E [label="Co-occurrence\ncounting"];
          F [label="Entity graph\nG = (V, E)", fillcolor="#e3ecf7"];
          G [label="PageRank\n(probabilistic)", fillcolor="#e3ecf7"];
          H [label="Evaluation\nP@k, R@k, AP", fillcolor="#e6f4e6"];
          A -> B -> C -> D -> E -> F -> G -> H;
        }""")

    # ---------------- NER -------------------------------------------------------------
    with tabs[1]:
        md(f"""### Named Entity Recognition
{T('NER')} is an information-extraction task that labels spans of text with categories such as
**PERSON**, **ORGANIZATION**, **LOCATION** and **MISC / CONCEPT**. Each recognised
{T('Entity')} later becomes a graph {T('Node')}.""")
        sent = (ent_span("Amartya Sen", "PERSON") + " studied at the " +
                ent_span("University of Cambridge", "ORGANIZATION") + " in " +
                ent_span("Cambridge", "LOCATION") + " and later researched " +
                ent_span("famine economics", "MISC/CONCEPT") + ".")
        md(f"<div class='section-card' style='font-size:17px;line-height:2'>{sent}</div>")
        st.markdown("""**Lightweight rule-based extractor used in this lab** (no model download needed):
1. Candidate = a run of *Capitalised Words* (may include the linking words *of / for*).
2. Remove sentence-starting stop-words (*The, He, In...*) and honorifics (*Dr., Prof.*).
3. Type decision, in this order:
   - contains *University, Institute, Bank, Ministry, College...* → **ORGANIZATION**
   - had an honorific → **PERSON**
   - found in the place-name gazetteer → **LOCATION**
   - exactly two capitalised words → **PERSON**
   - otherwise → **MISC / CONCEPT**""")
        md(f"A {T('Gazetteer')} is simply a look-up list; modern systems replace all these rules "
           "with statistical / neural sequence labellers (CRF, BiLSTM, BERT) but the *output* is the same.")

    # ---------------- Entity graph ----------------------------------------------------
    with tabs[2]:
        md(f"""### Entity {T('Co-occurrence')} Graph
Two entities that appear in the **same sentence** are linked. The {T('Edge weight')} is the number
of sentences that contain both. This gives an undirected weighted graph G = (V, E).""")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("""**Example corpus**
- **S1**: *Amartya Sen* studied at *Cambridge*.
- **S2**: *Amartya Sen* taught at *Harvard University*.
- **S3**: *Amartya Sen* met *Jean Dreze* at *Harvard University*.
- **S4**: *Jean Dreze* lives in *New Delhi*.

Pairs in S3: (Sen, Dreze), (Sen, Harvard), (Dreze, Harvard) → each gets +1.
(Sen, Harvard) also co-occurs in S2 → weight **2**.""")
        with c2:
            st.graphviz_chart(r"""
            graph G {
              layout=neato; overlap=false; splines=true;
              node [style=filled, fontname="Helvetica", fontsize=11, fontcolor=white];
              "Amartya Sen" [fillcolor="#1f77b4"];
              "Jean Dreze" [fillcolor="#1f77b4"];
              "Harvard University" [fillcolor="#ff7f0e"];
              "Cambridge" [fillcolor="#2ca02c"];
              "New Delhi" [fillcolor="#2ca02c"];
              "Amartya Sen" -- "Cambridge" [label="1"];
              "Amartya Sen" -- "Harvard University" [label="2", penwidth=3];
              "Amartya Sen" -- "Jean Dreze" [label="1"];
              "Jean Dreze" -- "Harvard University" [label="1"];
              "Jean Dreze" -- "New Delhi" [label="1"];
            }""")
        md(f"The same information is stored in a {T('Co-occurrence matrix')}; the number of edges at a node is its "
           f"{T('Degree')}. Such a graph is a miniature {T('Knowledge Graph')}. We keep only the top-N entities "
           "(a *few-node* graph) so that it stays readable, instead of using a full graph database.")

    # ---------------- PageRank --------------------------------------------------------
    with tabs[3]:
        md(f"""### Probabilistic ranking - {T('PageRank')}
Imagine a {T('Random surfer')} walking on the entity graph. At each step the surfer follows an edge
(probability proportional to {T('Edge weight', 'weight')}) with probability **d** (the {T('Damping factor')}),
or jumps to a random node with probability **1-d**.""")
        st.latex(r"PR_t(v)=\frac{1-d}{N}+d\sum_{u\in N(v)}PR_{t-1}(u)\cdot\frac{w(u,v)}{\sum_{k}w(u,k)}")
        md(f"Repeat until the scores stop changing ({T('Convergence')}). The result is a probability distribution "
           f"over nodes: an entity linked to many *important* entities gets a high score. "
           f"A node without edges is a {T('Dangling node')} - the surfer teleports uniformly from it.")
        ex = nx.Graph()
        ex.add_weighted_edges_from([("A", "B", 2), ("A", "C", 1), ("B", "C", 1), ("C", "D", 1)])
        dot = "digraph {rankdir=LR; node [shape=circle, style=filled, fillcolor='#f8e8e8'];"
        for u in ex.nodes:
            tot = sum(ex[u][v]["weight"] for v in ex[u])
            for v in ex[u]:
                dot += f' "{u}" -> "{v}" [label="{ex[u][v]["weight"] / tot:.2f}"];'
        dot += "}"
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**Random-surfer view** - labels are transition probabilities "
                        "= w(u,v) / Σ w(u,·)")
            st.graphviz_chart(dot)
        with c2:
            pr = nx.pagerank(ex, alpha=0.85)
            st.markdown("**Resulting PageRank (d = 0.85)**")
            st.dataframe(pd.DataFrame({"Node": list(pr), "PageRank": list(pr.values())})
                         .sort_values("PageRank", ascending=False), hide_index=True)
            st.caption("C wins: it is connected to everybody (highest weighted centrality).")

    # ---------------- Evaluation ------------------------------------------------------
    with tabs[4]:
        md(f"""### Comparative retrieval performance
Given a {T('Ground truth')} of relevant entities, we judge a ranked list with three numbers.""")
        rel_pos = {1, 3, 4, 7}
        boxes = "".join(f'<span class="rankbox" style="background:{"#2ca02c" if i in rel_pos else "#999"}">{i}</span>'
                        for i in range(1, 9))
        md(f"<div class='section-card'><b>Ranked list (green = relevant, R = 4)</b><br>{boxes}</div>")
        hits5 = len([p for p in rel_pos if p <= 5])
        precs = [(j + 1) / p for j, p in enumerate(sorted(rel_pos))]
        st.latex(rf"Precision@5=\frac{{{hits5}}}{{5}}={hits5 / 5:.2f}\qquad "
                 rf"Recall@5=\frac{{{hits5}}}{{4}}={hits5 / 4:.2f}")
        st.latex(r"AP=\frac{1}{R}\sum_{k:\,rel_k=1}Precision@k=\frac{1}{4}\left(\frac11+\frac23+\frac34+\frac47\right)="
                 + f"{sum(precs) / 4:.3f}")
        md(f"{T('Precision@k')} = purity of the top-k; {T('Recall@k')} = coverage of the relevant set; "
           f"{T('Average Precision')} rewards putting relevant items *early*.")

    # ---------------- Applications ----------------------------------------------------
    with tabs[5]:
        st.markdown("""### Where is this used in real life?
| Domain | How entity graphs + ranking are used |
|---|---|
| 🔍 **Web search** | Google-style *knowledge panels*, entity-based query understanding |
| 🧬 **Biomedical mining** | Genes ↔ diseases ↔ drugs graphs from PubMed abstracts; drug re-purposing |
| 🏦 **Finance & fraud** | Linking persons, companies and accounts to detect suspicious rings |
| 📰 **News analytics** | Tracking who/what/where trends; automatic timelines |
| ⚖️ **Legal / e-discovery** | Finding key persons and organisations in millions of documents |
| 🛒 **Recommendation** | Product–brand–user graphs; PageRank-like scores for popular items |
| 🕵️ **Intelligence & security** | Social-network analysis of key influencers |
| 🎓 **Scholarly search** | Author-institution-topic graphs; ranking influential researchers |
| 🤖 **LLM / RAG systems** | Knowledge-graph-augmented question answering |""")

    # ---------------- Further reading -------------------------------------------------
    with tabs[6]:
        st.markdown("""### E-books & open resources
- 📘 Jurafsky & Martin - *Speech and Language Processing (3rd ed. draft)* - NER / sequence labelling: https://web.stanford.edu/~jurafsky/slp3/
- 📘 Manning, Raghavan, Schütze - *Introduction to Information Retrieval* - Ch. 8 (Evaluation), Ch. 21 (Link analysis / PageRank): https://nlp.stanford.edu/IR-book/
- 📘 Easley & Kleinberg - *Networks, Crowds, and Markets* - Ch. 2 (Graphs), Ch. 14 (Link analysis & web search): https://www.cs.cornell.edu/home/kleinber/networks-book/
- 📘 Bird, Klein, Loper - *Natural Language Processing with Python (NLTK Book)* - Ch. 7 Extracting Information from Text: https://www.nltk.org/book/ch07.html
- 📘 Hogan et al. - *Knowledge Graphs* (open e-book / survey): https://arxiv.org/abs/2003.02320
- 📘 Barabási - *Network Science* (free online): https://networksciencebook.com/
- 📄 Page, Brin, Motwani, Winograd - *The PageRank Citation Ranking* (1999): http://ilpubs.stanford.edu:8090/422/""")

# ====================================================================================
# PAGE: PROCEDURE
# ====================================================================================
elif page == "Procedure":
    st.header("Procedure")
    md(f"""
1. Open the **Simulation** page.
2. Choose the input: *sample text*, *type/paste text*, or **upload a {T('Corpus')}** (txt, md, csv, pdf, docx - many files allowed).
3. Click **Extract Entities**. Inspect the intermediate tables: sentence segmentation, candidate-extraction trace (with the rule that fired), entity table.
4. Choose how many top entities to keep and click **Build Graph**. Inspect the {T('Co-occurrence matrix')}, the edge list and use the slider to watch the graph grow **edge by edge**.
5. Choose the {T('Damping factor')} and click **Compute PageRank**. Study the {T('Transition matrix')}, step through every {T('Iteration')}, see a worked calculation for any node and the {T('Convergence')} plot.
6. Mark your {T('Ground truth')} entities and study the {T('Precision@k')}, {T('Recall@k')} and {T('Average Precision')} tables of both rankings.
7. Finish with the **Post-Test**.
""")

# ====================================================================================
# PAGE: SIMULATION
# ====================================================================================
elif page == "Simulation":
    st.header("Simulation - Identify Graph Entities")

    SAMPLE_1 = (
        "Dr. Amartya Sen studied at the University of Cambridge before joining Harvard University "
        "in the United States. He later worked closely with Jean Dreze at the Delhi School of "
        "Economics in New Delhi. The World Bank and the United Nations have both cited his work on "
        "welfare economics. Amartya Sen was born in Santiniketan, West Bengal, and remains "
        "associated with Trinity College in Cambridge. The Reserve Bank of India has referenced his "
        "research on famine and poverty, and the Ministry of Finance in New Delhi consults "
        "economists such as Jean Dreze on public policy relating to poverty alleviation.")
    SAMPLE_2 = (
        "Prof. Jean Dreze collaborated with Amartya Sen on hunger and public action. "
        "The Delhi School of Economics hosted several joint seminars in New Delhi. "
        "Harvard University invited Amartya Sen to deliver lectures in Cambridge. "
        "The Reserve Bank of India and the World Bank discussed poverty measurement with Jean Dreze.")

    for k in ("ana", "ginfo", "prinfo"):
        st.session_state.setdefault(k, None)

    # ------------------------------------------------------------------ STEP 0 : input
    md('<span class="step-badge">Step 1</span> <b>Provide text / corpus</b>')
    source = st.radio("Input source",
                      ["Sample corpus (2 documents)", "Type / paste text",
                       "Upload corpus files (txt, md, csv, pdf, docx)"], horizontal=True)
    docs = []
    if source.startswith("Sample"):
        docs = [("sample_doc_1.txt", SAMPLE_1), ("sample_doc_2.txt", SAMPLE_2)]
        st.caption("Two short documents are pre-loaded so you can see corpus-level frequency and document frequency.")
    elif source.startswith("Type"):
        txt = st.text_area("Enter / edit input text:", value=SAMPLE_1, height=170)
        docs = [("typed_text", txt)]
    else:
        files = st.file_uploader("Upload one or more files (this forms your corpus)",
                                 type=["txt", "md", "csv", "pdf", "docx"], accept_multiple_files=True)
        for f in files or []:
            t = read_uploaded(f)
            if t.strip():
                docs.append((f.name, t))
        if not docs:
            st.info("Upload at least one readable file to continue.")

    if docs:
        with st.expander("👁 Preview corpus"):
            for n, t in docs:
                st.markdown(f"**{n}** - {len(t.split())} words")
                st.text(t[:600] + (" ..." if len(t) > 600 else ""))

    if st.button("🔍 Extract Entities", disabled=not docs):
        st.session_state.ana = analyse_corpus(docs)
        st.session_state.ginfo = None
        st.session_state.prinfo = None
        st.session_state.n_docs = len(docs)
        st.session_state.n_words = sum(len(t.split()) for _, t in docs)

    ana = st.session_state.ana
    if ana is None:
        st.info("Click **Extract Entities** to begin.")
        st.stop()

    df, occ, sentences = ana["df"], ana["occ"], ana["sentences"]
    if df.empty:
        st.error("No entities found. Try text with capitalised names.")
        st.stop()

    # ------------------------------------------------------------------ STEP 1 outputs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", st.session_state.n_docs, help=plain_help("Corpus"))
    c2.metric("Sentences", len(sentences), help=plain_help("Sentence segmentation"))
    c3.metric("Words", st.session_state.n_words)
    c4.metric("Distinct entities", len(df), help=plain_help("Entity"))

    md(f'<b>1a.</b> {T("Sentence segmentation")} - each sentence is a context window for co-occurrence')
    st.dataframe(pd.DataFrame(sentences).rename(columns={"id": "Sent #", "doc": "Document", "text": "Sentence"})
                 .head(40), hide_index=True, use_container_width=True, height=200)

    md(f'<b>1b.</b> Candidate-extraction trace - how every {T("NER", "NER")} decision was made')
    st.caption("Raw capitalised phrase → cleaned (stop-words/titles removed) → type + the exact rule that fired.")
    with st.expander("Show trace (first 150 rows)", expanded=True):
        st.dataframe(ana["trace"].head(150), hide_index=True, use_container_width=True, height=260)

    md(f'<b>1c.</b> Aggregated {T("Entity", "entity")} table')
    cl, cr = st.columns([2, 1])
    with cl:
        st.dataframe(df, hide_index=True, use_container_width=True, height=280, column_config={
            "Entity": st.column_config.TextColumn("Entity", help=plain_help("Entity")),
            "Frequency": st.column_config.NumberColumn("Frequency", help=plain_help("Frequency")),
            "Doc. freq.": st.column_config.NumberColumn("Doc. freq.", help=plain_help("Document frequency")),
            "Sentence IDs": st.column_config.TextColumn("Sentence IDs", help="Sentences in which the entity occurs; "
                                                        "used later to compute co-occurrence.")})
    with cr:
        tc = df["Type"].value_counts()
        fig1, ax1 = plt.subplots(figsize=(4, 3.4))
        ax1.bar(tc.index, tc.values, color=[COLOR_MAP.get(t, "#999") for t in tc.index])
        ax1.set_ylabel("Count")
        ax1.set_title("Entities by type")
        plt.xticks(rotation=25, ha="right", fontsize=8)
        st.pyplot(fig1)

    st.markdown("---")
    # ------------------------------------------------------------------ STEP 2 : graph
    md('<span class="step-badge">Step 2</span> <b>Build the entity co-occurrence graph (few nodes)</b>')
    max_n = min(15, len(df))
    if max_n < 3:
        st.warning("Need at least 3 entities to build a graph.")
        st.stop()
    if max_n > 3:
        n_nodes = st.slider("Number of top-frequency entities to keep as nodes", 3, max_n, min(8, max_n),
                            help="Only the most frequent entities are kept so the graph stays small and readable.")
    else:
        n_nodes = 3

    if st.button("🕸️ Build Graph"):
        top_df = df.head(n_nodes).reset_index(drop=True)
        G, edges = build_graph(top_df, occ)
        pos = nx.spring_layout(G, seed=42, k=1.2)
        st.session_state.ginfo = {"top": top_df, "G": G, "edges": edges, "pos": pos}
        st.session_state.prinfo = None

    gi = st.session_state.ginfo
    if gi is None:
        st.info("Click **Build Graph**.")
        st.stop()
    G, edges, pos, top_df = gi["G"], gi["edges"], gi["pos"], gi["top"]

    md(f'<b>2a.</b> Nodes = top-{len(top_df)} entities by {T("Frequency")}')
    st.dataframe(top_df, hide_index=True, use_container_width=True)

    md(f'<b>2b.</b> {T("Co-occurrence matrix")} - cell (i,j) = number of sentences containing both entities')
    ents = list(top_df["Entity"])
    Mco = np.zeros((len(ents), len(ents)), dtype=int)
    for i, a in enumerate(ents):
        for j, b in enumerate(ents):
            if i != j:
                Mco[i, j] = len(occ[a] & occ[b])
    cm1, cm2 = st.columns([1, 1])
    with cm1:
        st.dataframe(pd.DataFrame(Mco, index=ents, columns=ents), use_container_width=True)
    with cm2:
        figh, axh = plt.subplots(figsize=(5, 4.2))
        axh.imshow(Mco, cmap="Reds")
        axh.set_xticks(range(len(ents)))
        axh.set_yticks(range(len(ents)))
        axh.set_xticklabels(ents, rotation=60, ha="right", fontsize=7)
        axh.set_yticklabels(ents, fontsize=7)
        for i in range(len(ents)):
            for j in range(len(ents)):
                if Mco[i, j]:
                    axh.text(j, i, Mco[i, j], ha="center", va="center", fontsize=7)
        st.pyplot(figh)

    md(f'<b>2c.</b> Edge list - each non-zero off-diagonal cell becomes an {T("Edge")} with {T("Edge weight", "weight")} = shared sentences')
    edges_df = pd.DataFrame(edges)[["Edge #", "Entity A", "Entity B", "Weight", "Shared sentence IDs"]] if edges else pd.DataFrame()
    if edges:
        st.dataframe(edges_df, hide_index=True, use_container_width=True, column_config={
            "Weight": st.column_config.NumberColumn("Weight", help=plain_help("Edge weight"))})
    else:
        st.warning("No co-occurrences among the chosen nodes. Increase the node count or use richer text.")

    md('<b>2d.</b> Watch the graph being built <b>edge by edge</b> (strongest edges first)')
    n_e = len(edges)
    k_e = st.slider("Edges added so far", 0, max(n_e, 1), n_e, disabled=(n_e == 0),
                    help="Drag to replay graph construction. Nodes are placed first; every step adds one edge.")
    subset = [(e["Entity A"], e["Entity B"]) for e in edges[:k_e]]
    gc1, gc2 = st.columns([2, 1])
    with gc1:
        st.pyplot(draw_graph(G, pos, edge_subset=subset))
    with gc2:
        if k_e > 0:
            e = edges[k_e - 1]
            st.markdown(f"**Latest edge (#{k_e})**\n\n`{e['Entity A']}` — `{e['Entity B']}`\n\n"
                        f"They co-occur in sentence(s) **{e['Shared sentence IDs']}** → weight **{e['Weight']}**.")
        else:
            st.markdown("Only nodes placed - no edges yet. Node **size ∝ frequency**, colour = entity type.")
        deg = dict(G.degree())
        st.markdown("**Summary**")
        st.write(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
        st.dataframe(pd.DataFrame({"Entity": list(deg), "Degree": list(deg.values())})
                     .sort_values("Degree", ascending=False), hide_index=True, height=180,
                     column_config={"Degree": st.column_config.NumberColumn("Degree", help=plain_help("Degree"))})

    st.markdown("---")
    # ------------------------------------------------------------------ STEP 3 : PageRank
    md('<span class="step-badge">Step 3</span> <b>Probabilistic ranking (PageRank) vs frequency ranking</b>')
    damping = st.slider("Damping factor d", 0.50, 0.95, 0.85, 0.05, help=plain_help("Damping factor"))
    if st.button("📊 Compute PageRank"):
        if G.number_of_edges() == 0:
            st.warning("Graph has no edges - PageRank would be uniform. Add more nodes or richer text.")
        st.session_state.prinfo = pagerank_steps(G, damping)

    info = st.session_state.prinfo
    if info is None:
        st.info("Click **Compute PageRank**.")
        st.stop()

    nodes, n, d = info["nodes"], len(info["nodes"]), info["d"]
    hist, deltas = info["history"], info["deltas"]

    md(f'<b>3a.</b> Start: every node gets equal probability PR₀ = 1/N = 1/{n} = <b>{1 / n:.4f}</b>')
    md(f'<b>3b.</b> {T("Transition matrix")} M - column i shows where the {T("Random surfer")} goes from node i '
       f'(w(i,j) ÷ total weight of i). Each column sums to 1.')
    Mdf = pd.DataFrame(info["M"], index=nodes, columns=nodes)
    st.dataframe(Mdf.style.format("{:.3f}").background_gradient(cmap="Blues"), use_container_width=True)
    if (info["out"] == 0).any():
        md(f'Some nodes are {T("Dangling node", "dangling")}: their column is uniform 1/N.')

    md(f'<b>3c.</b> {T("Iteration", "Iteration")} explorer - PRₜ = (1-d)/N + d · M · PRₜ₋₁')
    T_max = len(deltas)
    t = st.slider("Iteration t", 1, max(T_max, 1), 1, help="Step through the algorithm to see how scores evolve.")
    it_df = pd.DataFrame({"Entity": nodes, f"PR({t - 1})": hist[t - 1], f"PR({t})": hist[t]})
    it_df["Change"] = it_df[f"PR({t})"] - it_df[f"PR({t - 1})"]
    ic1, ic2 = st.columns([1, 1])
    with ic1:
        st.dataframe(it_df.style.format({f"PR({t - 1})": "{:.4f}", f"PR({t})": "{:.4f}", "Change": "{:+.4f}"}),
                     hide_index=True, use_container_width=True)
        st.caption(f"{T('L1 delta')} at this iteration = {deltas[t - 1]:.6f}")
        md("", )
    with ic2:
        node_sel = st.selectbox("Worked example - pick a node v", nodes,
                                help="See exactly how PR(v) at this iteration is computed from its neighbours.")
        wdf, total = worked_example(info, node_sel, t)
        st.dataframe(wdf.style.format({c: "{:.4f}" for c in wdf.columns if c != "Neighbour u"}, na_rep="1/N (dangling)"),
                     hide_index=True, use_container_width=True)
        val = (1 - d) / n + d * total
        st.latex(rf"PR_{{{t}}}(\text{{{node_sel}}})=\frac{{1-{d:.2f}}}{{{n}}}+{d:.2f}\times{total:.4f}={val:.4f}")
        st.caption(f"Check: matches table value {hist[t][info['idx'][node_sel]]:.4f}")

    md(f'<b>3d.</b> {T("Convergence")} - the {T("L1 delta")} shrinks each iteration; stop when it is < 1e-6 '
       f'(here after <b>{T_max}</b> iterations)')
    cv1, cv2 = st.columns([1, 1])
    with cv1:
        figc, axc = plt.subplots(figsize=(5, 3))
        axc.semilogy(range(1, T_max + 1), np.maximum(deltas, 1e-12), marker="o")
        axc.set_xlabel("Iteration")
        axc.set_ylabel("L1 delta (log)")
        axc.grid(alpha=0.3)
        st.pyplot(figc)
    with cv2:
        figt, axt = plt.subplots(figsize=(5, 3))
        H = np.array(hist)
        for i, v in enumerate(nodes):
            axt.plot(H[:, i], label=v)
        axt.set_xlabel("Iteration")
        axt.set_ylabel("PageRank")
        axt.legend(fontsize=6, ncol=2)
        st.pyplot(figt)
    with st.expander("Show full iteration history table"):
        st.dataframe(pd.DataFrame(hist, columns=nodes).rename_axis("Iteration").style.format("{:.4f}"),
                     use_container_width=True)

    md('<b>3e.</b> Final scores and the graph resized by PageRank')
    pr_final = hist[-1]
    pr_map = dict(zip(nodes, pr_final))
    nx_pr = nx.pagerank(G, alpha=d, weight="weight")
    diff = max(abs(pr_map[v] - nx_pr[v]) for v in nodes)
    freq_map = nx.get_node_attributes(G, "freq")
    tot_f = sum(freq_map.values())
    rank_df = pd.DataFrame({"Entity": nodes, "Type": [G.nodes[v]["type"] for v in nodes],
                            "PageRank Score": [pr_map[v] for v in nodes],
                            "Frequency Score": [freq_map[v] / tot_f for v in nodes]})
    rank_df["PageRank Rank"] = rank_df["PageRank Score"].rank(ascending=False, method="min").astype(int)
    rank_df["Frequency Rank"] = rank_df["Frequency Score"].rank(ascending=False, method="min").astype(int)
    rank_df = rank_df.sort_values("PageRank Score", ascending=False).reset_index(drop=True)
    r1, r2 = st.columns([1, 1])
    with r1:
        st.dataframe(rank_df.style.format({"PageRank Score": "{:.4f}", "Frequency Score": "{:.4f}"}),
                     hide_index=True, use_container_width=True)
        st.caption(f"Our from-scratch iteration vs NetworkX's pagerank(): max difference = {diff:.2e} ✔")
    with r2:
        st.pyplot(draw_graph(G, pos, scores=pr_map, figsize=(6, 4.5)))
        st.caption("Node size now ∝ PageRank score.")
    figb, axb = plt.subplots(figsize=(8, 3.2))
    x, w = np.arange(len(rank_df)), 0.35
    axb.bar(x - w / 2, rank_df["PageRank Score"], w, label="PageRank (probabilistic)")
    axb.bar(x + w / 2, rank_df["Frequency Score"], w, label="Frequency-based")
    axb.set_xticks(x)
    axb.set_xticklabels(rank_df["Entity"], rotation=35, ha="right", fontsize=8)
    axb.set_ylabel("Normalised score")
    axb.legend()
    st.pyplot(figb)

    st.markdown("---")
    # ------------------------------------------------------------------ STEP 4 : evaluation
    md('<span class="step-badge">Step 4</span> <b>Comparative retrieval performance</b>')
    md(f"Select the entities <i>you</i> consider important - this is your {T('Ground truth')}.")
    relevant = st.multiselect("Ground-truth relevant entities", list(rank_df["Entity"]),
                              help=plain_help("Ground truth"))
    k = st.number_input("k", 1, len(rank_df), min(3, len(rank_df)),
                        help="Cut-off for Precision@k and Recall@k: how many top results we inspect.")
    if not relevant:
        st.caption("Select at least one relevant entity to see the calculations.")
        st.stop()

    k = int(k)
    rel = set(relevant)
    R = len(rel)
    pr_rank = list(rank_df.sort_values(["PageRank Score", "Entity"], ascending=[False, True])["Entity"])
    fr_rank = list(rank_df.sort_values(["Frequency Score", "Entity"], ascending=[False, True])["Entity"])
    res = {}
    tabs = st.tabs(["PageRank ranking", "Frequency ranking"])
    for tab, name, ranked in zip(tabs, ["PageRank (probabilistic)", "Frequency-based"], [pr_rank, fr_rank]):
        with tab:
            tdf, precs, ap = eval_table(ranked, rel)
            hits_k = int(tdf.loc[k - 1, "Hits so far"])
            p_k, r_k = hits_k / k, hits_k / R
            res[name] = (p_k, r_k, ap)
            st.dataframe(tdf.style.format({"Precision@rank": "{:.3f}"}), hide_index=True, use_container_width=True,
                         height=min(400, 40 + 35 * len(tdf)))
            st.latex(rf"Precision@{k}=\frac{{{hits_k}}}{{{k}}}={p_k:.3f}\qquad Recall@{k}=\frac{{{hits_k}}}{{{R}}}={r_k:.3f}")
            terms = " + ".join(f"{p:.3f}" for p in precs) or "0"
            st.latex(rf"AP=\frac{{1}}{{{R}}}\left({terms}\right)={ap:.3f}")
            st.caption("Ties in the frequency ranking are broken alphabetically.")

    perf = pd.DataFrame({"Method": list(res),
                         f"Precision@{k}": [v[0] for v in res.values()],
                         f"Recall@{k}": [v[1] for v in res.values()],
                         "Average Precision": [v[2] for v in res.values()]})
    st.subheader("Summary")
    m1, m2, m3 = st.columns(3)
    a, b = list(res.values())
    m1.metric(f"Precision@{k}: PageRank", f"{a[0]:.2f}", f"{a[0] - b[0]:+.2f} vs Freq", help=plain_help("Precision@k"))
    m2.metric(f"Recall@{k}: PageRank", f"{a[1]:.2f}", f"{a[1] - b[1]:+.2f} vs Freq", help=plain_help("Recall@k"))
    m3.metric("Avg Precision: PageRank", f"{a[2]:.2f}", f"{a[2] - b[2]:+.2f} vs Freq", help=plain_help("Average Precision"))
    sc1, sc2 = st.columns([1, 1])
    with sc1:
        st.dataframe(perf.style.format({c: "{:.2f}" for c in perf.columns[1:]}), hide_index=True,
                     use_container_width=True)
    with sc2:
        figp, axp = plt.subplots(figsize=(5.5, 3.2))
        xm = np.arange(3)
        axp.bar(xm - 0.16, perf.iloc[0, 1:], 0.32, label="PageRank")
        axp.bar(xm + 0.16, perf.iloc[1, 1:], 0.32, label="Frequency")
        axp.set_xticks(xm)
        axp.set_xticklabels([f"P@{k}", f"R@{k}", "AP"])
        axp.set_ylim(0, 1.05)
        axp.legend()
        st.pyplot(figp)
    if a[2] > b[2]:
        st.success("PageRank achieves the higher Average Precision: the graph structure helped push relevant entities up.")
    elif a[2] < b[2]:
        st.warning("Frequency achieves the higher Average Precision for this ground truth - graph structure did not help here. "
                   "Try a different ground truth or a larger corpus.")
    else:
        st.info("Both methods give the same Average Precision for this ground truth.")

# ====================================================================================
# PAGE: POST-TEST
# ====================================================================================
elif page == "Post-Test":
    st.header("Post-Test")
    POST = [
        {"q": "Which is NOT a typical named-entity category?", "options": ["Person", "Organization", "Location", "Verb phrase"],
         "answer": "Verb phrase", "why": "NER labels proper names, not verbs."},
        {"q": "In a co-occurrence graph, an edge means:", "options": ["Same spelling length",
         "Entities appear together in the same sentence/context", "Same programming language", "They are antonyms"],
         "answer": "Entities appear together in the same sentence/context", "why": "Co-occurrence defines the edge."},
        {"q": "PageRank is best described as:", "options": ["A sorting algorithm", "A random-surfer probabilistic ranking",
         "A string-matching algorithm", "A compression technique"],
         "answer": "A random-surfer probabilistic ranking", "why": "Scores are the stationary probabilities of a random walk with teleportation."},
        {"q": "Increasing the damping factor d towards 1 means the surfer:", "options": ["Teleports more often",
         "Follows edges more, teleports less", "Stops moving", "Ignores edge weights"],
         "answer": "Follows edges more, teleports less", "why": "d = probability of following an edge; 1-d = teleport."},
        {"q": "Precision@k measures:", "options": ["Fraction of relevant items among the top-k results",
         "Number of graph edges", "Retrieval time", "Fraction of all relevant items found"],
         "answer": "Fraction of relevant items among the top-k results", "why": "hits_in_top_k / k."},
        {"q": "If R = 5 relevant entities exist and top-4 contains 2 of them, Recall@4 is:", "options": ["0.5", "0.4", "0.2", "2"],
         "answer": "0.4", "why": "2 / 5 = 0.4."},
        {"q": "Compared with frequency ranking, graph-based ranking also exploits:", "options": ["Alphabetical order",
         "Relationships between entities", "File size", "Number of vowels"],
         "answer": "Relationships between entities", "why": "PageRank uses the edge structure."},
        {"q": "PageRank iterations are stopped when:", "options": ["The graph has 10 nodes", "The L1 delta falls below a tolerance",
         "The first node reaches 1", "The ground truth is empty"],
         "answer": "The L1 delta falls below a tolerance", "why": "That indicates convergence."},
    ]
    run_quiz("post", POST)

# ====================================================================================
# PAGE: REFERENCES
# ====================================================================================
elif page == "References":
    st.header("References")
    st.subheader("📘 E-books (freely available online)")
    st.markdown("""
1. Jurafsky, D., Martin, J. H. *Speech and Language Processing* (3rd ed. draft) - https://web.stanford.edu/~jurafsky/slp3/
2. Manning, C. D., Raghavan, P., Schütze, H. *Introduction to Information Retrieval*, Cambridge Univ. Press, 2008 - https://nlp.stanford.edu/IR-book/
3. Easley, D., Kleinberg, J. *Networks, Crowds, and Markets*, Cambridge Univ. Press, 2010 - https://www.cs.cornell.edu/home/kleinber/networks-book/
4. Bird, S., Klein, E., Loper, E. *Natural Language Processing with Python*, O'Reilly, 2009 - https://www.nltk.org/book/
5. Hogan, A. et al. *Knowledge Graphs*, ACM Computing Surveys, 2021 - https://arxiv.org/abs/2003.02320
6. Barabási, A.-L. *Network Science*, Cambridge Univ. Press, 2016 - https://networksciencebook.com/
""")
    st.subheader("📄 Papers")
    st.markdown("""
7. Page, L., Brin, S., Motwani, R., Winograd, T. *The PageRank Citation Ranking: Bringing Order to the Web*, Stanford InfoLab, 1999.
8. Nadeau, D., Sekine, S. *A survey of named entity recognition and classification*, Lingvisticae Investigationes, 2007.
""")
    st.subheader("🔗 Web resources")
    st.markdown("""
9. IIT Kharagpur Virtual Labs - https://vlab.iitkgp.ac.in/
10. NetworkX documentation - https://networkx.org/
11. Streamlit documentation - https://docs.streamlit.io/
""")

# ====================================================================================
# PAGE: FEEDBACK
# ====================================================================================
elif page == "Feedback":
    st.header("Feedback")
    st.slider("How useful was this experiment?", 1, 5, 4)
    st.text_area("Share your feedback / suggestions about this virtual lab experiment:")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")

st.markdown("---")
st.caption("Streamlit-based Virtual Lab - Experiment 26-30: Identify Graph Entities. Graph visualisation uses "
           "NetworkX + Matplotlib (few-node in-memory graph); no external graph database (e.g. Neo4j) is used.")