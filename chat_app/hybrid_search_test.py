# hybrid_search_test.py
# TODO:
# - questions
# - sources
# - Q <-> S connection list (Qids and Sids?)
# -	no rehydration on bm25
# -	
# -

TEST_SOURCES = [
	"wiki_pages/Linux - Wikipedia.html",
	"wiki_pages/Computer science - Wikipedia.html",
	"wiki_pages/Malware - Wikipedia.html",
	"wiki_pages/Minecraft - Wikipedia.html",
	"wiki_pages/6-7 meme - Wikipedia.html",
	"wiki_pages/IKEA - Wikipedia.html",
	"wiki_pages/Number theory - Wikipedia.html",
	"wiki_pages/Poznan University of Technology - Wikipedia.html",
	"wiki_pages/Poznan - Wikipedia.html",
	"wiki_pages/Chimpanzee - Wikipedia.html",
	"wiki_pages/SCP Foundation - Wikipedia.html",
]
KEYWORD_SEARCHES = [
	"What types of window managers exist for X11?",
	"What are crucial areas of computer science?",
	"In what ways can malware be classified?",
	"Who originally created Minecraft?",
	"When did six seven meme emerged?",
	"When was the first IKEA store opened?",
	"What is the earliest historical find of an arithmetical nature?",
	"What is a Poznań University of Technology?",
	"What was officially named Haupt- und Residenzstadt Posen?",
	"What does chimpanzee's diet consists of?",

]
SEMNATIC_SEARCHES = [
	"How does Linux separate kernel functionality from user-space programs?",
	"How did computer science develop as an academic discipline?",
	"Why is social engineering commonly used in malware attacks?",
	"Why is Minecraft considered a sandbox game?",
	"How does collaborative writing shape the SCP universe?",
	"Why does IKEA rely on flat-pack furniture as part of its business model?",
	"Why are prime numbers fundamental in number theory?",
	"What role does the university play in Polish engineering education?",
	"Why is Poznań historically significant for Poland?",
	"How do social structures influence chimpanzee behavior?",

]

FUZZY_SEARCHES = [
	"how linux keeps kernel apart from apps",
	"how computer science became its own field",
	"why malware tricks users instead of hacking systems",
	"why minecraft has no fixed goals",
	"how people collectively write scp stories",
	"why ikea furniture comes in boxes",
	"why primes matter so much in math",
	"why poznan tech uni is important in poland",
	"why poznan matters in polish history",
	"how chimps live together in groups",

]

from .rag_store import RAGStore
from .rag_retriever import RAGRetriever
import json
import os

store = RAGStore(chroma_dir="chroma_reseach")
rag = RAGRetriever(store)
sources_with_ids = dict()

# for source in TEST_SOURCES:
# 	ids = store.ingest(source)
# 	sources_with_ids[source] = ids

# with open('sources_with_ids.json', 'w') as f:
# 	json.dump(sources_with_ids, f)

core_chunks = dict()
for query in KEYWORD_SEARCHES + SEMNATIC_SEARCHES + FUZZY_SEARCHES:
	dense = store.query(query, n_results=10, include=("documents","metadatas","distances"))
	d_ids = dense.get("ids", [[]])[0] if "ids" in dense else []
	d_docs = dense.get("documents", [[]])[0]
	d_meta = dense.get("metadatas", [[]])[0]
	d_dists = dense.get("distances", [[]])[0]
	print(f"############################ THE Q: {query} ############################")
	for i, (doc, d_id) in enumerate(zip(d_docs, d_ids)):
		print(f"{i}. Chunk with id '{d_id}' ############################")
		print(doc, "\n")
	chosen_core_chunk = int(input("Which one good sir?\n>"))
	core_chunks[query] = d_ids[chosen_core_chunk]
	os.system('cls' if os.name == 'nt' else 'clear')

with open('core_chunks.json', 'w') as f:
	json.dump(core_chunks, f)
