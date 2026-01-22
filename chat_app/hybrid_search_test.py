# hybrid_search_test.py
# TODO:
# - questions
# - sources
# - Q <-> S connection list (Qids and Sids?)
# -	
# -
# -

TEST_SOURCES = [
	"wiki_pages/6-7 meme - Wikipedia.html",
	"wiki_pages/Chimpanzee - Wikipedia.html",
	"wiki_pages/Computer science - Wikipedia.html",
	"wiki_pages/IKEA - Wikipedia.html",
	"wiki_pages/Linux - Wikipedia.html",
	"wiki_pages/Malware - Wikipedia.html",
	"wiki_pages/Minecraft - Wikipedia.html",
	"wiki_pages/Number theory - Wikipedia.html",
	"wiki_pages/Poznań - Wikipedia.html",
	"wiki_pages/Poznań University of Technology - Wikipedia.html",
	"wiki_pages/SCP Foundation - Wikipedia.html",
]
KEYWORD_SEARCHES = [
	"What types of window managers exist for X11?",
	"What are the main subfields of computer science?",
	"What are the main types of malware?",
	"What game modes are available in Minecraft?",
	"What is the SCP Foundation?",
	"When was IKEA founded?",
	"What is number theory?",
	"When was Poznań University of Technology established?",
	"Where is Poznań located?",
	"Where do chimpanzees naturally live?",

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


store = RAGStore(chroma_dir="chroma_reseach")
store.ingest(TEST_SOURCES)

rag = RAGRetriever(store)

sources_with_ids = dict()

for source in TEST_SOURCES:
	ids = store.ingest(source)
	sources_with_ids[source] = ids

with open('sources_with_ids.json', 'w') as f:
	json.dump(sources_with_ids, f)

