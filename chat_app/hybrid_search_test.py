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
	"What free software license is used for the Linux kernel and many components of the GNU Project?",
	"When did computer science begin to be established as a distinct academic discipline?",
	"What is a computer virus and how does it spread to other programs or files?",
	"How is the Minecraft world generated as players explore it?",
	"What possible meanings have been suggested for the number mentioned in the song lyrics?",
	"Why did IKEA announce plans in 2019 to begin testing a furniture rental model?",
	"What arithmetic operations are studied in arithmetic for combining and transforming numbers?",
	"When was Poznan University of Technology officially founded and who was its first rector?",
	"Why was Poznan an important political and cultural center before the Christianization of Poland?",
	"When and where was the name “chimpanze” first recorded in written sources?",

]
SEMANTIC_SEARCHES = [
	"Which licensing model requires that software derived from shared source code must also remain freely redistributable?",
	"How did the expansion of computing beyond numerical calculation contribute to the formation of a new scientific field?",
	"How does a type of malicious software propagate by embedding itself into other executable components without a user’s awareness?",
	"How does a sandbox game create an effectively endless environment using a random or user-defined starting parameter?",
	"How have cultural, geographic, and linguistic interpretations been used to explain a numerical reference in a song?",
	"What environmental and consumption-related concerns motivated a large furniture retailer to experiment with renting products instead of selling them outright?",
	"How does the branch of mathematics concerned with numbers examine the ways values can be combined and transformed?",
	"How did a technical higher-education institution in Poznan evolve through several name and status changes before becoming a university?",
	"How did an early fortified settlement contribute to the formation of the first Polish state before its conversion to Christianity?",
	"How did a term referring to a great ape enter English through contact with African languages and later publications?",

]

FUZZY_SEARCHES = [
	"How does reciprocal licensing help distinguish a widely used free operating system from proprietary ones?",
	"What developments led to computing becoming recognized as its own area of academic study?",
	"How can harmful software hide inside legitimate programs and continue spreading when those programs are run?",
	"Why does the game world feel infinite even though technical limits still exist?",
	"Why does the artist describe the significance of the number in the song as intentionally unclear?",
	"How was renting furniture expected to address both sustainability concerns and customer needs for affordability and convenience?",
	"How do basic numerical procedures allow numbers to be combined into new values?",
	"What historical events and transformations shaped the development of a technical school in Poznan into a university after World War II?",
	"What role did this early stronghold play in the emergence of a new state and its adoption of Christianity?",
	"How did the name used for this primate evolve in English from early references to its modern and shortened forms?",

]

QUESTIONS = {}

i = 0
for q in KEYWORD_SEARCHES + SEMANTIC_SEARCHES + FUZZY_SEARCHES:
	QUESTIONS[q] = i
	if i == 9:
		i = 0
		continue
	else:
		i += 1

K = 3

# from .rag_store import RAGStore
# from .rag_retriever import RAGRetriever
import json
import os

# store = RAGStore(chroma_dir="chroma_reseach")
# rag = RAGRetriever(store)
# sources_with_ids = dict()

# for source in TEST_SOURCES:
# 	ids = store.ingest(source)
# 	sources_with_ids[source] = ids

# with open('sources_with_ids.json', 'w') as f:
# 	json.dump(sources_with_ids, f)

# core_chunks = dict()
# for query in KEYWORD_SEARCHES:
# 	dense = store.sparse_query(query, n_results=10)
# 	d_ids = dense.get("ids", [[]])[0] if "ids" in dense else []
# 	d_docs = dense.get("documents", [[]])[0]
# 	d_meta = dense.get("metadatas", [[]])[0]
# 	d_dists = dense.get("distances", [[]])[0]
# 	print(f"############################ THE Q: {query} ############################")
# 	for i, (doc, d_id) in enumerate(zip(d_docs, d_ids)):
# 		print(f"{i}. Chunk with id '{d_id}' ############################")
# 		print(doc, "\n")
# 	chosen_core_chunk = int(input("Which one good sir?\n>"))
# 	core_chunks[query] = [d_ids[chosen_core_chunk], chosen_core_chunk]
# 	os.system('cls' if os.name == 'nt' else 'clear')

# with open('core_chunks.json', 'w') as f:
# 	json.dump(core_chunks, f)

the_ids = [
	"c32600d5cc40a6f4ab5c1cae15b1824ec467f04f", 
	"cedefd9620cee2d849d8517db04cc1e5adac0494",
	"c5f82e938eea27aba5557984ab7361d874816405",
	"f896dcef05642d2a18d4737c2c0fc18ecff65563",
	"b7df9a16363ba1b9e2970abd1a17673b2865af7f",
	"cf20c76e2ee5f8fd1308b52415e38278b058453e",
	"8496eaa759fbb76e3f9bd6021074a1e249d116ec",
	"f021a2fce6ca227a3e10b0c302d8c672e71b65d7",
	"c61a3018281dd7bacc9b47e5db14758e8e95a3c3",
	"ee25a689db8f849a81aec57a6ee3a2479e470fdf",
]

# recall = 1 or 0
# mrr = 1/k

# results = dict()

# for query in KEYWORD_SEARCHES + SEMANTIC_SEARCHES + FUZZY_SEARCHES:
# 	dense = store.query(query, n_results=K, include=("documents","metadatas","distances"))
# 	d_ids = dense.get("ids", [[]])[0] if "ids" in dense else []
# 	sparse = store.sparse_query(query, n_results=K)
# 	s_ids = sparse.get("ids", [[]])[0] if "ids" in dense else []
# 	hybrid = rag.hybrid_query(query, top_k=K, include_ids=True)
# 	h_ids = hybrid.get("ids", [[]])[0] if "ids" in dense else []

# 	results[query] = {"dense": d_ids, "sparse": s_ids, "hybrid": h_ids}

# with open('results.json', 'w') as f:
# 	json.dump(results, f)

with open("results.json", "r") as f:
    results = json.loads(f.read())

recalls = {}	# {q: {s:int,...}} -> 
mrrs = {}
for q, data in results.items():
	correct_id = the_ids[QUESTIONS[q]]
	recall = {}
	mrr = {}
	for search, ids in data.items():
		 

		if correct_id in ids:
			recall[search] = 1
			mrr[search] = 1.0/(ids.index(correct_id) + 1)
		else:
			recall[search] = 0
			mrr[search] = 0.0
	recalls[q] = recall
	mrrs[q] = mrr

# print(recalls)

searches = ["dense", "sparse", "hybrid"]
types = ["BM25", "SEMANTIC", "FUZZY"]
mean_recall = {s: {t: 0.0 for t in types} for s in searches}
mean_mrr = {s: {t: 0.0 for t in types} for s in searches}

def get_query_type(query):
	if query in KEYWORD_SEARCHES:
		return "BM25"
	if query in SEMANTIC_SEARCHES:
		return "SEMANTIC"
	if query in FUZZY_SEARCHES:
		return "FUZZY"

for s in searches:
	for q, recall in recalls.items():
		qtype = get_query_type(q)
		mean_recall[s][qtype] += recall[s]
	for q, mrr in mrrs.items():
		qtype = get_query_type(q)
		mean_mrr[s][qtype] += mrr[s]
for s in searches:
	for t in types:
		mean_recall[s][t] = round(mean_recall[s][t] / 10, 2)
		mean_mrr[s][t] = round(mean_mrr[s][t] / 10, 2)

print(mean_recall)
print(mean_mrr)
