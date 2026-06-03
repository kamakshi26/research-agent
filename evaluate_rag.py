from ragas import evaluate
from ragas.metrics import NonLLMStringSimilarity
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from datasets import Dataset
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import re

# ─── Step 1: Build RAG pipeline ──────────────────────────────
print("Building RAG pipeline...")

reader = PdfReader("p15.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

def clean_pdf_text(text):
    text = re.sub(r'Page \d+ of \d+.*?\n', '', text)
    text = re.sub(r'\.{3,}.*?\d+', '', text)
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

clean_text = clean_pdf_text(full_text)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = [Document(page_content=clean_text)]
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Qdrant.from_documents(
    documents=chunks,
    embedding=embeddings,
    location=":memory:",
    collection_name="eval_docs"
)

llm = ChatOpenAI(
    model="gpt-4o",
    api_key="github token",
    base_url="https://models.inference.ai.azure.com",
    temperature=0
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Answer using ONLY the context below.
If not in context say "I don't know."

Context: {context}
Question: {question}
Answer:"""
)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# ─── Step 2: Define test questions ───────────────────────────
print("Running test questions...")

test_questions = [
    "What is the social security tax rate for 2026?",
    "What is the FUTA tax rate?",
    "What is the penalty for depositing taxes 8 days late?",
    "What is the social security wage base limit for 2026?",
    "What is the backup withholding rate?"
]

ground_truths = [
    "The social security tax rate for 2026 is 6.2% each for the employer and employee.",
    "The FUTA tax rate is 6.0%.",
    "The penalty for depositing taxes 6-15 days late is 5%.",
    "The social security wage base limit for 2026 is $184,500.",
    "The backup withholding rate is 24%."
]

# ─── Step 3: Collect answers and contexts ────────────────────
questions = []
answers = []
contexts = []

for question in test_questions:
    print(f"  Q: {question}")
    result = chain.invoke({"query": question})
    answer = result["result"]
    source_docs = result["source_documents"]
    context = [doc.page_content for doc in source_docs]

    questions.append(question)
    answers.append(answer)
    contexts.append(context)
    print(f"  A: {answer[:100]}...")

# ─── Step 4: Run RAGAS evaluation ────────────────────────────
print("\nRunning RAGAS evaluation...")

dataset = Dataset.from_dict({
    "user_input": questions,
    "response": answers,
    "retrieved_contexts": contexts,
    "reference": ground_truths
})

ragas_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

results = evaluate(
    dataset=dataset,
    metrics=[
        NonLLMStringSimilarity()
    ],
    embeddings=ragas_embeddings
)

# ─── Step 5: Print scores ─────────────────────────────────────
print("\n" + "="*50)
print("RAGAS EVALUATION RESULTS")
print("="*50)
df = results.to_pandas()
print(df.to_string())
print("\nAverage scores:")
for col in df.select_dtypes(include="number").columns:
    print(f"  {col}: {df[col].mean():.3f}")
print("="*50)