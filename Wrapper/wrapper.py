import hashlib
import os
from dotenv import load_dotenv
load_dotenv()

import sentry_sdk
from langgraph.graph import StateGraph, END
from langchain.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from state import AgentState

sentry_dsn = os.getenv("SENTRY_DSN")
sentry_sdk.init(
    dsn=sentry_dsn,
    traces_sample_rate=1.0
)

class LangGraphPDFWrapper:

    def __init__(self):
        # Initialize Groq API model
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0
        )
        self.graph = self._build_graph()


#Node.1 LLM decides query
    def llm_decide(self, state: AgentState):
        with sentry_sdk.start_transaction(op="node", name="llm_decide"):
            try:
                prompt = f"""
You are an AI assistant. You have access to PDF documents.

User question:
{state["input"]}

Generate a short search query to retrieve relevant information from the PDF.
Return ONLY the query text without quotes.
"""
                response = self.llm.invoke([HumanMessage(content=prompt)])
                
                sentry_sdk.capture_message(f"llm_decide generated query: {response.content}")
                return {"search_query": response.content.strip('"')}
            
            except Exception as e:
                sentry_sdk.capture_exception(e)
                raise


#Node 2. pdf tool
    def pdf_tool(self, state: AgentState):
        with sentry_sdk.start_transaction(op="node", name="pdf_tool"):
            try:
                pdf_path = state["pdf_path"]

                # Generate unique hash for PDF
                with open(pdf_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                index_dir = "faiss_indexes"
                index_path = os.path.join(index_dir, file_hash)

                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

                # If FAISS index exists → load it
                if os.path.exists(index_path):
                    sentry_sdk.capture_message(f"Loading existing FAISS index for {pdf_path}")
                    db = FAISS.load_local(
                        index_path,
                        embeddings,
                        allow_dangerous_deserialization=True
                    )

                # Otherwise create FAISS index
                else:
                    sentry_sdk.capture_message(f"Creating FAISS index for {pdf_path}")

                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()

                    sentry_sdk.capture_message(f"Loaded PDF: {pdf_path} with {len(docs)} pages")

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )

                    documents = splitter.split_documents(docs)

                    db = FAISS.from_documents(documents, embeddings)

                    os.makedirs(index_dir, exist_ok=True)
                    db.save_local(index_path)

                    sentry_sdk.capture_message(f"Saved FAISS index to {index_path}")

                results = db.similarity_search(state["search_query"], k=3)

                content = "\n".join([r.page_content for r in results])

                sentry_sdk.capture_message(
                    f"PDF tool returned {len(results)} top chunks for query: {state['search_query']}"
                )

                return {"search_results": content}

            except Exception as e:
                sentry_sdk.capture_exception(e)
                raise
                
#Node 3. LLM generates final answer
    def final_llm(self, state: AgentState):
        with sentry_sdk.start_transaction(op="node", name="final_llm"):
            try:
                prompt = f"""
You are an AI assistant.

User question:
{state['input']}

Information retrieved from PDF documents:
{state['search_results']}

Answer the user question using the retrieved information.
"""
                response = self.llm.invoke([HumanMessage(content=prompt)])
                sentry_sdk.capture_message("final_llm generated answer")
                return {"output": response.content}
            
            except Exception as e:
                sentry_sdk.capture_exception(e)
                raise

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("decide", self.llm_decide)
        builder.add_node("pdf_tool", self.pdf_tool)
        builder.add_node("final", self.final_llm)

        builder.set_entry_point("decide")
        builder.add_edge("decide", "pdf_tool")
        builder.add_edge("pdf_tool", "final")
        builder.add_edge("final", END)

        return builder.compile()

    def run(self, user_input: str, pdf_path: str):
        try:
            result = self.graph.invoke({
                "input": user_input,
                "pdf_path": pdf_path
            })
            return result["output"]
        except Exception as e:
            sentry_sdk.capture_exception(e)
            raise