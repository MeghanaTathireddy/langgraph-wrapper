import hashlib
import os
from dotenv import load_dotenv
load_dotenv()

# Observability
import sentry_sdk
from langfuse import Langfuse
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor

# OpenTelemetry (Phoenix tracing)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# LangGraph / LangChain
from langgraph.graph import StateGraph, END
from langchain.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from state import AgentState


# -----------------------------
# Observability Setup
# -----------------------------

# Sentry
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    traces_sample_rate=1.0
)

# Langfuse
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

# OpenTelemetry + Phoenix
resource = Resource.create({"service.name": "langgraph-wrapper"})

trace.set_tracer_provider(TracerProvider(resource=resource))

otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:6006/v1/traces"
)

span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument LangChain
LangChainInstrumentor().instrument()

tracer = trace.get_tracer(__name__)


# -----------------------------
# LangGraph Wrapper
# -----------------------------

class LangGraphPDFWrapper:

    def __init__(self):

        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0
        )

        self.graph = self._build_graph()

    def llm_decide(self,state:AgentState):
        with sentry_sdk.start_transaction(op="node",name="llm_decide"):
            trace=langfuse.trace(name="llm_decide")
            span=trace.span(name="llm_decide_query",input=state["input"])

    # -----------------------------
    # Node 1 – LLM decides query
    # -----------------------------

    def llm_decide(self, state: AgentState):

        with sentry_sdk.start_transaction(op="node", name="llm_decide"):

            trace_obj = langfuse.trace(name="llm_decide")
            span = trace_obj.span(
                name="llm_decide_query",
                input=state["input"]
            )

            try:

                prompt = f"""
You are an AI assistant. You have access to PDF documents.

    User question:
    {state["input"]}

Generate a short search query to retrieve relevant information from the PDF.

Return ONLY the query text without quotes.
"""

                response = self.llm.invoke([HumanMessage(content=prompt)])

                span.end(output=response.content)

                return {"search_query": response.content.strip('"')}

            except Exception as e:
                sentry_sdk.capture_exception(e)
                raise

    def pdf_tool(self,state:AgentState):
        with sentry_sdk.start_transaction(op="node",name="pdf_tool"):
            trace=langfuse.trace(name="pdf_tool")

    # -----------------------------
    # Node 2 – PDF Retrieval Tool
    # -----------------------------

    def pdf_tool(self, state: AgentState):

        with sentry_sdk.start_transaction(op="node", name="pdf_tool"):

            trace_obj = langfuse.trace(name="pdf_tool")

            try:

                pdf_path = state["pdf_path"]

                with open(pdf_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                index_dir="faiss_indexes"
                index_path=os.path.join(index_dir,file_hash)

                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )

                if os.path.exists(index_path):

                    db = FAISS.load_local(
                        index_path,
                        embeddings,
                        allow_dangerous_deserialization=True
                    )

                else:

                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()

                    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
                    documents=splitter.split_documents(docs)

                    db=FAISS.from_documents(documents,embeddings)

                    os.makedirs(index_dir,exist_ok=True)
                    db.save_local(index_path)

                results = db.similarity_search(
                    state["search_query"],
                    k=3
                )

                content = "\n".join(
                    [r.page_content for r in results]
                )

                span = trace_obj.span(
                    name="pdf_retrieval",
                    input=state["search_query"]
                )

                span.end(output=content)

                return {"search_results":content}

            except Exception as e:
                sentry_sdk.capture_exception(e)
                raise


    # -----------------------------
    # Node 3 – Final LLM Answer
    # -----------------------------

    def final_llm(self, state: AgentState):

        with sentry_sdk.start_transaction(op="node", name="final_llm"):

            trace_obj = langfuse.trace(name="final_llm")

            span = trace_obj.span(
                name="final_llm_response",
                input=state["input"]
            )

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

                span.end(output=response.content)

                return {"output": response.content}

            except Exception as e:
                sentry_sdk.capture_exception(e)
                raise


    # -----------------------------
    # Build LangGraph
    # -----------------------------

    def _build_graph(self):

        builder = StateGraph(AgentState)

        builder.add_node("decide",self.llm_decide)
        builder.add_node("pdf_tool",self.pdf_tool)
        builder.add_node("final",self.final_llm)

        builder.set_entry_point("decide")

        builder.add_edge("decide", "pdf_tool")
        builder.add_edge("pdf_tool", "final")
        builder.add_edge("final", END)

        return builder.compile()


    # -----------------------------
    # Run wrapper
    # -----------------------------

    def run(self, user_input: str, pdf_path: str):

        try:

            result = self.graph.invoke({
                "input": user_input,
                "pdf_path": pdf_path
            })

            return result["output"]

        except Exception as e:
            sentry_sdk.capture_exception(e)
            print("ERROR:", e)
            raise