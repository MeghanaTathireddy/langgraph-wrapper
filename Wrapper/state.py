from typing import TypedDict

class AgentState(TypedDict, total=False):
    input: str
    pdf_path: str
    search_query: str
    search_results: str
    output: str
