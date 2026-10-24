"""Streamlit helpers shared by all pages."""

import streamlit as st
from langchain_qdrant import QdrantVectorStore

from research_analyst.agent import ResearchAgent, get_checkpointer
from research_analyst.vectorstore import get_vectorstore


@st.cache_resource
def get_store() -> QdrantVectorStore:
    # One client for every page and rerun: Qdrant local mode locks its folder.
    return get_vectorstore()


@st.cache_resource
def get_research_graph():
    return ResearchAgent(get_store()).build(get_checkpointer())
