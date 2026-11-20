"""ASGI entry point for Vercel; the local Streamlit command remains unchanged."""

import os
from pathlib import Path

# Vercel's request-body limit is lower than the standalone app's 10 MB limit.
os.environ["MACROMATE_UPLOAD_LIMIT_MB"] = "4"

import streamlit as st

app = st.App(Path(__file__).with_name("app.py"))
