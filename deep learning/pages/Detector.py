import os
import sys
import importlib.util
import streamlit as st

# Reuse the existing app logic by importing streamlit_app.py as a module
# This keeps a single source of truth for analytics/prediction.
MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py")
MODULE_PATH = os.path.abspath(MODULE_PATH)

st.set_page_config(page_title="Detector • Fake Review", page_icon="🕵️", layout="wide")

spec = importlib.util.spec_from_file_location("streamlit_app_module", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules["streamlit_app_module"] = mod
spec.loader.exec_module(mod)
