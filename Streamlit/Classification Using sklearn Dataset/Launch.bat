@echo off
call activate my-env
D:
cd D:\jupyter\GitHub\Streamlit
streamlit run Streamlit_App.py
call conda deactivate