mkdir -p ~/.streamlit/echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
[theme]
base="dark"
primaryColor="#ff2d00"
backgroundColor="#002522"
" > ~/.streamlit/config.toml