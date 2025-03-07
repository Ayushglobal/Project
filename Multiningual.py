import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Language detection handling
try:
    from langdetect import detect
except ImportError:
    st.error("Missing required package: langdetect. Install with: pip install langdetect")
    st.stop()

# Improved language detection with Indian language support
def detect_query_language(text):
    """Enhanced language detection with fallback"""
    lang_map = {
        # Major global languages
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'zh-cn': 'Chinese',
        'ar': 'Arabic',
        
        # Indian languages
        'hi': 'Hindi',
        'bn': 'Bengali',
        'ta': 'Tamil',
        'te': 'Telugu',
        'mr': 'Marathi',
        'gu': 'Gujarati',
        'kn': 'Kannada',
        'ml': 'Malayalam',
        'pa': 'Punjabi',
        'or': 'Odia'
    }
    
    try:
        lang_code = detect(text)
        return lang_map.get(lang_code, 'English')
    except:
        # Fallback for short text
        if any(0x0900 <= ord(c) <= 0x097F for c in text): return 'Hindi'
        if any(0x0C80 <= ord(c) <= 0x0CFF for c in text): return 'Kannada'
        return 'English'

# Multilingual UI setup
st.markdown("""
<style>
    :root {
        --primary: #2E86C1;
        --secondary: #185adb;
        --font: 'Arial Unicode MS', sans-serif;
    }
    .main {
        background-color: #f8f9fa;
        color: #212529;
        font-family: var(--font);
    }
    .chat-message {
        max-width: 80%;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: var(--primary);
        color: white;
        margin-left: auto;
    }
    .bot-message {
        background-color: #e9ecef;
        margin-right: auto;
    }
    .lang-tag {
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_llm():
    return ChatOllama(
        model="mixtral",
        base_url="http://localhost:11434",
        temperature=0.3,
        system="""
        You are a multilingual academic assistant. Follow these rules:
        1. Respond in the same language as the question
        2. Support 12+ languages including major Indian languages
        3. Explain concepts with local examples
        4. Use simple vocabulary
        """
    )

llm = load_llm()

st.title("🌏 Multilingual Study Buddy")
st.markdown("### Ask academic questions in any language")

# Chat history management
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(f'<div class="chat-message {msg["role"]}-message">{msg["content"]}</div>', 
                   unsafe_allow_html=True)
        st.markdown(f'<div class="lang-tag">Language: {msg["language"]}</div>', 
                   unsafe_allow_html=True)

# User input
user_query = st.chat_input("Type your question here...")

PROMPT_TEMPLATE = """{language} question: {question}

Guidelines:
- Respond in {language}
- Use age-appropriate language
- Include real-world examples
- Break complex concepts into steps
- Highlight key terms"""

if user_query:
    # Detect language
    detected_lang = detect_query_language(user_query)
    
    # Update chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query,
        "language": detected_lang
    })
    
    # Generate response
    with st.spinner("📚 Analyzing your question..."):
        try:
            prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            chain = prompt | llm | StrOutputParser()
            
            response = chain.invoke({
                "question": user_query,
                "language": detected_lang
            })
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "language": detected_lang
            })
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Language support sidebar
with st.sidebar:
    st.markdown("### 🌍 Supported Languages")
    st.markdown("""
    - English
    - हिन्दी (Hindi)
    - বাংলা (Bengali)
    - தமிழ் (Tamil)
    - తెలుగు (Telugu)
    - മലയാളം (Malayalam)
    - ಕನ್ನಡ (Kannada)
    - Español (Spanish)
    - Français (French)
    - 中文 (Chinese)
    - العربية (Arabic)
    """)
    
    st.markdown("### ⚙️ System Check")
    if st.button("Test Connection"):
        try:
            test_resp = llm.invoke("Hello")
            st.success(f"✅ Connected (Response time: {test_resp.response_metadata['duration']:.1f}s)")
        except:
            st.error("❌ Connection failed - Start Ollama server")

# Terminal instructions
st.markdown("""
```bash
# Run these commands in separate terminals:
1. ollama serve
2. streamlit run app.py"
"""
)