import streamlit as st
from rag_pipeline import answer_query, retrieve_docs
from vector_database import refresh_vectorstore, process_uploaded_pdf, cleanup_pdf, cleanup_all_resources
import time
import os
import atexit

# ============= Initialize Session State =============
if "messages" not in st.session_state:
    st.session_state.messages = []  # Store chat messages
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False  # Track if PDF is processed
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None  # Store current PDF name
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None  # Store PDF file path

# ============= Cleanup Functions =============
def cleanup_on_exit():
    """Clean up all resources when the app exits"""
    print("🧹 Cleaning up resources on exit...")
    cleanup_all_resources()
    print("✨ Cleanup completed")

# Register cleanup function to run on exit
atexit.register(cleanup_on_exit)

# ============= Page Configuration =============
st.set_page_config(
    page_title="LawAi Sonu",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============= Custom CSS =============
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        max-width: 900px;
        margin: 0 auto;
        padding: 0.3rem;
    }
    
    /* Title styling */
    .stTitle {
        font-size: 1.0rem ;
        margin: 0.3rem 0 !important;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 0.8rem;
        margin: 0.2rem 0;
        border-radius: 8px;
        background-color: #ffffff;
    }
    .stChatMessage[data-testid="stChatMessage"] {
        background-color: #f8f9fa;
    }
    .stChatMessage[data-testid="stChatMessage"] .stMarkdown {
        font-size: 0.95rem;
        line-height: 1.5;
        color: #000000;
    }
    
    /* Chat input styling */
    .stChatInput {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #ffffff;
        padding: 0.5rem;
        box-shadow: 0 -1px 3px rgba(0,0,0,0.1);
        z-index: 1000;
    }
    .stChatInputContainer {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Content spacing */
    .main-content {
        margin-bottom: 25px;
    }
    
    /* Alert boxes */
    .stInfo, .stSuccess, .stWarning {
        padding: 0.5rem !important;
        margin: 0.3rem 0 !important;
        border-radius: 6px !important;
    }
    
    /* Footer */
    .footer-divider {
        margin: 0.3rem 0 !important;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .stApp {
            padding: 0.2rem;
        }
        .stChatMessage {
            padding: 0.6rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============= Main Application =============
with st.container():
    # Title
    st.title("⚖️ LawAi Sonu")
    
    # Status Bar
    status_col1, status_col2 = st.columns([3, 1])
    with status_col1:
        if st.session_state.current_pdf:
            st.info(f"📄 Current PDF: {st.session_state.current_pdf}")
        else:
            st.info("💡 Upload a PDF or ask general legal questions")
    
    with status_col2:
        if st.session_state.pdf_processed:
            st.success("✅ PDF Ready")
        else:
            st.warning("⚠️ No PDF")

    # PDF Upload Section
    uploaded_file = st.file_uploader(
        "📄 Upload PDF (Optional)", 
        type="pdf", 
        help="Upload a PDF to ask specific questions about it"
    )

    # PDF Processing
    if uploaded_file:
        if st.session_state.current_pdf != uploaded_file.name:
            print(f"📥 Processing new PDF: {uploaded_file.name}")
            
            # Cleanup previous PDF if exists
            if st.session_state.pdf_path:
                print("🗑️ Removing previous PDF...")
                cleanup_pdf(st.session_state.pdf_path)
            
            # Process new PDF
            st.session_state.current_pdf = uploaded_file.name
            st.session_state.pdf_processed = False
            st.session_state.messages = []
        
        if not st.session_state.pdf_processed:
            with st.spinner("📚 Processing PDF..."):
                saved_path = process_uploaded_pdf(uploaded_file)
                st.session_state.pdf_path = saved_path
                refresh_vectorstore()
                st.session_state.pdf_processed = True
                st.success("✅ PDF processed successfully!")
                st.info("📝 You can now ask questions about the document")

    # Chat Interface
    st.markdown('<div style="margin-bottom: -25px; margin-top: -10px;">### 💬 Chat</div>', unsafe_allow_html=True)
    
    # Chat History Display
    with st.container():
        st.markdown('<div class="main-content" style="margin-top: -10px; padding-top: 0;">', unsafe_allow_html=True)
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat Input
    st.markdown('<div class="stChatInput">', unsafe_allow_html=True)
    st.markdown('<div class="stChatInputContainer">', unsafe_allow_html=True)
    
    # Handle User Input
    prompt = st.chat_input("⚖️ Ask your question:")
    if prompt:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("⚖️ Analyzing..."):
                try:
                    # Get relevant documents if PDF is processed
                    retrieved_docs = retrieve_docs(prompt) if st.session_state.pdf_processed else []
                    
                    # Get chat history
                    chat_history = "\n".join(
                        [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[:-1]]
                    )
                    
                    # Generate response
                    response = answer_query(
                        query=prompt,
                        chat_history=chat_history,
                        has_document=st.session_state.pdf_processed
                    )

                    # Display response with typing effect
                    response_placeholder = st.empty()
                    displayed_text = ""
                    
                    for char in response:
                        displayed_text += char
                        formatted_text = displayed_text.replace('\n', '\n\n')
                        response_placeholder.markdown(formatted_text)
                        time.sleep(0.003)
                    
                    # Save response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as error:
                    st.error(f"❌ Error: {error}")
    
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Footer
    st.markdown('<div class="footer-divider">---</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        ⚖️ Legal Q&A | AI Powered
    </div>
    """, unsafe_allow_html=True)

# Cleanup on session end
if st.session_state.pdf_path and not st.session_state.current_pdf:
    print("🧹 Cleaning up session resources...")
    cleanup_pdf(st.session_state.pdf_path)
    st.session_state.pdf_path = None
    print("✨ Session cleanup completed")
