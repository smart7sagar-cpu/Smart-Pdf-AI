import io
import json
import re
import time

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

# =====================================================================
# Config
# =====================================================================
MODELS = ["gemini-3.8-flash", "gemini-2.5-flash"]  # 404 aaye to agla try hoga
MAX_CHARS = 2_500_000  # ~6 lakh tokens; isse bada context truncate hoga
LANGUAGES = {
    "Hindi": "Hindi (Devanagari script)",
    "Hinglish": "Hinglish (Hindi written in Roman/English script)",
    "English": "English",
}
SUMMARY_STYLES = {
    "Detailed summary": "a clear, detailed summary in well-organised bullet points with short headings",
    "Short summary": "a short summary of at most 8 bullet points",
    "Exam notes": "concise exam-style revision notes: headings, key definitions, formulas, dates and numbers",
    "Section-wise": "a section-by-section (or chapter-by-chapter) summary following the document's structure",
}
TOOLS = {
    "Key terms (glossary)": "Make a glossary of the 15-25 most important terms/concepts with a one-line explanation each.",
    "FAQ": "Write 8-12 frequently asked questions a reader may have about this material, each with a short answer.",
    "Important numbers & dates": "List every important number, date, amount, deadline and name in a table with what it refers to.",
    "Action items": "List all action items, instructions, rules or to-dos mentioned, in order of importance.",
}

st.set_page_config(page_title="AI PDF Assistant", page_icon="📄", layout="wide")

# =====================================================================
# Design
# =====================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Hind:wght@400;500;600&family=Mukta:wght@600;700;800&display=swap');

.stApp { font-family: 'Hind', 'Noto Sans Devanagari', sans-serif; }
h1, h2, h3, h4 { font-family: 'Mukta', 'Hind', sans-serif !important; letter-spacing: -0.01em; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1180px; }

/* Hero */
.hero {
  background: #1B2A41; color: #F5F6F8;
  border-radius: 18px; padding: 34px 38px; margin-bottom: 22px;
  border-left: 8px solid #F0A030;
}
.hero h1 { color: #FFFFFF; font-size: 2.3rem; margin: 0 0 6px 0; padding: 0; }
.hero p { color: #C9D3E0; font-size: 1.05rem; margin: 0; max-width: 640px; }
.hero .by { margin-top: 14px; font-size: .9rem; color: #F0A030; }

/* Metrics */
[data-testid="stMetric"] {
  background: #FFFFFF; border: 1px solid #E1E5EB; border-radius: 14px;
  padding: 14px 18px;
}
[data-testid="stMetricLabel"] p { color: #5B6B80; font-size: .85rem; }
[data-testid="stMetricValue"] { font-family: 'Mukta', sans-serif; font-weight: 700; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #E1E5EB; }
.stTabs [data-baseweb="tab"] { font-weight: 600; padding: 10px 18px; }
.stTabs [aria-selected="true"] { color: #1F4E79; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
  border-radius: 10px; font-weight: 600; border: 1px solid #1F4E79; padding: .5rem 1.1rem;
}
.stButton > button[kind="primary"] { background: #1F4E79; color: #fff; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: #F0A030; color: #1B2A41; }

/* Chat */
[data-testid="stChatMessage"] {
  background: #FFFFFF; border: 1px solid #E1E5EB; border-radius: 14px; padding: 14px 18px;
}

/* Result panel */
.panel { background: #FFFFFF; border: 1px solid #E1E5EB; border-radius: 14px; padding: 20px 24px; }

/* Sidebar */
[data-testid="stSidebar"] { border-right: 1px solid #E1E5EB; }

/* Empty state */
.empty { background: #FFFFFF; border: 1px dashed #B8C2D0; border-radius: 14px; padding: 22px 26px; }
.empty h4 { margin: 0 0 6px 0; }
.empty p { margin: 0; color: #5B6B80; }
.footer { text-align: center; color: #8693A5; font-size: .85rem; margin-top: 40px; }

@media (max-width: 640px) {
  .hero { padding: 24px 22px; }
  .hero h1 { font-size: 1.7rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================================
# API client (With Multi-Key Rotation)
# =====================================================================
import random

if "GEMINI_KEYS" in st.secrets:
    # Saari keys me se har baar koi ek random key uthayega
    selected_key = random.choice(st.secrets["GEMINI_KEYS"])
    try:
        client = genai.Client(api_key=selected_key)
    except Exception as e:
        st.error(f"Client initialize nahi ho paya: {e}")
        st.stop()
else:
    st.error(
        "GEMINI_KEYS nahi mili. Streamlit → Settings → Secrets me list add karein:\n\n"
        'GEMINI_KEYS = ["key1", "key2", "key3"]'
    )
    st.stop()



# =====================================================================
# Helpers
# =====================================================================
def is_not_found(e):
    m = str(e)
    return "404" in m or "NOT_FOUND" in m


def is_retryable(e):
    m = str(e)
    return any(c in m for c in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))


def friendly_error(e):
    m = str(e)
    if "API key" in m or "API_KEY" in m or "PERMISSION_DENIED" in m:
        return "API key galat hai ya is model ki permission nahi hai. Secrets me key check karein."
    if "429" in m or "RESOURCE_EXHAUSTED" in m:
        return "Gemini ki limit/quota khatam ho gayi. Thodi der baad try karein."
    if "503" in m or "UNAVAILABLE" in m:
        return "Gemini abhi busy hai. Kuch der baad dobara try karein."
    return f"Error: {m}"


def model_order():
    first = st.session_state.get("model", MODELS[0])
    return [first] + [m for m in MODELS if m != first]


def make_config(system=None, temperature=0.2, json_mode=False):
    kwargs = dict(system_instruction=system, temperature=temperature)
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    return types.GenerateContentConfig(**kwargs)


def generate(prompt, system=None, temperature=0.2, json_mode=False):
    """Non-streaming call. Retry + model fallback. Kabhi crash nahi karta."""
    last = None
    for model in model_order():
        for attempt in range(3):
            try:
                r = client.models.generate_content(
                    model=model, contents=prompt,
                    config=make_config(system, temperature, json_mode),
                )
                return r.text or ""
            except Exception as e:
                last = e
                if is_retryable(e) and attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                break
        if not is_not_found(last):
            break
    raise RuntimeError(friendly_error(last))


def stream_answer(prompt, system=None, temperature=0.2):
    """Streaming generator (st.write_stream ke liye)."""
    last = None
    for model in model_order():
        started = False
        try:
            for chunk in client.models.generate_content_stream(
                model=model, contents=prompt, config=make_config(system, temperature)
            ):
                if chunk.text:
                    started = True
                    yield chunk.text
            return
        except Exception as e:
            last = e
            if not started and is_not_found(e):
                continue
            break
    yield f"\n\n❌ {friendly_error(last)}"


@st.cache_data(show_spinner=False)
def extract_pages(files_data):
    """{filename: {"pages": [text,...], "error": str|None}} — cached."""
    out = {}
    for name, data in files_data:
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    pass
            pages = []
            for p in reader.pages:
                try:
                    pages.append((p.extract_text() or "").strip())
                except Exception:
                    pages.append("")
            out[name] = {"pages": pages, "error": None}
        except Exception as e:
            out[name] = {"pages": [], "error": str(e)}
    return out


@st.cache_data(show_spinner=False)
def ocr_pdf(name, data, model):
    """Scanned PDF ko Gemini se padhwata hai. Page-wise list return karta hai."""
    prompt = (
        "Transcribe ALL text in this PDF exactly as written (Hindi/English/any language). "
        "Before each page's text write a line exactly like: ===PAGE 1===, ===PAGE 2===, and so on. "
        "Do not add commentary."
    )
    r = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=data, mime_type="application/pdf"), prompt],
        config=types.GenerateContentConfig(temperature=0),
    )
    parts = re.split(r"===PAGE\s+\d+===", r.text or "")
    pages = [p.strip() for p in parts[1:]] if len(parts) > 1 else [(r.text or "").strip()]
    return pages


def build_context(docs, selected):
    chunks = []
    for name in selected:
        for i, text in enumerate(docs[name], 1):
            if text:
                chunks.append(f"\n--- [Document: {name} | Page: {i}] ---\n{text}")
    full = "".join(chunks)
    return full[:MAX_CHARS], len(full) > MAX_CHARS


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


# =====================================================================
# Session defaults
# =====================================================================
for key, default in {
    "model": MODELS[0], "messages": [], "summary": None, "tool_out": None,
    "quiz": None, "quiz_checked": False, "sig": None,
}.items():
    st.session_state.setdefault(key, default)

# =====================================================================
# Sidebar (upload + settings)
# =====================================================================
with st.sidebar:
    st.markdown("### 📄 Documents")
    uploaded_files = st.file_uploader(
        "PDF files upload karein", type=["pdf"], accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.markdown("### ⚙️ Settings")
    language = st.selectbox("Jawab ki bhasha", list(LANGUAGES), index=0)
    st.selectbox("AI model", MODELS, key="model")
    temperature = st.slider("Creativity", 0.0, 1.0, 0.2, 0.1,
                            help="Kam value = zyada exact jawab. Zyada = zyada free-flow.")
    use_ocr = st.toggle("Scanned PDF ke liye OCR", value=True,
                        help="Jin PDFs me text select nahi hota, unhe Gemini se padha jayega.")
    if st.button("🗑️ Sab kuch reset karein", use_container_width=True):
        for k in ("messages", "summary", "tool_out", "quiz", "quiz_checked"):
            st.session_state[k] = [] if k == "messages" else (False if k == "quiz_checked" else None)
        st.rerun()

# =====================================================================
# Hero
# =====================================================================
st.markdown(
    """
<div class="hero">
  <h1>AI PDF Assistant</h1>
  <p>Apni PDF upload karein. Summary banwayein, sawal poochein, quiz banayein —
  har jawab document aur page number ke saath.</p>
  <div class="by">Made by SAGAR</div>
</div>
""",
    unsafe_allow_html=True,
)

if not uploaded_files:
    c1, c2, c3 = st.columns(3)
    for col, title, text in [
        (c1, "Summary", "Short, detailed, exam notes ya section-wise — apni pasand ka style chunein."),
        (c2, "Sawal-jawab", "PDF se chat karein. Jawab me document ka naam aur page number aata hai."),
        (c3, "Quiz aur study tools", "MCQ quiz, glossary, FAQ aur important dates ek click me."),
    ]:
        col.markdown(f'<div class="empty"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)
    st.info("Shuru karne ke liye left sidebar me ek ya zyada PDF upload karein.")
    st.markdown('<div class="footer">AI PDF Assistant · Made by SAGAR</div>', unsafe_allow_html=True)
    st.stop()

# =====================================================================
# Load documents
# =====================================================================
files_data = tuple((f.name, f.getvalue()) for f in uploaded_files)
raw = extract_pages(files_data)

docs, notes = {}, []
for name, data in files_data:
    info = raw[name]
    if info["error"]:
        notes.append(f"❌ **{name}** padh nahi paya: {info['error']}")
        continue
    pages = list(info["pages"])
    if not any(pages) and use_ocr:
        try:
            with st.spinner(f"{name}: scanned PDF hai, OCR chal raha hai..."):
                pages = ocr_pdf(name, data, st.session_state.model)
        except Exception as e:
            notes.append(f"⚠️ **{name}** ka OCR fail hua: {friendly_error(e)}")
    if any(pages):
        docs[name] = pages
    else:
        notes.append(f"⚠️ **{name}** me text nahi mila.")

for n in notes:
    st.warning(n)
if not docs:
    st.stop()

with st.sidebar:
    selected = st.multiselect("Kin documents par kaam karein", list(docs), default=list(docs))
if not selected:
    st.info("Sidebar me kam se kam ek document chunein.")
    st.stop()

context, truncated = build_context(docs, selected)

# Files ya selection badle to purana output hata do
sig = (tuple((n, len(d)) for n, d in files_data), tuple(selected))
if st.session_state.sig != sig:
    st.session_state.update(sig=sig, messages=[], summary=None, tool_out=None,
                            quiz=None, quiz_checked=False)

# =====================================================================
# Stats
# =====================================================================
n_pages = sum(len(docs[n]) for n in selected)
n_words = len(context.split())
m1, m2, m3, m4 = st.columns(4)
m1.metric("Documents", len(selected))
m2.metric("Pages", f"{n_pages:,}")
m3.metric("Words", f"{n_words:,}")
m4.metric("Approx tokens", f"{len(context) // 4:,}")
if truncated:
    st.info(f"Document bahut bada hai, isliye sirf shuru ke {MAX_CHARS:,} characters use ho rahe hain.")

lang = LANGUAGES[language]
SYSTEM = (
    "You are a precise document assistant. Answer ONLY from the provided context. "
    f"Write the answer in {lang}. Be clear and well structured. "
    "Always cite sources inline like (Document: <name>, Page: <number>). "
    "If the answer is not in the context, say clearly that it was not found in the PDF."
)

tab_sum, tab_chat, tab_quiz, tab_tools = st.tabs(["📝 Summary", "💬 Chat", "🧠 Quiz", "🔑 Study tools"])

# ---------------------------------------------------------------- Summary
with tab_sum:
    style = st.selectbox("Summary ka style", list(SUMMARY_STYLES))
    if st.button("Summary banao", type="primary"):
        with st.spinner("Summary ban rahi hai..."):
            try:
                st.session_state.summary = generate(
                    f"Write {SUMMARY_STYLES[style]} of the document(s) below. "
                    f"Mention page numbers where useful.\n\n{context}",
                    system=SYSTEM, temperature=temperature,
                )
            except Exception as e:
                st.error(str(e))
    if st.session_state.summary:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(st.session_state.summary)
        st.markdown("</div>", unsafe_allow_html=True)
        st.download_button("⬇️ Summary download karein (.md)", st.session_state.summary,
                           file_name="summary.md", mime="text/markdown")
    else:
        st.markdown('<div class="empty"><h4>Abhi koi summary nahi hai</h4>'
                    "<p>Upar style chunein aur “Summary banao” dabayein.</p></div>",
                    unsafe_allow_html=True)

# ---------------------------------------------------------------- Chat
with tab_chat:
    if not st.session_state.messages:
        st.markdown('<div class="empty"><h4>PDF se kuch bhi poochein</h4>'
                    "<p>Neeche sawal likhein ya inme se ek chunein.</p></div>",
                    unsafe_allow_html=True)
        q1, q2, q3 = st.columns(3)
        for col, text in [
            (q1, "Is document ke main points kya hain?"),
            (q2, "Important dates aur numbers batao."),
            (q3, "Conclusion ya final result kya hai?"),
        ]:
            if col.button(text, use_container_width=True):
                st.session_state.pending = text
                st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    typed = st.chat_input("Apna sawal likhein...")
    question = st.session_state.pop("pending", None) or typed

    if question:
        history = st.session_state.messages[-8:]
        convo = "\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in history)
        prompt = (
            f"Context:\n{context}\n\nPrevious conversation:\n{convo or '(none)'}\n\n"
            f"Question: {question}"
        )
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            answer = st.write_stream(stream_answer(prompt, SYSTEM, temperature))
        st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.session_state.messages:
        transcript = "\n\n".join(
            f"{'Sawal' if m['role'] == 'user' else 'Jawab'}: {m['content']}"
            for m in st.session_state.messages
        )
        cA, cB = st.columns([1, 1])
        cA.download_button("⬇️ Chat download karein", transcript, file_name="chat.txt")
        if cB.button("Chat saaf karein"):
            st.session_state.messages = []
            st.rerun()

# ---------------------------------------------------------------- Quiz
with tab_quiz:
    n_q = st.slider("Kitne sawal?", 3, 10, 5)
    if st.button("Quiz banao", type="primary"):
        with st.spinner("Quiz ban raha hai..."):
            try:
                raw_quiz = generate(
                    f"Create {n_q} multiple-choice questions from the document(s) below, in {lang}. "
                    "Return ONLY a JSON array. Each item has keys: question (string), "
                    "options (array of exactly 4 strings), answer_index (integer 0-3), "
                    "explanation (short string), source (document name and page).\n\n" + context,
                    temperature=0.4, json_mode=True,
                )
                st.session_state.quiz = parse_json(raw_quiz)
                st.session_state.quiz_checked = False
                for k in [k for k in st.session_state if str(k).startswith("quiz_q")]:
                    del st.session_state[k]
            except Exception as e:
                st.error(f"Quiz nahi ban paya: {e}")

    quiz = st.session_state.quiz
    if quiz:
        for i, q in enumerate(quiz):
            st.markdown(f"**{i + 1}. {q['question']}**")
            choice = st.radio("Option", q["options"], index=None, key=f"quiz_q{i}",
                              label_visibility="collapsed")
            if st.session_state.quiz_checked:
                right = q["options"][q["answer_index"]]
                if choice == right:
                    st.success("Sahi ✅")
                elif choice is None:
                    st.info(f"Jawab nahi diya. Sahi jawab: {right}")
                else:
                    st.error(f"Galat ❌ — sahi jawab: {right}")
                st.caption(f"{q.get('explanation', '')}  ({q.get('source', '')})")
        if not st.session_state.quiz_checked:
            if st.button("Jawab check karein"):
                st.session_state.quiz_checked = True
                st.rerun()
        else:
            score = sum(
                1 for i, q in enumerate(quiz)
                if st.session_state.get(f"quiz_q{i}") == q["options"][q["answer_index"]]
            )
            st.metric("Aapka score", f"{score} / {len(quiz)}")
    else:
        st.markdown('<div class="empty"><h4>Khud ko test karein</h4>'
                    "<p>Sawal ginti chunein aur “Quiz banao” dabayein.</p></div>",
                    unsafe_allow_html=True)

# ---------------------------------------------------------------- Tools
with tab_tools:
    tool = st.radio("Kya chahiye?", list(TOOLS), horizontal=True)
    if st.button("Banao", type="primary", key="tool_btn"):
        with st.spinner(f"{tool} ban raha hai..."):
            try:
                st.session_state.tool_out = (tool, generate(
                    f"{TOOLS[tool]}\n\n{context}", system=SYSTEM, temperature=temperature))
            except Exception as e:
                st.error(str(e))
    if st.session_state.tool_out:
        t_name, t_text = st.session_state.tool_out
        st.markdown(f"#### {t_name}")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(t_text)
        st.markdown("</div>", unsafe_allow_html=True)
        st.download_button("⬇️ Download (.md)", t_text, file_name="output.md", mime="text/markdown")

st.markdown('<div class="footer">AI PDF Assistant · Made by SAGAR</div>', unsafe_allow_html=True)
