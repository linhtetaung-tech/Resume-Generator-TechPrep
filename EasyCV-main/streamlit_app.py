"""
Resume Builder & Chat Coach — Streamlit (single file)

How to deploy on Streamlit Cloud:
1) Create a new app and point it to this file as `streamlit_app.py`.
2) In the same repo, add a `requirements.txt` with at least:
   streamlit
   openai>=1.3.0
   google-generativeai>=0.8.0
   markdown
3) In Streamlit Cloud, set Secrets:
   - OPENAI_API_KEY: your OpenAI key (if using OpenAI)
   - GOOGLE_API_KEY: your Google Generative AI key (if using Gemini)

Local run:
   pip install -r requirements.txt
   streamlit run streamlit_app.py
"""

import os
import json
from textwrap import dedent

import streamlit as st

# Optional dependency for HTML export
try:
    import markdown as md_lib
except Exception:
    md_lib = None

# ---------- App Config ----------
st.set_page_config(
    page_title="Resume Builder & Chat Coach",
    page_icon="📄",
    layout="wide",
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {role, content}
if "profile" not in st.session_state:
    st.session_state.profile = {
        "name": "",
        "email": "",
        "phone": "",
        "location": "",
        "links": [],  # list of strings
        "summary": "",
        "education": [],  # list of {school, degree, start, end, details}
        "skills": [],     # list of strings
        "experiences": [],# list of {title, org, start, end, bullets}
        "projects": [],   # list of {name, url, start, end, bullets}
        "certs": []       # list of strings
    }
if "generated_md" not in st.session_state:
    st.session_state.generated_md = ""
if "generated_html" not in st.session_state:
    st.session_state.generated_html = ""

# ---------- Providers ----------
PROVIDER_OPENAI = "OpenAI"
PROVIDER_GEMINI = "Gemini"

@st.cache_resource(show_spinner=False)
def get_openai_client():
    try:
        from openai import OpenAI
        # Try environment variable first, then Streamlit secrets
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets.get("OPENAI_API_KEY", "")
            except (AttributeError, KeyError, FileNotFoundError):
                api_key = ""
        if not api_key or api_key.strip() == "":
            return None
        return OpenAI(api_key=api_key)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def get_gemini_client():
    try:
        import google.generativeai as genai
        # Try environment variable first, then Streamlit secrets
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets.get("GOOGLE_API_KEY", "")
            except (AttributeError, KeyError, FileNotFoundError):
                api_key = ""
        if not api_key or api_key.strip() == "":
            return None
        genai.configure(api_key=api_key)
        return genai
    except Exception:
        return None

# Simple wrapper to call chosen model

def call_llm(provider, model, system_prompt, user_prompt, history=None):
    history = history or []
    if provider == PROVIDER_OPENAI:
        client = get_openai_client()
        if client is None:
            raise RuntimeError("OpenAI client not available. Please add OPENAI_API_KEY to Streamlit Cloud secrets (Manage app > Settings > Secrets).")
        # Convert to Chat Completions
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for m in history:
            if m["role"] in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": user_prompt})
        try:
            resp = client.chat.completions.create(model=model, messages=messages, temperature=0.7)
            return resp.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "api key" in error_msg.lower() or "invalid" in error_msg.lower():
                raise RuntimeError("Authentication failed. Please check that your OPENAI_API_KEY in Streamlit Cloud secrets is valid and not expired.")
            raise

    elif provider == PROVIDER_GEMINI:
        genai = get_gemini_client()
        if genai is None:
            raise RuntimeError("Gemini client not available. Please add GOOGLE_API_KEY to Streamlit Cloud secrets (Manage app > Settings > Secrets).")
        # Build a single prompt (Gemini can accept system in content)
        try:
            model_obj = genai.GenerativeModel(model)
            # Convert history into parts
            conv = []
            if system_prompt:
                conv.append({"role": "user", "parts": [f"SYSTEM:\n{system_prompt}"]})
            for m in history:
                role = "user" if m["role"] == "user" else "model"
                conv.append({"role": role, "parts": [m["content"]]})
            conv.append({"role": "user", "parts": [user_prompt]})
            chat = model_obj.start_chat(history=conv)
            resp = chat.send_message("Please respond to the last user message only.")
            return resp.text
        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "api key" in error_msg.lower() or "invalid" in error_msg.lower():
                raise RuntimeError("Authentication failed. Please check that your GOOGLE_API_KEY in Streamlit Cloud secrets is valid and not expired.")
            raise

    else:
        raise ValueError("Unknown provider")

# ---------- Prompts ----------

def build_system_prompt():
    return dedent(
        f"""
        You are a concise, ATS-savvy resume writer and coach. Optimize bullet points using strong, specific verbs,
        quantify impact where possible, and follow modern ATS conventions (no tables, no images). Prefer the STAR
        pattern for achievements. Keep content truthful: only rewrite using facts provided by the user profile and chat.
        Keep sentences concise. Output plain Markdown unless specifically asked for HTML.
        """
    ).strip()

def build_resume_instruction(profile_json, style, sections):
    return dedent(
        f"""
        Create a {style} software/tech resume in **Markdown** using ONLY this JSON as the source of truth:
        ```json
        {profile_json}
        ```
        Sections to include (if non-empty): {", ".join(sections)}.
        Rules:
        - Use a clear Markdown structure with headers (e.g., `# Name`, `## Education`).
        - Name and contact on one line at the top.
        - Experience & projects: use 3-5 bullets each; begin each bullet with a strong verb; quantify when possible.
        - Keep it to one page worth of text (concise). No images, no tables.
        - Do NOT invent content. If data is missing, omit that bullet/section.
        Return only the Markdown.
        """
    ).strip()

# ---------- UI: Sidebar ----------
with st.sidebar:
    st.header("⚙️ Model Settings")
    provider = st.radio("Provider", [PROVIDER_OPENAI, PROVIDER_GEMINI], index=0)
    if provider == PROVIDER_OPENAI:
        model = st.text_input("OpenAI Model", value="gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets.get("OPENAI_API_KEY", "")
            except (AttributeError, KeyError, FileNotFoundError):
                api_key = ""
        key_ok = bool(api_key and api_key.strip())
        st.caption("🔑 OPENAI_API_KEY " + ("✅ found" if key_ok else "❌ missing"))
        if not key_ok:
            st.warning("⚠️ Please add OPENAI_API_KEY to Streamlit Cloud secrets (Manage app > Settings > Secrets) or environment variables.")
    else:
        model = st.text_input("Gemini Model", value="gemini-flash-latest")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets.get("GOOGLE_API_KEY", "")
            except (AttributeError, KeyError, FileNotFoundError):
                api_key = ""
        key_ok = bool(api_key and api_key.strip())
        st.caption("🔑 GOOGLE_API_KEY " + ("✅ found" if key_ok else "❌ missing"))
        if not key_ok:
            st.warning("⚠️ Please add GOOGLE_API_KEY to Streamlit Cloud secrets (Manage app > Settings > Secrets) or environment variables.")

    st.divider()
    st.header("🧾 Export")
    export_fmt = st.radio("Format", ["Markdown", "HTML"], index=0)
    st.caption("You can always download both after generation.")

# ---------- UI: Main Tabs ----------

st.title("📄 Resume Builder & Chat Coach")
st.write("Describe what you've done. I'll help turn it into a clean, ATS-friendly resume.")

profile_tab, chat_tab, generate_tab, preview_tab = st.tabs([
    "Profile Form",
    "Chat Coach",
    "Generate Resume",
    "Preview & Download",
])

# ---------- Profile Form ----------
with profile_tab:
    prof = st.session_state.profile

    # ---------- Contact ----------
    st.subheader("Contact")
    c1, c2 = st.columns(2)
    with c1:
        prof["name"] = st.text_input("Full Name", value=prof.get("name", ""))
        prof["email"] = st.text_input("Email", value=prof.get("email", ""))
        prof["phone"] = st.text_input("Phone", value=prof.get("phone", ""))
    with c2:
        prof["location"] = st.text_input("Location (City, State)", value=prof.get("location", ""))
        links_text = st.text_area("Links (one per line)", value="\n".join(prof.get("links", [])))
        prof["links"] = [l.strip() for l in links_text.splitlines() if l.strip()]

    # ---------- Summary ----------
    st.subheader("Summary (optional)")
    prof["summary"] = st.text_area("1–3 sentences", value=prof.get("summary", ""))

    # ---------- Skills ----------
    st.subheader("Skills (comma-separated)")
    skills_text = st.text_input(
        "e.g., Python, React, SQL, AWS",
        value=", ".join(prof.get("skills", [])),
    )
    prof["skills"] = [s.strip() for s in skills_text.split(",") if s.strip()]

    # ---------- Education ----------
    st.subheader("Education")
    edu_list = prof.get("education", [])
    num_edu = st.number_input(
        "Number of education entries",
        min_value=0,
        max_value=10,
        value=len(edu_list),
        step=1,
        key="num_edu",
    )

    # adjust the list length
    while len(edu_list) < num_edu:
        edu_list.append({
            "school": "",
            "degree": "",
            "start": "",
            "end": "",
            "details": "",
        })
    while len(edu_list) > num_edu:
        edu_list.pop()

    for i, edu in enumerate(edu_list):
        with st.expander(f"Education #{i+1}", expanded=True):
            edu["school"] = st.text_input(
                "School",
                value=edu.get("school", ""),
                key=f"edu_school_{i}",
            )
            edu["degree"] = st.text_input(
                "Degree (e.g. B.A. in Computer Science)",
                value=edu.get("degree", ""),
                key=f"edu_degree_{i}",
            )
            cols = st.columns(2)
            with cols[0]:
                edu["start"] = st.text_input(
                    "Start (e.g. 2023)",
                    value=edu.get("start", ""),
                    key=f"edu_start_{i}",
                )
            with cols[1]:
                edu["end"] = st.text_input(
                    "End (or 'Present')",
                    value=edu.get("end", ""),
                    key=f"edu_end_{i}",
                )
            edu["details"] = st.text_area(
                "Details (GPA, honors, relevant coursework…)",
                value=edu.get("details", ""),
                key=f"edu_details_{i}",
            )
    prof["education"] = edu_list

    # ---------- Projects ----------
    st.subheader("Projects")
    proj_list = prof.get("projects", [])
    num_proj = st.number_input(
        "Number of projects",
        min_value=0,
        max_value=10,
        value=len(proj_list),
        step=1,
        key="num_proj",
    )

    while len(proj_list) < num_proj:
        proj_list.append({
            "name": "",
            "url": "",
            "start": "",
            "end": "",
            "bullets": [],
        })
    while len(proj_list) > num_proj:
        proj_list.pop()

    for i, proj in enumerate(proj_list):
        with st.expander(f"Project #{i+1}", expanded=True):
            proj["name"] = st.text_input(
                "Project name",
                value=proj.get("name", ""),
                key=f"proj_name_{i}",
            )
            proj["url"] = st.text_input(
                "URL (optional)",
                value=proj.get("url", ""),
                key=f"proj_url_{i}",
            )
            cols = st.columns(2)
            with cols[0]:
                proj["start"] = st.text_input(
                    "Start",
                    value=proj.get("start", ""),
                    key=f"proj_start_{i}",
                )
            with cols[1]:
                proj["end"] = st.text_input(
                    "End",
                    value=proj.get("end", ""),
                    key=f"proj_end_{i}",
                )
            bullets_text = st.text_area(
                "Bullets (one per line；write down your notes，AI will edit for you)",
                value="\n".join(proj.get("bullets", [])),
                key=f"proj_bullets_{i}",
            )
            proj["bullets"] = [b.strip() for b in bullets_text.splitlines() if b.strip()]
    prof["projects"] = proj_list

    # ---------- Experience ----------
    st.subheader("Experience")
    exp_list = prof.get("experiences", [])
    num_exp = st.number_input(
        "Number of experience entries",
        min_value=0,
        max_value=10,
        value=len(exp_list),
        step=1,
        key="num_exp",
    )

    while len(exp_list) < num_exp:
        exp_list.append({
            "title": "",
            "org": "",
            "start": "",
            "end": "",
            "bullets": [],
        })
    while len(exp_list) > num_exp:
        exp_list.pop()

    for i, exp in enumerate(exp_list):
        with st.expander(f"Experience #{i+1}", expanded=True):
            exp["title"] = st.text_input(
                "Title",
                value=exp.get("title", ""),
                key=f"exp_title_{i}",
            )
            exp["org"] = st.text_input(
                "Company / Organization",
                value=exp.get("org", ""),
                key=f"exp_org_{i}",
            )
            cols = st.columns(2)
            with cols[0]:
                exp["start"] = st.text_input(
                    "Start",
                    value=exp.get("start", ""),
                    key=f"exp_start_{i}",
                )
            with cols[1]:
                exp["end"] = st.text_input(
                    "End",
                    value=exp.get("end", ""),
                    key=f"exp_end_{i}",
                )
            bullets_text = st.text_area(
                "Bullets (one per line；AI will help you to make it as the professional bullets)",
                value="\n".join(exp.get("bullets", [])),
                key=f"exp_bullets_{i}",
            )
            exp["bullets"] = [b.strip() for b in bullets_text.splitlines() if b.strip()]
    prof["experiences"] = exp_list

    # ---------- Certifications ----------
    st.subheader("Certifications (one per line)")
    certs_text = st.text_area(
        "",
        value="\n".join(prof.get("certs", [])),
        key="certs_text",
    )
    prof["certs"] = [c.strip() for c in certs_text.splitlines() if c.strip()]

    st.success("Profile saved in session (not uploaded). Proceed to Chat or Generate.")


# ---------- Chat Coach ----------
with chat_tab:
    st.caption("Ask questions like: ‘How do I quantify my tutoring project?’ or paste rough bullets for improvement.")
    # Display history
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_msg = st.chat_input("Type your question or paste rough bullets…")
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    reply = call_llm(
                        provider,
                        model,
                        build_system_prompt(),
                        user_msg,
                        history=st.session_state.chat_history[:-1],
                    )
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.markdown(reply)
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"❌ Error: {error_msg}")
                    # Remove the user message from history if the call failed
                    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                        st.session_state.chat_history.pop()

# ---------- Generate Resume ----------
with generate_tab:
    st.subheader("Generate from your profile + chat context")
    style = st.selectbox("Style", ["standard ATS", "early-career student", "data/ML focus", "frontend focus"], index=0)

    default_sections = [
        "Contact",
        "Summary",
        "Skills",
        "Education",
        "Experience",
        "Projects",
        "Certifications"
    ]
    sections = st.multiselect("Include sections", default_sections, default=default_sections)

    if st.button("🚀 Generate Resume (Markdown)", type="primary"):
        with st.spinner("Generating resume…"):
            profile_json = json.dumps(st.session_state.profile, ensure_ascii=False)
            instruction = build_resume_instruction(profile_json, style, sections)
            try:
                md_out = call_llm(
                    provider,
                    model,
                    build_system_prompt(),
                    instruction,
                    history=st.session_state.chat_history,
                )
                st.session_state.generated_md = md_out.strip()
                # Build HTML either via markdown lib or minimal wrapper
                if md_lib is not None:
                    html_body = md_lib.markdown(st.session_state.generated_md, extensions=["tables", "fenced_code"])
                else:
                    # Minimal fallback: wrap in <pre>
                    html_body = f"<pre>{st.session_state.generated_md}</pre>"
                st.session_state.generated_html = dedent(f"""
                <!doctype html>
                <html lang="en">
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <title>Resume</title>
                <style>
                  body {{ max-width: 820px; margin: 2rem auto; font: 14px/1.5 -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial; color: #111; }}
                  h1,h2,h3 {{ margin-top: 1.2rem; }}
                  ul {{ margin: .4rem 0 .8rem 1.2rem; }}
                </style>
                <body>{html_body}</body>
                </html>
                """)
                st.success("Done! Check the Preview & Download tab.")
            except Exception as e:
                st.error(f"Generation error: {e}")

# ---------- Preview & Download ----------
with preview_tab:
    st.subheader("Preview")
    left, right = st.columns(2)
    with left:
        st.markdown("**Markdown**")
        st.code(st.session_state.generated_md or "(Nothing yet — generate first.)", language="markdown")
        st.download_button(
            "Download .md",
            data=st.session_state.generated_md,
            file_name="resume.md",
            mime="text/markdown",
            disabled=not bool(st.session_state.generated_md),
        )
    with right:
        st.markdown("**HTML** (auto-converted)")
        if st.session_state.generated_html:
            st.components.v1.html(st.session_state.generated_html, height=600, scrolling=True)
        st.download_button(
            "Download .html",
            data=st.session_state.generated_html,
            file_name="resume.html",
            mime="text/html",
            disabled=not bool(st.session_state.generated_html),
        )

st.divider()
st.caption("Privacy: nothing is stored server-side by this app beyond Streamlit session state. Add your own persistence if needed.")
