# Resume-Generator-TechPrep
📄 AI-Powered Resume Builder

Generate ATS-optimized, job-tailored resumes using your profile and job descriptions.

This project is a Streamlit-based web application that helps job seekers generate high-quality, professional resumes using AI. Instead of manually writing JSON, users can fill out simple forms for their profile, education, projects, and experience. By combining this information with an optional job description, the app uses LLMs (OpenAI or Gemini) to generate a polished, tailored resume in Markdown, HTML, and PDF formats.

## 🚀 Features

### ✨ 1. Form-based Profile Input

Users can fill out:

* Contact information

* Summary

* Skills

* Education

* Projects

* Experience

* Certifications

All through a clean, intuitive UI

### 🎯 2. Job Description-Aware Resume Generation

Users can paste any job description (JD).
The AI will:

* Extract required skills & keywords

* Reorder or emphasize relevant experience

* Rewrite bullets using strong action verbs

* Tailor the summary & skills to match the role

* Preserve factual accuracy (no hallucinated jobs)

* This produces a more targeted and competitive resume for each job application.

### 📄 3. Multi-Format Resume Output

The app can export resumes in:

* Markdown

* HTML

* PDF (using a clean, professional template)

Perfect for uploading to job portals or sharing as attachments.


### 🧠 4. Chat-Based Resume Coaching

A built-in AI chat allows users to:

* Improve bullet points

* Clarify project descriptions

* Rewrite experience more professionally

* Get resume tips and feedback

### 🔌 5. Dual LLM Provider Support

You can choose between:

* OpenAI (GPT-4o / GPT-4o-mini / others)

* Google Gemini (gemini-flash-latest / others)

Just provide your API keys in environment variables or Streamlit Secrets.


### 🛠️ Tech Stack

* Frontend/UI: Streamlit

* ackend: Python

* AI Providers: OpenAI, Google Gemini

* PDF Rendering: HTML → PDF via CSS styling

* State Management: Streamlit Session State

### 🔑 Environment Variables

Create a `.streamlit/secrets.toml` file or export them as environment variables:

```toml
OPENAI_API_KEY="your-openai-key"
GOOGLE_API_KEY="your-google-genai-key"
```

### 💡 How It Works

User fills out profile info through interactive forms.

User pastes an optional job description to tailor the resume.

App builds a structured JSON version of the profile.

App sends to LLM with a strong resume-writing prompt.

LLM returns professional resume text.

App renders output in Markdown, HTML, and PDF.

User can download the final resume or refine further via chat.


### 🎯 Project Goals

This project aims to:

* Help applicants create stronger, more competitive resumes

* Automate resume tailoring for each job application

* Simplify editing by removing complex JSON input

* Provide clean, professional resume templates

* Offer AI-powered writing and improvement suggestions

* Empower students and early-career developers during their job search
  

### 📄 License

MIT License.
