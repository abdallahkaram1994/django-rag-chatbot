# Django Docs Q&A

Ask a question about Django 5.2 and get an answer generated from the official documentation, with links to the exact sections it used. If the docs don't cover the question, it says so instead of guessing.

**Live demo:** [https://django-docs-qa.onrender.com](https://django-docs-qa.onrender.com)

## How it works

```
docs (.txt)  ->  chunk  ->  clean + add section  ->  embed  ->  Chroma
                                                                  |
question -> embed -> top 5 chunks -> distance check -> Gemini -> answer + cited sources
```

- **Chunking:** a LangChain `RecursiveCharacterTextSplitter` with custom regex separators that split on RST headings and directives first. Each chunk keeps its heading trail (for example `Customizing authentication > Substituting a custom User model`) and its source file.
- **Embeddings:** `BAAI/bge-small-en-v1.5` through `fastembed` (ONNX, no torch), so it runs in a small container.
- **Retrieval:** Chroma, top 5 by distance. A question whose best match is too far away is refused without calling the LLM.
- **Generation:** Gemini, prompted to answer only from the numbered excerpts, cite them as `[1]`, and say when the excerpts don't cover the question.
- **Serving:** FastAPI (`POST /ask`, `GET /search`, `GET /health`, interactive docs at `/docs`) with the Gradio chat UI mounted at `/`. A per-IP and daily rate limit protects the free Gemini quota.
- **Sources:** merged by section and linked to the section anchor on docs.djangoproject.com.

## Evaluation

`evals/questions.jsonl` holds 35 hand-labelled questions (each with the doc sections that count as a correct hit) and 6 out-of-scope questions.

| Setup | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| bge-small via sentence-transformers (fp32) | 51% | 71% | 80% | 0.614 |
| bge-small via fastembed (quantized ONNX) | _fill in_ | _fill in_ | _fill in_ | _fill in_ |
| + section title in embedded text | _fill in_ | _fill in_ | _fill in_ | _fill in_ |

Run it with `python -m scripts.eval_retrieval [store_folder]`.

Known weakness: questions that don't use the docs' own vocabulary (for example "store data for a visitor between requests" when the docs say "sessions").

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # add GOOGLE_API_KEY and GEMINI_MODEL
python -m scripts.build_index
uvicorn app.main:app --reload
python -m pytest -q
```

## Deploy

Releases deploy themselves: publishing a GitHub Release (for example tag `v1.0.0`) runs the tests, tells Render to deploy that exact commit through a deploy hook, waits until `/version` reports the new commit, and shows the live link on the workflow run.

One-time setup:

1. Create a Render account (sign in with GitHub) and create a **Blueprint** from this repo. It reads `render.yaml` and asks for `GOOGLE_API_KEY` and `GEMINI_MODEL`.
2. In the Render service, open **Settings** and copy the **Deploy Hook** URL.
3. In the GitHub repo, open **Settings > Secrets and variables > Actions**. Add a secret `RENDER_DEPLOY_HOOK_URL` (the hook URL) and a variable `RENDER_URL` (the service's `https://....onrender.com` address).
4. Publish a release: GitHub > Releases > Draft a new release, or `gh release create v1.0.0 --generate-notes`.

The built `chroma/` index is committed, so deploys don't re-embed the docs.

## Limitations

- Each question is answered on its own, with no chat memory.
- The rate limiter is in-memory and per instance.
- The documentation covers Django 5.2 only.

## Credits

The Django documentation is copyright the Django Software Foundation and is used under the BSD 3-Clause license. Django is a registered trademark of the Django Software Foundation; this project is not affiliated with or endorsed by it.
