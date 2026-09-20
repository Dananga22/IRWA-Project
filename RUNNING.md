# Running MedExplain AI (Windows)

All commands are run from the project root: `D:\Y3S2\IRWA\IRWA-Project`

---

## 1. Create and activate a virtual environment

**PowerShell**

```powershell
cd D:\Y3S2\IRWA\IRWA-Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, either run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

or use Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

You know it worked when the prompt is prefixed with `(.venv)`.

---

## 2. Install dependencies

**Start here — fast, about 50 MB:**

```powershell
pip install -r requirements-core.txt
```

This runs everything: all five agents, LangGraph orchestration, the MCP protocol layer, the safety gate and the web app. Retrieval uses the built-in BM25 backend.

**Optional — dense vector retrieval:**

```powershell
pip install -r requirements.txt
```

This adds ChromaDB and sentence-transformers, which pull PyTorch — roughly a 2 GB download, and the embedding model downloads on first use. Only do this on a good connection, and never on the day of the evaluation. If it fails or the model cannot download, the system logs the reason and falls back to BM25 automatically.

---

## 3. Configure

```powershell
copy .env.example .env
notepad .env
```

Set these three lines:

```
GEMINI_API_KEY=<your key>
GEMINI_MODEL=gemini-3.6-flash
GEMINI_THINKING_LEVEL=LOW
```

Then verify the key, the model and the SDK version in one command:

```powershell
python scripts/test_gemini.py
```

It checks the SDK, the key, the model availability and a real generation call,
and names whichever step fails. Run it before every demo. To see every model the
key can reach:

```powershell
python scripts/test_gemini.py --list
```

**Note on `gemini-3.6-flash`:** Gemini 3.x removed `temperature`, `top_p` and
`top_k` and replaced them with `thinking_level`. The explanation agent handles
both model families automatically, but it needs `google-genai>=2.0.0`. If you see
a warning about `thinking_level`, run `pip install -U google-genai`.

**You can skip this.** With no key the system uses a grounded fallback generator that builds explanations directly from retrieved passages — still cited, still safety-checked. It is slightly less fluent than Gemini but it cannot fail on a quota or a dead connection.

Free key: https://aistudio.google.com/app/apikey

---

## 4. Build the knowledge base index

```powershell
python scripts/build_index.py
```

Expected output:

```
Corpus contains 41 passages.
Indexed 41 passages using backend: bm25-inmemory
Sources represented: CDC, MedlinePlus, WHO
```

Run this once after install, and again any time you edit `knowledge_base/corpus.json`.

Test a query:

```powershell
python scripts/build_index.py --query "HbA1c"
```

---

## 5. Run it

### The pipeline, with the full agent trace — use this for the demo

```powershell
python scripts/run_pipeline.py sample_reports/sample_cbc_report.pdf
```

You will see each agent hop, the MCP tool calls, the retrieved sources, the safety verdict, and the final explanation. Also try:

```powershell
python scripts/run_pipeline.py sample_reports/sample_lipid_panel.pdf
python scripts/run_pipeline.py sample_reports/sample_metabolic_panel.pdf
```

### The safety gate demonstration

```powershell
python scripts/demo_safety_gate.py
```

Feeds crafted unsafe drafts through the Safety Verification Agent and shows exactly what is blocked and why, plus prompt-injection detection and PII redaction. Strong material for the Responsible AI section of the mid evaluation.

### The web application

```powershell
python app.py
```

Then open http://127.0.0.1:8000

Check what is actually running:

```
http://127.0.0.1:8000/api/v1/health
```

Expected:

```json
{
  "status": "ok",
  "orchestrator": "langgraph",
  "retrieval_backend": "bm25-inmemory",
  "indexed_passages": 41,
  "mcp_transport": "stdio",
  "llm_configured": false,
  "max_upload_mb": 10
}
```

### The MCP server on its own

```powershell
python -m agents.retrieval.mcp_server --list-tools
```

Shows the tools the retrieval agent publishes over the Model Context Protocol. Useful evidence when an examiner asks what protocol the agents use.

### Tests

```powershell
python -m unittest discover -s tests -v
```

43 tests covering retrieval, the safety rules, the revise loop and the security controls.

### Verify every citation URL still resolves

```powershell
python scripts/verify_sources.py
```

Run this the day before the evaluation. A citation pointing at a dead link is worse than no citation.

---

## Demo-day sequence

Run these in order, with the terminal projected:

```powershell
python scripts/test_gemini.py                           # key and model are live
python -m agents.retrieval.mcp_server --list-tools      # the protocol exists
python scripts/run_pipeline.py sample_reports/sample_cbc_report.pdf
python scripts/demo_safety_gate.py                      # the guardrail works
type logs\audit.jsonl                                   # the audit trail
```

Then start `python app.py` and upload a report through the browser.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'medexplain'`**
You are not in the project root, or the virtual environment is not active. `cd D:\Y3S2\IRWA\IRWA-Project` and re-activate.

**`Knowledge base not found` / `no passages`**
Run `python scripts/build_index.py`.

**`ChromaDB unavailable ... Falling back to BM25 retrieval backend`**
Not an error. The embedding model could not be downloaded or loaded, so the system used the lexical backend instead. Everything still works. To force it and silence the warning, set `VECTOR_BACKEND=bm25` in `.env`.

**`No GEMINI_API_KEY configured. Using grounded fallback generator.`**
Not an error. Add a key to `.env` if you want Gemini-written explanations.

**`MCP stdio session failed ... Falling back to in-process transport`**
The MCP subprocess could not start. The same tool contract is still honoured in-process. To force one or the other, set `MCP_TRANSPORT=stdio` or `MCP_TRANSPORT=inprocess` in `.env`.

**`The API rejected this key` from `test_gemini.py`**
The key is wrong, expired, or from a different Google product. Get a Gemini API
key at https://aistudio.google.com/app/apikey. The system still runs without one.

**`gemini-3.6-flash is not in this key's model list`**
Run `python scripts/test_gemini.py --list` to see what the key can reach, then set
`GEMINI_MODEL` in `.env` to one of those.

**`This SDK cannot set thinking_level`**
Your `google-genai` is older than 2.0. Run `pip install -U google-genai`. The
model still works, it just uses its own default thinking level, which is slower.

**Clicking සිංහල says the result is no longer available**
The server was restarted after the report was analysed, or the result expired
(24 hours by default). The page now re-runs the analysis automatically and
translates, so this should be invisible. If you see it, just analyse the report
again. Auto-reload is off by default for this reason; set `RELOAD=1` in `.env`
only while developing.

**`error while attempting to bind on address ('127.0.0.1', 8000)` / port already in use**
An earlier `python app.py` is still running. Stop it:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen |
  Select-Object -ExpandProperty OwningProcess |
  ForEach-Object { Stop-Process -Id $_ -Force }
```

Then start it again. Or change the port on the last line of `app.py`.

**PowerShell will not run the activation script**
See step 1 — use `Set-ExecutionPolicy` or Command Prompt.
