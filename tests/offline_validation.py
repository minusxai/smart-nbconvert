"""Deterministic dependency integration checks; no external services are contacted."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
import tempfile
import nbformat
from smart_nbconvert import get_notebook_state, get_chat_messages, replace_image_links, get_oai_client

notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("Example"), nbformat.v4.new_code_cell("1", outputs=[nbformat.v4.new_output("display_data", data={"image/png": "aGVsbG8="})])])
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "example.ipynb"
    nbformat.write(notebook, path)
    state = get_notebook_state(path)
assert len(state["cells"]) == 2
assert state["cells"][1]["outputs"][0]["data"] == {"image_idx": 1}
assert len(state["images"]) == 1
messages = get_chat_messages(state, "Summarize")
assert messages[0]["role"] == "system"
assert messages[1]["content"][-1] == state["images"][0]
assert "data:image/jpeg;base64,aGVsbG8=" in replace_image_links("<image_idx>1</image_idx>", state["images"])
os.environ["OPENAI_API_KEY"] = "synthetic-test-key"
client = get_oai_client()
assert client.api_key == "synthetic-test-key"
client.close()
print("Notebook image extraction, message construction, replacement and SDK construction passed")
