import os
from pathlib import Path
from uuid import uuid4

from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from services.chunker import chunk_text
from services.embeddings import EmbeddingService
from services.llm import OllamaServiceError, generate_answer
from services.pdf_loader import extract_text_from_pdf
from services.vectordb import VectorDBService


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

ALLOWED_EXTENSIONS = {".pdf"}

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)

embedding_service = EmbeddingService()
vector_db = VectorDBService(str(VECTOR_DB_DIR))


def get_user_id() -> str:
    if "user_id" not in session:
        session["user_id"] = uuid4().hex
    return session["user_id"]


def allowed_pdf(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_error):
    flash("This PDF is too large. Please upload a PDF smaller than 100 MB.")
    return redirect(url_for("index"))


@app.route("/", methods=["GET", "POST"])
def index():
    user_id = get_user_id()

    if request.method == "POST":
        if "pdf" in request.files:
            return handle_upload(user_id)

        question = request.form.get("question", "").strip()
        if question:
            return handle_question(user_id, question)

        flash("Upload a PDF or enter a question.")
        return redirect(url_for("index"))

    return render_template(
        "index.html",
        document_name=session.get("document_name"),
        answer=session.get("answer"),
        question=session.get("question"),
        chat_history=session.get("chat_history", []),
    )


def handle_upload(user_id: str):
    uploaded_file = request.files.get("pdf")

    if not uploaded_file or uploaded_file.filename == "":
        flash("Choose a PDF file first.")
        return redirect(url_for("index"))

    if not allowed_pdf(uploaded_file.filename):
        flash("Only PDF files are supported.")
        return redirect(url_for("index"))

    document_id = uuid4().hex
    filename = secure_filename(uploaded_file.filename)
    stored_name = f"{document_id}_{filename}"
    pdf_path = UPLOAD_DIR / stored_name
    try:
        uploaded_file.save(pdf_path)
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
    except Exception:
        flash("Could not read this PDF. Try a text-based PDF instead of a scanned image PDF.")
        return redirect(url_for("index"))

    if not chunks:
        flash("No readable text was found in this PDF.")
        return redirect(url_for("index"))

    try:
        embeddings = embedding_service.embed_documents(chunks)
        vector_db.reset_user_collection(user_id)
        vector_db.add_document_chunks(
            user_id=user_id,
            document_id=document_id,
            chunks=chunks,
            embeddings=embeddings,
            source_name=filename,
        )
    except MemoryError:
        flash("This PDF is too large to index with the available memory. Try a smaller PDF.")
        return redirect(url_for("index"))
    except Exception:
        flash("Indexing failed for this PDF. Try a smaller PDF or restart the app and upload again.")
        return redirect(url_for("index"))

    session["document_id"] = document_id
    session["document_name"] = filename
    session["answer"] = None
    session["question"] = None
    session["chat_history"] = []
    flash(f"Indexed {len(chunks)} chunks from {filename}.")
    return redirect(url_for("index"))


def handle_question(user_id: str, question: str):
    if not session.get("document_id"):
        flash("Upload and index a PDF before asking a question.")
        return redirect(url_for("index"))

    question_embedding = embedding_service.embed_query(question)
    matches = vector_db.search(user_id=user_id, query_embedding=question_embedding, n_results=4)
    context_chunks = matches.get("documents", [[]])[0]

    if not context_chunks:
        flash("No matching context was found. Try uploading the PDF again.")
        return redirect(url_for("index"))

    context = "\n\n".join(context_chunks)
    try:
        answer = generate_answer(question=question, context=context)
    except OllamaServiceError as exc:
        flash(str(exc))
        return redirect(url_for("index"))

    history = session.get("chat_history", [])
    history.append({"question": question, "answer": answer})
    session["chat_history"] = history[-10:]
    session["question"] = question
    session["answer"] = answer

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
