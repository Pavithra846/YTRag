import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
import logging
from pydantic import BaseModel, Field
from src.chat_service import ChatService

app = FastAPI()
chat_service = ChatService()
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

class CreateChatResponse(BaseModel):
    chat_id : str  # UUID string

class UploadDocumentResponse(BaseModel):
    chat_id: str
    message: str                 # e.g. "Successfully indexed N new files." / "No new files to upload."
    indexed_files: int | None = None

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)

class Source(BaseModel):
    file_name: str | None = None
    page: int | str | None = None  # match actual metadata type emitted by loaders/store

class AskResponse(BaseModel):
    chat_id: str
    question: str
    answer: str
    sources: list[Source]


@app.post("/chats", response_model=CreateChatResponse, status_code=201)
def create_chat()-> CreateChatResponse:
    chat_id = chat_service.create_chat()
    return CreateChatResponse(chat_id=chat_id)

@app.post("/chats/{chat_id}/documents", response_model=UploadDocumentResponse)
async def upload_documents(chat_id: str, files: list[UploadFile] = File(...)) -> UploadDocumentResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            temp_files: list[tuple[Path, str]] = []
            counter = 1
            for idx, upload in enumerate(files):
                original_filename = Path(upload.filename or "unknown").name
                file_temp_dir = temp_dir_path / str(idx)
                file_temp_dir.mkdir(parents=True, exist_ok=True)

                destination = file_temp_dir / original_filename
                with destination.open("wb") as buffer:
                    shutil.copyfileobj(upload.file, buffer)

                temp_files.append((destination, original_filename))
            message = chat_service.upload_documents(chat_id, temp_files)  # <-- main call
            return UploadDocumentResponse(chat_id=chat_id, message=message)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    

@app.post("/chats/{chat_id}/ask", response_model = AskResponse)
def Ask_question(chat_id: str, body:AskRequest)-> AskResponse:
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        response = chat_service.ask(chat_id, question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileExistsError as e:
        raise HTTPException(status_code=409,detail=str(e))
    except RuntimeError as e:
        # e.g. missing GROQ_API_KEY in RAGSearch init
        raise HTTPException(status_code=503, detail="Answer generation unavailable")
    except Exception:
        logger.exception("Unexpected error while processing /ask")

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your question."
        )
    return AskResponse(chat_id=chat_id,
                       question=question,
                       answer=response.get("answer", ""),
                       sources=response.get("sources", []))
