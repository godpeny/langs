import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from langchain_core.messages import HumanMessage

from langserve import APIHandler

from app.graph import graph
from app.utils import extract_content_and_urls
from app.embed import embed, save_embed
from app.db import select_all

"""
    Fast API with langserve
"""
app = FastAPI(
    title="Langs",
    version="0.0.0",
    description="Server for Lang-* components combined.",
)

api_handler = APIHandler(graph, path="/api/v1")


@app.post("/api/v1/invoke", include_in_schema=False)
async def request_invoke(request: Request) -> JSONResponse:
    """Handle a request."""
    # The API Handler validates the parts of the request
    # that are used by the runnnable (e.g., input, config fields)
    body = await request.json()
    msg = body["input"]
    events = graph.stream(
        {
            "messages": [
                HumanMessage(
                    content=msg
                )
            ],
        }
    )

    last_event = None
    for event in events:
        print("!")
        print(event)
        last_event = event

    response = extract_content_and_urls(last_event)
    return JSONResponse(content=response)


@app.post("/api/v1/embed", include_in_schema=False)
async def request_embed(request: Request) -> JSONResponse:
    """Handle a request."""
    # The API Handler validates the parts of the request
    # that are used by the runnnable (e.g., input, config fields)
    body = await request.json()

    dir = "app/data"
    # check body["condition"] exists
    if "condition" in body:
        file_list = [f for f in os.listdir(dir) if f.endswith('.txt') and body["condition"] in f]
    else:
        file_list = [f for f in os.listdir(dir) if f.endswith('.txt')]

    # save embedding using save_embed function
    for file in file_list:
        vectorstore = embed(file)
        save_embed(file, vectorstore)

    return JSONResponse(content=file_list)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
