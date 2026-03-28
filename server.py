import uuid
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from main import Memorex
from datamodels import EpisodeType
from pydantic import BaseModel

app = FastAPI(title="MemoRex API")

class EpisodeInput(BaseModel):
    episode_body: str
    name: str = "User Memory"
    source_description: str = "Web UI"

class SearchInput(BaseModel):
    query: str

# Create a singleton Memorex instance
memorex = None

@app.on_event("startup")
async def startup_event():
    global memorex
    memorex = Memorex()
    # Ensure indices are created on startup
    await memorex.driver.build_indices_and_constraints()

@app.post("/api/episodes")
async def add_episode(episode_data: EpisodeInput):
    new_uuid = uuid.uuid4().hex
    
    await memorex.add_episode(
        uuid=new_uuid,
        name=episode_data.name,
        episode_body=episode_data.episode_body,
        source_description=episode_data.source_description,
        reference_time=datetime.now(),
        source=EpisodeType.message,
        group_id="web_ui_group",
    )
    
    return {"status": "success", "message": "Episode added successfully", "uuid": new_uuid}

@app.get("/api/search")
async def search_graph(query: str):
    results = await memorex.search(query=query)
    
    # Serialize results to simple dictionary structure
    nodes_data = [
        {"uuid": n.uuid, "name": n.name, "summary": n.summary} 
        for n in results.nodes
    ]
    edges_data = [
        {"uuid": e.uuid, "name": e.name, "fact": e.fact, "source": e.source_node_uuid, "target": e.target_node_uuid}
        for e in results.edges
    ]
    episodes_data = [
        {"uuid": ep.uuid, "name": ep.name, "content": ep.content}
        for ep in results.episodes
    ]
    
    return {
        "nodes": nodes_data,
        "edges": edges_data,
        "episodes": episodes_data
    }

# Mount static directory to serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
