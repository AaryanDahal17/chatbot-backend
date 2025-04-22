from fastapi import FastAPI
from pydantic import BaseModel
from rag_chain import chain

app = FastAPI()

class Query(BaseModel):
    query: str

@app.post("/query")
async def process_query(query: Query):
    response = chain.invoke(query.query)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 