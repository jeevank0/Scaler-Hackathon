import os

from api.main import app


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "running"}


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
