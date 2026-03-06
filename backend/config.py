from pydantic import BaseModel

class Settings(BaseModel):
    app_title: str = "Tensor Network Lab"
    cors_allow_origins: list[str] = ["*"]  # fine for local dev

settings = Settings()