from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    raw_path_fd001_train: str
    raw_path_fd001_test: str
    processed_path_fd001_train: str
    processed_path_fd001_test: str

    class Config:
        env_file = ".env"

if __name__ == "__main__":
    settings = Settings()

    # Acessar os valores
    print(settings.raw_path_fd001_train)