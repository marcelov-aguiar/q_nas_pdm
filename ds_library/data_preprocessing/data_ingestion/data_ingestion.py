from abc import ABC, abstractmethod

class DataIngestion:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def process(self) -> None:
        pass
    
    @abstractmethod
    def save_to_parquet(self) -> None:
        pass