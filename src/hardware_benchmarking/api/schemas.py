from pydantic import BaseModel, Field
from typing import Optional


class CPUPredictionRequest(BaseModel):
    nb_of_Cores: float = Field(..., json_schema_extra={"example": 8.0})
    nb_of_Threads: float = Field(..., json_schema_extra={"example": 16.0})
    TDP: float = Field(..., json_schema_extra={"example": 65.0})
    Cache: float = Field(..., json_schema_extra={"example": 16.0})
    Lithography: float = Field(..., json_schema_extra={"example": 14.0})
    Max_Memory_Size: Optional[float] = Field(64.0, json_schema_extra={"example": 64.0})
    Max_Memory_Bandwidth: Optional[float] = Field(41.6, json_schema_extra={"example": 41.6})
    Max_nb_of_Memory_Channels: Optional[float] = Field(2.0, json_schema_extra={"example": 2.0})
    Recommended_Customer_Price: Optional[float] = Field(300.0, json_schema_extra={"example": 300.0})
    Vertical_Segment: Optional[str] = Field("Desktop", json_schema_extra={"example": "Desktop"})
    Product_Collection: Optional[str] = Field("8th Generation", json_schema_extra={"example": "8th Generation"})


class GPUPredictionRequest(BaseModel):
    Max_Power: float = Field(..., json_schema_extra={"example": 250.0})
    Memory: float = Field(..., json_schema_extra={"example": 8192.0})
    Memory_Bandwidth: float = Field(..., json_schema_extra={"example": 448.0})
    Memory_Bus: float = Field(..., json_schema_extra={"example": 256.0})
    Memory_Speed: float = Field(..., json_schema_extra={"example": 1750.0})
    Process: float = Field(..., json_schema_extra={"example": 16.0})
    ROPs: float = Field(..., json_schema_extra={"example": 64.0})
    TMUs: float = Field(..., json_schema_extra={"example": 160.0})
    Manufacturer: Optional[str] = Field("Nvidia", json_schema_extra={"example": "Nvidia"})
    Notebook_GPU: Optional[str] = Field("No", json_schema_extra={"example": "No"})
    Name: Optional[str] = Field("unknown", json_schema_extra={"example": "GeForce RTX 3080"})
