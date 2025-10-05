from pydantic import BaseModel, Field, ConfigDict, TypeAdapter
from typing import Literal, Union, Annotated


class LipsyncMessage(BaseModel):
    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")


class LipsyncConfigMessage(LipsyncMessage):
    type: Literal["config"] = "config"
    video_fps: int
    audio_sample_rate: int


class LipsyncConfigAckMessage(LipsyncMessage):
    type: Literal["config_ack"] = "config_ack"


class LipsyncAudioInputMessage(LipsyncMessage):
    type: Literal["audio_input"] = "audio_input"
    audio_data: bytes


class LipsyncVideoInputMessage(LipsyncMessage):
    type: Literal["video_input"] = "video_input"
    video_frame: bytes


class LipsyncInterruptAudioMessage(LipsyncMessage):
    type: Literal["interrupt_audio"] = "interrupt_audio"


class LipsyncSyncedOutputMessage(LipsyncMessage):
    type: Literal["synced_result"] = "synced_result"
    video_frame: bytes
    audio_frame: bytes


class LipsyncErrorMessage(LipsyncMessage):
    type: Literal["error"] = "error"
    message: str


LipsyncClientMessage = Annotated[
    Union[LipsyncConfigMessage, LipsyncAudioInputMessage, LipsyncVideoInputMessage, LipsyncInterruptAudioMessage],
    Field(discriminator="type"),
]
LipsyncServerMessage = Annotated[
    Union[LipsyncConfigAckMessage, LipsyncSyncedOutputMessage, LipsyncErrorMessage], Field(discriminator="type")
]

LipsyncServerMessageAdapter = TypeAdapter(LipsyncServerMessage)
