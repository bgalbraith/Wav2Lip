import os

import gradio as gr

from inference import main


def fn(checkpoint: str, video: str, audio: str):
    opts = {
        "checkpoint_path": f"checkpoints/{checkpoint}",
        "face": video,
        "audio": audio,
    }

    main(opts)
    return "results/result_voice.mp4"


checkpoint = gr.Dropdown(
    choices=[c for c in os.listdir("./checkpoints") if c[-4:] == ".pth"],
    label="checkpoint",
)

face = gr.File(label="face")
audio = gr.File(label="audio")

app = gr.Interface(
    fn=fn,
    inputs=[checkpoint, face, audio],
    outputs=["video"],
    allow_flagging="never",
)


if __name__ == "__main__":
    app.launch()
