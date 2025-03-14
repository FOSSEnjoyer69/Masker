import gradio as gr

from sam.sam_tools import get_point_from_gradio_click, MODELS as sam_models, run_image_sam

with gr.Blocks(title="Masker", css=".app { max-width: 100% !important; }") as app:
    with gr.Row():
        input_image = gr.Image()
        with gr.Column():
            sam_points = gr.State([])
            sam_labels = gr.State([])

            gr.Markdown("<h1 style='text-align: center'>Mask Generation</h1>")
            sam_model = gr.Dropdown(label="Model", choices=sam_models, interactive=True)
            sam_point_type = gr.Radio(label="Point Type", value="include", choices=["include", "exclude"])
            sam_point_map = gr.Image()
            run_sam_btn = gr.Button("Generate Mask")

            sam_point_map.select(get_point_from_gradio_click, inputs=[sam_point_type, sam_points, sam_labels, input_image], outputs=[sam_points, sam_labels, sam_point_map])

    with gr.Row():
        composite_image = gr.Image()
        mask_image = gr.Image()
        
        run_sam_btn.click(run_image_sam, inputs=[input_image, sam_model, sam_points, sam_labels], outputs=[composite_image, mask_image])

        input_image.upload(lambda x: x, inputs=[input_image], outputs=[sam_point_map])
app.launch()