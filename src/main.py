import gradio as gr

from sam.sam_tools import get_point_from_gradio_click, MODELS as sam_models, run_image_sam
from Masking.mask_editing import add_mask, trim_mask, clear_mask, grow_mask, shrink_mask

with gr.Blocks(title="Masker", css=".app { max-width: 100% !important; }") as app:
    with gr.Row():
        input_image = gr.Image()
        with gr.Column():
            sam_points = gr.State([])
            sam_labels = gr.State([])

            gr.Markdown("<h1 style='text-align: center'>Mask Generation</h1>")
            sam_model = gr.Dropdown(label="Model", choices=sam_models, interactive=True)
            with gr.Row():
                sam_point_type = gr.Radio(label="Point Type", value="include", choices=["include", "exclude"])
                clear_sam_points = gr.Button("Clear")
            sam_point_map = gr.Image()
            run_sam_btn = gr.Button("Generate Mask")

            clear_sam_points.click(lambda x: (x, [], []), inputs=[input_image], outputs=[sam_point_map, sam_points, sam_labels])
            sam_point_map.select(get_point_from_gradio_click, inputs=[sam_point_type, sam_points, sam_labels, input_image], outputs=[sam_points, sam_labels, sam_point_map])

    with gr.Row():
        with gr.Column():
            composite_image = gr.Sketchpad(label="Composite", layers=True)
            with gr.Row():
                add_mask_btn = gr.Button("+")
                trim_mask_btn = gr.Button("-")
                clear_mask_btn = gr.Button("C")

                shrink_mask_btn = gr.Button("Shrink")
                shrink_grow_mask_amount_slider = gr.Slider(show_label=False, minimum=1, maximum=100, step=1, min_width=768)
                grow_mask_btn = gr.Button("Grow")

        mask_image = gr.Image(label="Mask")
        
        add_mask_btn.click(add_mask, inputs=[input_image, composite_image, mask_image], outputs=[composite_image, mask_image])
        trim_mask_btn.click(trim_mask, inputs=[input_image, composite_image, mask_image], outputs=[composite_image, mask_image])
        clear_mask_btn.click(clear_mask, inputs=[input_image], outputs=[composite_image, mask_image])

        shrink_mask_btn.click(shrink_mask, inputs=[input_image, mask_image, shrink_grow_mask_amount_slider], outputs=[composite_image, mask_image])
        grow_mask_btn.click(grow_mask, inputs=[input_image, mask_image, shrink_grow_mask_amount_slider], outputs=[composite_image, mask_image])

        run_sam_btn.click(run_image_sam, inputs=[input_image, sam_model, sam_points, sam_labels], outputs=[composite_image, mask_image])

        input_image.upload(lambda x: (x, gr.update(value=x)), inputs=[input_image], outputs=[sam_point_map, composite_image])
app.launch()