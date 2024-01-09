import argparse
import sys
sys.path.append('./yolov7') 
sys.path.insert(0, './segment-anything')
import cv2
from PIL import Image
import numpy as np
import torch
from torch.nn import functional as F
import gradio as gr
from omegaconf import OmegaConf
from interactive_coloring.pipeline import InteractiveColoringPipeLine




if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Tobigs-19 Vision Conference")
    
    parser.add_argument('--config_path', default='models/interactivce-coloring.yaml')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
    
    # load config file
    config = OmegaConf.load(args.config_path)

    # Instantiate PipeLine from config
    pipe = InteractiveColoringPipeLine.from_config(config)

    # Load colorizer model checkpoints
    pipe.colorizer.load_checkpoint(ldm_path="./models/anything-v3-full.safetensors",
                                cldm_path="./models/control_v11p_sd15s2_lineart_anime.pth")

    with gr.Blocks() as demo:
        with torch.no_grad():            
            with gr.Row():
                # Col A: Input image
                with gr.Column(scale=1):
                    gr.HTML("<h3><center>Input</center></h3>")
                    input_img = gr.Image(label='Input image', show_label=False)
                            
                # Col B: Mask
                with gr.Column(scale=1):
                    gr.HTML("<h3><center>Input</center></h3>")
                    seg_img = gr.Image(label='Segmentation map', show_label=False, interactive=False)
                    seg_btn = gr.Button(value='Check Segmentation map')
                    
                #Col C: Inpainted image
                with gr.Column(scale=1):
                    gr.HTML("<h3><center>Output</center></h3>")
                    colored_img = gr.Image(label='Inpainted image', interactive=False)
                    text_prompt = gr.Textbox(label='Text prompt')    
                    color_btn = gr.Button(value='Inpaint')


            def on_seg_btn_clicked(input_img):
                pipe.set_lineart(input_img)
           
                segmentation_map = pipe._segmentation_map
                n_labels = len(np.unique(segmentation_map))
                map_color = np.zeros_like(segmentation_map)[:, :, None]
                map_color = np.concatenate((map_color, map_color, map_color), axis=2)
                for i in range(1, n_labels):
                    map_color[segmentation_map==i] = [int(j) for j in np.random.randint(0,255,3)]
                return map_color
  
            seg_btn.click(on_seg_btn_clicked,
                          inputs=input_img,
                          outputs=seg_img)
            
            def on_color_btn_clicked(input_img, text_prompt):
                pipe.set_lineart(input_img, text_prompt)
                pipe.update_color(input_img)
                return  pipe.robot_turn_coloring(n_target=1000)
            color_btn.click(on_color_btn_clicked,
                            inputs=[input_img, text_prompt],
                            outputs=colored_img)
        demo.launch()