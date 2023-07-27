#!/usr/bin/env python3
import argparse
import torch
from clip_interrogator import Config, Interrogator, list_caption_models, list_clip_models

try:
    import gradio as gr
except ImportError:
    print("Gradio is not installed, please install it with 'pip install gradio'")
    exit(1)

parser = argparse.ArgumentParser()
parser.add_argument("--lowvram", action='store_true', help="Optimize settings for low VRAM")
parser.add_argument('-s', '--share', action='store_true', help='Create a public link')
args = parser.parse_args()

if not torch.cuda.is_available():
    print("CUDA is not available, using CPU. Warning: this will be very slow!")

config = Config(cache_path="cache")
if args.lowvram:
    config.apply_low_vram_defaults()
ci = Interrogator(config)

def image_analysis(image, clip_model_name):
    if clip_model_name != ci.config.clip_model_name:
        ci.config.clip_model_name = clip_model_name
        ci.load_clip_model()

    image = image.convert('RGB')
    image_features = ci.image_to_features(image)

    top_mediums = ci.mediums.rank(image_features, 10)
    #top_artists = ci.artists.rank(image_features, 0)
    #top_movements = ci.movements.rank(image_features, 2)
    #top_trendings = ci.trendings.rank(image_features, 0)
    top_flavors = ci.flavors.rank(image_features, 50)


    medium_ranks = {medium: sim for medium, sim in zip(top_mediums, ci.similarities(image_features, top_mediums))}
    #artist_ranks = {artist: sim for artist, sim in zip(top_artists, ci.similarities(image_features, top_artists))}
    #movement_ranks = {movement: sim for movement, sim in zip(top_movements, ci.similarities(image_features, top_movements))}
    #trending_ranks = {trending: sim for trending, sim in zip(top_trendings, ci.similarities(image_features, top_trendings))}
    # atribut to flavor_ranks the top 20 flavors (top_flavor) and the similarities between the image and the top 20 flavors
    # iterate over flavor and sim in zip(top_flavors, ci.similarities(image_features, top_flavors))
    # zip(top_flavors, ci.similarities(image_features, top_flavors)) returns a list of tuples (flavor, similarity)
    list_tuples_flavor_similarity = zip(top_flavors, ci.similarities(image_features, top_flavors))
    # this syntax is called list comprehension. It is a way to create a list from another list. 
    # In this case, we are creating a list of tuples (flavor, similarity) from the list of tuples returned by zip.
    # the syntax {flavor: sim for flavor, sim in list_tuples_flavor_similarity} is called dictionary comprehension. 
    # It is a way to create a dictionary from another list. 
    # In this case, we are creating a dictionary with the flavor as key and the similarity as value.
    flavor_ranks = {flavor: sim for flavor, sim in list_tuples_flavor_similarity}
    
    return medium_ranks, flavor_ranks

def image_to_prompt(image, mode, clip_model_name, blip_model_name):
    if blip_model_name != ci.config.caption_model_name:
        ci.config.caption_model_name = blip_model_name
        ci.load_caption_model()

    if clip_model_name != ci.config.clip_model_name:
        ci.config.clip_model_name = clip_model_name
        ci.load_clip_model()

    image = image.convert('RGB')
    if mode == 'best':
        return ci.interrogate(image)
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    elif mode == 'fast':
        return ci.interrogate_fast(image)
    elif mode == 'negative':
        return ci.interrogate_negative(image)

def prompt_tab():
    with gr.Column():
        with gr.Row():
            image = gr.Image(type='pil', label="Image")
            with gr.Column():
                mode = gr.Radio(['best', 'fast', 'classic', 'negative'], label='Mode', value='best')
                clip_model = gr.Dropdown(list_clip_models(), value=ci.config.clip_model_name, label='CLIP Model')
                blip_model = gr.Dropdown(list_caption_models(), value=ci.config.caption_model_name, label='Caption Model')
        prompt = gr.Textbox(label="Prompt")
    button = gr.Button("Generate prompt")
    button.click(image_to_prompt, inputs=[image, mode, clip_model, blip_model], outputs=prompt)

def analyze_tab():
    with gr.Column():
        with gr.Row():
            image = gr.Image(type='pil', label="Image")
            model = gr.Dropdown(list_clip_models(), value='ViT-L-14/openai', label='CLIP Model')
        with gr.Row():
            medium = gr.Label(label="Medium", num_top_classes=5)
            #artist = gr.Label(label="Artist", num_top_classes=5)        
            #movement = gr.Label(label="Movement", num_top_classes=5)
            #trending = gr.Label(label="Trending", num_top_classes=5)
            flavor = gr.Label(label="Breed", num_top_classes=5)
    button = gr.Button("Analyze")
    button.click(image_analysis, inputs=[image, model], outputs=[medium, flavor])

with gr.Blocks() as ui:
    gr.Markdown("# <center>🕵️‍♂️ CLIP Interrogator 🕵️‍♂️</center>")
    with gr.Tab("Prompt"):
        prompt_tab()
    with gr.Tab("Analyze"):
        analyze_tab()

ui.launch(show_api=False, debug=True, share=args.share)
