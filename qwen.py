import random
import os
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000

from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.utils import FlexibleArgumentParser
import time

def run_qwen2_5_vl(question: str, modality: str):
    model_name = "Qwen/Qwen2.5-VL-72B-Instruct"
    llm = LLM(
        model=model_name,
        max_model_len=12000,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        tensor_parallel_size=8,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,  
    )
    while True:
        if modality == "image":
            placeholder = "<|image_pad|>"
        elif modality == "video":
            placeholder = "<|video_pad|>"

        prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
                f"{question}<|im_end|>\n"
                "<|im_start|>assistant\n")
        stop_token_ids = None
    return llm, prompt, stop_token_ids



def get_multi_modal_input(args):
    
    if args.modality == "image":
        image = Image.open("/data/inputs/images/image9.jpg").convert("RGB")
        # img_question="""
        # You are an AI specialized in recognizing and extracting text from images.Your mission is to analyze the image document and generate the result in QwenVL Document Parser HTML format using specified tags while maintaining user privacy and data integrity.
        # """
        if args.output_type =='html':
            img_question = """
             You are tasked with converting a scanned document image into a fully-renderable HTML document. The image may contain complicated tables, handwritten elements, stamps, logos, and other details. Your goal is to create an accurate HTML representation of the document using only the information visible in the image.

                Follow these steps to create the HTML document:

                1. Analyze the image content:
                - Identify the main sections of the document.
                - Note any tables, headings, paragraphs, lists, and special elements (stamps, logos, handwritten notes).
                - Pay attention to the overall layout and structure.

                2. Create the basic HTML structure:
                - Begin with the standard HTML5 doctype and `<html>`, `<head>`, and `<body>` tags.
                - Include a `<meta charset="UTF-8">` tag in the `<head>` section.
                - Add a `<title>` tag with an appropriate title based on the document content.

                3. Represent the document content:
                - Use appropriate HTML elements to structure the content (e.g., `<h1>`, `<h2>`, `<p>`, `<ul>`, `<ol>`, `<li>`).
                - For tables:
                    - Use `<table>`, `<tr>`, `<th>`, and `<td>` tags.
                    - Implement colspan and rowspan attributes if necessary.
                    - Ensure the table structure accurately reflects the original.
                - For handwritten elements:
                    - Use `<span>` tags with a class attribute (e.g., `<span class="handwritten">`).
                - For stamps and logos:
                    - Use `<div>` tags with appropriate class attributes (e.g., `<div class="stamp">`, `<div class="logo">`).
                    - Describe the visual appearance of stamps and logos using text content within these divs.

                4. Apply basic styling:
                - Create a `<style>` section in the `<head>` of the document.
                - Define basic styles for the document layout.
                - Add specific styles for handwritten text, stamps, and logos.
                - Use CSS to approximate the visual appearance of the original document (e.g., fonts, colors, spacing).

                5. Review and refine:
                - Ensure all content from the image is represented in the HTML.
                - Check that the structure and layout closely match the original document.
                - Verify that tables are correctly formatted and aligned.

                6. Output the final HTML:
                - Provide the complete HTML code, including doctype, `<html>`, `<head>`, and `<body>` tags.
                - Ensure all opening tags have corresponding closing tags.
                - Use proper indentation for readability.
                - Begin your response with `<!DOCTYPE html>` and end it with `</html>`.
            """
        elif args.output_type == 'doc_type':
            # img_question = """ 
            # You are an AI specialized in recognizing and extracting text from images. Your mission is to:
            #   1. Analyze the image document 
            #   2. Identify the document type from the following categories:
            #   - Circulars, Notifications, Orders, Memorandums, Gazettes, Acts/Rules, Policies, Resolutions, Guidelines, Licenses, Minutes of Meetings, Forms, Receipts, Bonds, Tenders,resumes,bio-data.
            #   3. provide the **document type** in plain text.
            # """
            img_question = """
            You are an AI specialized in analyzing and extracting text from images. Your task is to:

            Examine the provided image document and analyze its content.
            Identify its document type by selecting the most appropriate category from the following predefined list:
                Circulars
                Notifications
                Orders
                Memorandums
                Gazettes
                Acts/Rules
                Policies
                Resolutions
                Guidelines
                Licenses
                Minutes of Meetings
                Forms
                Receipts
                Bonds
                Tenders
                Resumes
                Bio-data
            🔹 Important: The document type must be selected strictly from the above categories. Provide the document type only in plain text without additional explanations.


            """
    
        return {
            "data": image,
            "question": img_question,
        }

    if args.modality == "video":
        video = VideoAsset(name="sample_demo_1.mp4",
                           num_frames=args.num_frames).np_ndarrays
        vid_question = "Why is this video funny?"

        return {
            "data": video,
            "question": vid_question,
        }

    msg = f"Modality {args.modality} is not supported."
    raise ValueError(msg)


def apply_image_repeat(folder_path, prompt, modality):
    inputs = []
    for file in os.listdir(folder_path):
        if file.endswith(('.png','.jpg','.jpeg')): 
            try:      
                image=Image.open(os.path.join(folder_path,file))
                inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {
                        modality: image
                    },
                    "image_name":file
                })
            except Exception as e:
                print(f"error loading{file}:{e}")
        return inputs    
    

def folders_pdf(folder_path, prompt, modality):
   
    inputs = []
    cnt=0
    for imgfolder in os.listdir(folder_path):
        cnt+=1
        for imgfile in os.listdir(os.path.join(folder_path,imgfolder)):
            name= os.path.join(folder_path,imgfolder,imgfile)
            if name.endswith(('.png','.jpg','.jpeg')):
                image=Image.open(os.path.join(folder_path,imgfolder,imgfile))
                inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {
                        modality: image
                    },
                    "image_name":imgfile
                })
      
    return inputs               


model_example_map = {
    "qwen2_5_vl": run_qwen2_5_vl,
}

def main(args):
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")
    modality = args.modality
    type_of_input = args.type_of_input
    input_folder_path = args.input_folder_path
    output_html_dir = args.output_html_dir
    output_type = args.output_type
    
    os.makedirs(output_html_dir,exist_ok = True)
    mm_input = get_multi_modal_input(args)
    data = mm_input["data"]
    question = mm_input["question"]
    llm, prompt, stop_token_ids = model_example_map[model](question, modality)
    sampling_params = SamplingParams(
        temperature=0.8,        
        top_p=0.9,              
        top_k=50,               
        max_tokens=20,         
        #n=2,
        stop_token_ids=stop_token_ids  
    )
    assert args.num_prompts > 0
    if type_of_input == 'images':
        inputs = apply_image_repeat(input_folder_path,prompt,modality)
    elif type_of_input == 'subfolder_images':
        inputs = folders_pdf(input_folder_path,prompt,modality) 
    start_time = time.time()
    #while True:
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    #import IPython;IPython.embed() 
    image_names = [entry["image_name"] for entry in inputs]

    elapsed_time = time.time() - start_time
    
    if args.output_type == 'html':
        cnt=0
        for o in outputs:
            generated_text = o.outputs[0].text
            print(generated_text)
    
            image_name =image_names[cnt][:-4]
            cnt+=1
            print(image_name)

            html_filename = f"{image_name}.html"
            html_filepath = os.path.join(output_html_dir, html_filename)
        
            with open(html_filepath, "w", encoding="utf-8") as f:
                f.write(generated_text)
        
            print(f"Saved HTML: {html_filepath}")
    elif args.output_type == 'doc_type':
        op_list={}
        cnt=0
        for o in outputs:
            generated_text = o.outputs[0].text
            print(generated_text)
    
            image_name =image_names[cnt][:-4]
            cnt+=1
            print(image_name)
            op_list[image_name]=generated_text
        with open('/data/results/qwen_doc_type_72b.txt',"w",encoding = 'utf-8') as f:
            for key,value in op_list.items():
                f.write(f"{key}:{value}\n")         
    print("-- generate time = {}".format(elapsed_time))

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models for text generation')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="qwen2_5_vl",
                        choices=model_example_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument('--num-prompts',
                        type=int,
                        default=2,
                        help='Number of prompts to run.')
    parser.add_argument('--modality',
                        type=str,
                        default="image",
                        choices=['image', 'video'],
                        help='Modality of the input.')
    parser.add_argument('--num-frames',
                        type=int,
                        default=16,
                        help='Number of frames to extract from the video.')

    parser.add_argument(
        '--image-repeat-prob',
        type=float,
        default=None,
        help='Simulates the hit-ratio for multi-modal preprocessor cache'
        ' (if enabled)')

    parser.add_argument(
        '--disable-mm-preprocessor-cache',
        action='store_true',
        help='If True, disables caching of multi-modal preprocessor/mapper.')

    parser.add_argument(
        '--time-generate',
        action='store_true',
        help='If True, then print the total generate() call time')
    parser.add_argument(
        '--type-of-input',type = str,default = 'subfolder_images', choices =['images','subfolder_images']
    )
    parser.add_argument('--input-folder-path',default="./qwen_testing",type = str)
    parser.add_argument('--output-html-dir',default="./outputs",type = str)
    parser.add_argument('--output-type',type=str,default="html" ,  choices =['html','doc_type'])
    args = parser.parse_args()
    main(args)
