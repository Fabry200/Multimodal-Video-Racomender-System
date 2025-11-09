import ollama
import torch
from ollama import chat
import cv2
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import pipeline
import transformers
import pytesseract
import librosa
from audio_extract import extract_audio
import warnings
import re 
import json

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*Special tokens have been added.*")

# moondream:1.8b for vision
# llama3.2:1b for language

class VideoEmbedding:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)


#_________Video segmentation_______________#
    def video_segmentation(self, path):
        vid = cv2.VideoCapture(path)
        if not vid.isOpened():
            print("Error: Unable to open video")
            return
        
        
        frame_dir = path.replace('.mp4','')

        audiodir = f"{path}".replace('.mp4','.wav')
        if not(os.path.exists(audiodir)):
            extract_audio(input_path=f"{path}", output_path=audiodir, output_format="wav")

        os.makedirs(frame_dir, exist_ok=True)

        frames = []
        count = 0
        step = 25
        success = True
        while success:
            success, image = vid.read()
            if success:
                cv2.imwrite(f'{frame_dir}/frame{count}.jpg', image)
                frames.append(f'{frame_dir}/frame{count}.jpg')
                count += 1
                vid.set(cv2.CAP_PROP_POS_FRAMES, count * step)
        vid.release()
        
        return self.coordinator_function(frames, audiodir)


    def coordinator_function(self, videoframes,audiodir):
        #print(audiodir)

        full_description = ""
        for i, frame in enumerate(videoframes):
            frame_desc = self.image_meaning_extraction(frame)
            #print(f"Frame {i} description: {frame_desc}")
            full_description += f"\nframe {i}: {frame_desc}"
        
        full_description = full_description.strip()

        audio_desc=self.audio_meaning_extraction(audiodir)
        video_desc=self.meaning_from_video(full_description+f'\n Audio Description: {audio_desc}')
        #print(video_desc)
        return self.embedding_extractor(video_desc)


#___________________meaning production____________#
    def image_meaning_extraction(self,img_path):
        image = Image.open(img_path)
        ocr_text = pytesseract.image_to_string(image)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_length=100, num_beams=5, early_stopping=True)
        caption = self.processor.decode(output[0], skip_special_tokens=True)

        if len(ocr_text)==0:
            return caption
        else:
            return caption+' OCR Text: '+ocr_text.replace('\n',' ').strip()
        
   
    def audio_meaning_extraction(self,path):
        pipe = pipeline(model="openai/whisper-tiny", task="automatic-speech-recognition")

     
        result = pipe(path)

        return result['text']

    def meaning_from_video(self, frame_description):
        prompt1=f"""
                    Determine the overall meaning of the video based on the following frame descriptions:

                    {frame_description}

                    Summarize the main actions and key elements concisely.
                """


        response = chat(
            model='llama3.2:1b',
            
            messages=[{
                'role': 'user',
                'content': prompt1
            }]
        )

        return response.message.content


#___________________embedding production____________#
    def embedding_extractor(self, description):
        batch = ollama.embed(model='embeddinggemma', input=[description])
        embedding = batch["embeddings"][0]  
        return embedding


'''
def tags_cleaning(raw_output):
    metadata = json.loads(raw_output)
'''

# Usage
if __name__ == "__main__":
    video_analyzer = VideoEmbedding()
    g=video_analyzer.video_segmentation('video/video2.mp4')
