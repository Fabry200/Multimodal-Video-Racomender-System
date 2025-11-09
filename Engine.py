#prende un video e lo confronta con i top 3 scorer tramite la cosine similiarity
from embedd import VideoEmbedding
from user import User
import os
from numpy.linalg import norm
import numpy as np

class Engine:
    def __init__(self,user):
        self.user=user
    
    def video_search(self, dir):
        files={f_name:0 for f_name in os.listdir(dir) if f_name.endswith('.mp4')}
        system=VideoEmbedding()
        for file in files.keys():

            embedding=system.video_segmentation(f'{dir}/{file}')
           
            scores=[]
            for e,_,_ in self.user.embeddings[:3]:
                score=self.score(embedding,e)
                scores.append(score)
            weights = np.exp(scores) / np.sum(np.exp(scores))
            files[file]=np.dot(weights, scores)
            
        
        return sorted(files, key=lambda video: files[video], reverse=True)[0]

    
    def score(self, input_embedding, user_embedding):
        user_embedding = user_embedding / np.linalg.norm(user_embedding)
        input_embedding = input_embedding / np.linalg.norm(input_embedding)
        cosine = np.dot(input_embedding, user_embedding) / (norm(input_embedding) * norm(user_embedding))
        return cosine
        


