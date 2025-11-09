from embedd import VideoEmbedding
import random 
import numpy as np
from sklearn.preprocessing import StandardScaler
import chromadb

client = chromadb.Client()

class User:
    def __init__(self,id):
        self.id=id
        self.preferences= client.create_collection(name=f"User_{id}")
        self.i=[]
        self.viewing_time=[]
        self.embeddings=[]
        self.avg_viewing_time=0
        self.std=0
        self.system=VideoEmbedding()

      

    def watch(self,videopath,watch_time=None):
        embedding=self.system.video_segmentation(f'{videopath}')
        self.i.append(0)
       
        if watch_time is None:
            watch_time=random.random()*10

        self.viewing_time.append(watch_time)
        if len(self.viewing_time)==1:
            score=1
        else:
            score=self.score(watch_time)
        #print(score)
        
        self.embeddings.append([embedding,watch_time,score])
        self.embeddings=sorted(self.embeddings, key=lambda x: x[1], reverse=True)
        self.preferences.add(
                ids=[f"v{len(self.i)}"],                
                embeddings=[embedding], 
                metadatas=[{'user_score': score}]     
           
            )
        self.update()

    def update(self):
        if len(self.embeddings)>100: # if user watch 100 videos we delete the last 20 to make space
            self.embeddings=self.embeddings[20:]

        self.avg_viewing_time=np.mean(self.viewing_time)
        self.std=np.std(self.viewing_time)

    def score(self,watch_time):
        times = np.array(self.viewing_time)
        
        if watch_time>np.max(times): 
            for emb in self.embeddings:
                emb[2]=(emb[1]-np.min(times)) / (watch_time- np.min(times))
            normalized=1
        else:
            normalized= (watch_time-np.min(times)) / (np.max(times)- np.min(times))
        


        return normalized

    
if __name__ == '__main__' :
    G=User(2)
    G.watch('video/video1.mp4')
    G.watch('video/video1.mp4')
    G.watch('video/video2.mp4')
    G.watch('video/video1.mp4')
    

    print(G.std, G.avg_viewing_time)
