from Engine import Engine
from user import User
import numpy as np
from embedd import VideoEmbedding
class Menu:
    def __init__(self):
        self.users=[]
        self.engines=[]
    
    def main_menu(self):
        c=''
        while c!='exit':
            print('\n')
            print('Hello, select an option: \n1)Create a user \n2)Select a user\n3)View users\nType exit to quit')
            c=input(':')
            match c:
                case '1':
                    c1=''
                    while c1 != '2':
                        uid=input('Choose a Userid: ')
                        user=User(uid)
                        self.users.append((uid,user))
                        self.engines.append(Engine(user))
                        c1=input('Done, want to create more? 1)Yes 2)No: ')
                case '2':
                    c1=''
                    while c1 != 'exit':
                        i=0
                        for uid,_ in self.users:
                            print(f'\n{i}:',uid)
                            i+=1
                        c1=input('Choose a user, Type exit to quit: ')
                        if c1!='exit' and c1.isnumeric():
                            self.user_menu(int(c1))
                            break

                case '3':
                    for uid,_ in self.users:
                        print(uid)

    def user_menu(self, n):
        c=''
        user=self.users[n][1]
        name=self.users[n][0]
        while c!='exit':
            c=input(f'Selected user {name}:\n1)Watch a video \n2)Watch a racomended video \n3)Do queries on the users watched videos in natural language\n4)View user statistics\n5)Go to main menu \n6)Exit\n:')
            match c:
                case '1':
                    c1=''
                    while c1 != 'exit':
                        c1=input('Specify video path ex video/test/file.mp4 or type exit to quit: ')
                        time=input('\nSpecify time (seconds) or left this blank: ')
                        if time =="":
                            user.watch(c1)
                        else:
                            user.watch(c1,float(time))
                        c1=input('Watched! want to watch another video? 1)Yes 2)No')
                        if c1 == '2':
                            break
                case '2':
                    c1=''
                    while c1 != '2':
                        selected=self.engines[n].video_search(f'video/unseen')
                        print(f'System selected: {selected}')
                        time=input('\nSpecify time or left this blank: ')
                        if time =="":
                            user.watch('video/unseen/'+selected)
                        else:
                            user.watch('video/unseen/'+selected,float(time))
                        user.watch('video/unseen/'+selected)
                        c1=input('Watched! want to watch another video? 1)Yes 2)No')
                case '3':
                    input_query=''
                    while input_query != '2':
                        input_query=input('Write the query in natural language or press 2 to quit: ')
                        input_embedding=VideoEmbedding().embedding_extractor(input_query)
                        result=user.preferences.query(   
                            query_embeddings=[input_embedding]
                        )
                        print(result['ids'], result['metadatas'])
                case '4':
                    print(f'\nAVG View time: {user.avg_viewing_time}\nStandard deviation: {user.std}')
                case '5':
                    return self.main_menu()
                case '6':
                    return 0
                

g=Menu()
g.main_menu()