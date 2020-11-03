import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
from tkinter import *
from inception_resnet_v1 import *
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.models import model_from_json
from os import listdir
import time
import threading

global yuzOnay
global name_var
global entryString
global cameraDurum
global e1
global uyeIsmi


e1 = ""

#%% FaceNet

face_cascade = cv2.CascadeClassifier('haarsCode/frontalface.xml')


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(160, 160))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)   
    
    img = preprocess_input(img)
    return img




model = InceptionResNetV1()

model.load_weights('weights/facenet_weights.h5')




def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance



threshold = 21 




uye_fotograflari = "database/"

kullanicilar = dict()
i = 0
for file in listdir(uye_fotograflari):
	uyeIsmi, extension = file.split(".")
   
	img = preprocess_image('database/%s.jpg' % (uyeIsmi))
    
	representation = model.predict(img)[0,:]
   
	i += 1
	kullanicilar[uyeIsmi] = representation







#%%


width, height = 800, 600
cap = cv2.VideoCapture(0)
global lmain

root = tk.Tk()
root.title("Yüz Tanıma Güvenlik")
root.geometry("300x250")
global newWindow
newWindow = None

def fotograf_Cek():
    
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)  
      
    
    
    entryString =  name_var.get()
    if(entryString != ""):
        
        path = 'database/%s.jpg' % (entryString)
        out = cv2.imwrite(path, frame)
        durumLabel["text"] = "Sisteme kaydedildiniz."
    else:
        durumLabel["text"] = "İsim Soyisim Giriniz."
        

def camera_Onay():
    
    global cameraDurum
    if(cameraDurum == False):
         cameraDurum = True
         videoThread = threading.Thread(target=show_frame, args=())
         videoThread.start()
         cameraButon["state"] = "normal"
    else:
        cameraDurum = False
        videoThread = threading.Thread(target=show_frame, args=())
        videoThread.start()



def show_frame():
    if (cameraDurum == True):
               
        _, frame = cap.read()
        
        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_frame)   
      
    else:
        print("kapalı")
        
   
    

def yeniUyem():
    global cameraButon
    global name_var
    global entryString
    global newWindow
    global lmain
    global durumLabel
    newWindow = Toplevel(root)
    newWindow.title("Yeni Kullanıcı")
    newWindow.geometry("600x700")
    Button(newWindow, text = "Kamerayı Aç", command = camera_Onay).pack()
    
    

    durumLabel =  Label(newWindow,text ="")
    durumLabel.pack()
    
    cameraButon = Button(newWindow, text = "Fotoğraf Çek", state = "disabled", command = fotograf_Cek)
    
        
    cameraButon.pack()
    Label(newWindow, text = "").pack()
    Label(newWindow,  text = "İsim Soyisim").pack()
    
    name_var = tk.StringVar()
    e1 = Entry(newWindow, textvariable=name_var)
   
    
    
    e1.pack()
    lmain = Label(newWindow)
    lmain.pack() 
   
  
			    
    
    
def baslaTani():   
    global uyeIsmi
    global sayac    
    sayac = 0
    uyeIsmi = ""
    while(True):
        
        sayac += 1
        ret, img = cap.read()
        faces = face_cascade.detectMultiScale(img, 1.3, 5)

        for (x,y,w,h) in faces:
            if w > 130:
                cv2.rectangle(img, (x,y), (x+w,y+h), (67, 67, 67), 1) 

                detected_face = img[int(y):int(y+h), int(x):int(x+w)] 
                detected_face = cv2.resize(detected_face, (160, 160)) 

                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis = 0)               
                img_pixels /= 127.5
                img_pixels -= 1

                captured_representation = model.predict(img_pixels)[0,:]

                distances = []

                for i in kullanicilar:
                    uyeIsmi = i
                    source_representation = kullanicilar[i]

                    distance = findEuclideanDistance(captured_representation, source_representation)

                    print(uyeIsmi,": ",distance)
                    distances.append(distance)

                label_name = 'unknown'
                index = 0
                for i in kullanicilar:
                    uyeIsmi = i
                    if index == np.argmin(distances):
                        if distances[index] <= threshold:
                            print("detected: ",uyeIsmi)

                            
                            similarity = 100 + (20 - distance)
                            if similarity > 99.99: similarity = 95.99

                            label_name = "%s (%s%s)" % (uyeIsmi, str(round(similarity,2)), '%')

                            break

                    index = index + 1

                cv2.putText(img, label_name, (int(x+w+15), int(y-64)), cv2.FONT_HERSHEY_SIMPLEX, 1, (67,67,67), 2)

                #connect face and text
                cv2.line(img,(x+w, y-64),(x+w-25, y-64),(67,67,67),1)
                cv2.line(img,(int(x+w/2),y),(x+w-25,y-64),(67,67,67),1)

        uye["text"] = uyeIsmi
        cv2.imshow('img',img)
        
                
        if uyeIsmi != "" :
            tk.messagebox.showinfo(title="Merhaba", message="Giriş yaptınız")
            break

    
	
    cap.release()
    cv2.destroyAllWindows() 	
    
            
def UyeBilgileri():
    global uye
    
    newWindoww = Toplevel(root)
    newWindoww.title("Üye")
    newWindoww.geometry("600x400")    
    Label(newWindoww, text = "Yüzünüzü kameraya gösteriniz.", bg="red").pack()
    Button(newWindoww, text = "Giriş",  padx = 10, pady = 10 ,command = baslaTani).pack()
    uye =  Label(newWindoww,text ="", bg ="blue", fg= "white")
    uye.pack()
    

      
  
       
    
    
    

    
    
    
    
cameraDurum = False
yuzOnay = False

labelSpace2 = tk.Label(text="")
btnYeniUye = tk.Button(text = "Yeni Üye Kaydı", bg="green", fg = "white", font = "Arial 10" ,padx =20 , command = yeniUyem)
labelSpace = tk.Label(text="")
btnOduncKitap = tk.Button(text = "Giriş Yap", bg="blue", fg = "white", font = "Arial 10",  padx = 20, command = UyeBilgileri)



  
btnYeniUye.pack()
labelSpace.pack()
labelSpace2.pack()



root.mainloop()