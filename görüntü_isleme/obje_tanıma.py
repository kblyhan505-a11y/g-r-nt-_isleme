import cv2 
import torch   #YOLO modelinin çalıştığı derin öğrenme kütüphanesi
from ultralytics import YOLO #nesne tanıma modeli kütüphanesi


device= torch.device('cuda' if torch.cuda.is_available() else 'cpu') #ekran kartı(gpu) varsa cuda yoksa cpu


def main(): #programın ana fonksiyonu
    #model
    model=  YOLO("yolo11n.pt") #önceden eğitilmiş modeli yüklüyoruz
    model.to(device) #modeli seçen cihaza taşıyoruz

    #video yakalama  indexli dfeault kamera
    cap=cv2.VideoCapture()
    if not cap.isOpened():
        print("Kamera Kapalı Durumda")
        return #fonksiyondnan çık devam etme
    while True: #kamera görüntüsünü sürekli işlemek için sonsuz döngü
        #çerceve yakalama
        ret, frame= cap.read() #kameradan bir kare (frame) okunuyor ret okumanın başarılı olup olmadığını belirtir
        if not ret:
            print("Hata: Çerçeve okunmadı")
            break #döngüden çık
        results=model(frame)  #okunan kareyi yolo modeline vererek nsne tespiti yapılıyor 
        #sonuçları işleme

        for result in results:  # modelin döndürdüğü her bir sonuç için 
            for box in result.boxes:   #o sonuçta ki tespit edilen kutu (box için)
                #koordinat 
                x1,y1,x2,y2= box.xyxy[0] #kutunun sol ,üst (x1 y1) sağ alt (x2 y2) koordinatları alıyor
                label_id= int(box.cls[0].item()) #tespit edilen nesnenin sınıd idsi alınıyor
                confidance=box.conf[0].item() #modelin bu tahmine olan güven skorıu (0-1 arası)
                class_label=model.names[label_id]  #modelden sınıf id sine karşşılık gelen isim alınıyor 
                #başlığı text yazacak
                label_text=f"{class_label}: {confidance:.2f}"
                cv2.rectangle(frame, (int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                #tespit edilen nesnenin etrafında yeşil diktörtgen çiziliyor
                cv2.putText(frame, label_text, (int(x1),int(y1)-5),cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,255,0),1)  #diktörtgenin üstüne (-5) etiket ve güven skorunu yazdırılıyor 

        #çerceveyi çizdirme
        cv2.imshow('çerçeve', frame)  #işlenmiş kareyi pencerede gösteriyoruz
        if cv2.waitKey(1)==ord("q"):
            break     # q tuşuna basıldıysa döngüden çık
    cap.release() #kamerayı serbst bırak 
    cv2.destroyAllWindows()  #tüm pencereleri kapatır 

if __name__=="__main__":   #dosya doğrudan çalıştırıldıysa 
    main()   #ana fonksiyonu çalıştır 

