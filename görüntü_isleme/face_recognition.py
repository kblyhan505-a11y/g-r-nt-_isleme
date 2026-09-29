import cv2     #görüntü işleme ve kamera etkileşimi için openCv ktüphanesi

cap = cv2.VideoCapture(0)     #varsayılan kamerayı (index 0) açar

if not cap.isOpened():         #kamera açılamadıysa (bağlı değil/ başka bir uygulama kullanıyor vb.)
    raise RuntimeError("Kamera açılamadı.")   #anlamsız bir hata yerine net bir hata verir.
face_cascade= cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"     #yüz tespiti için Haar Cascade modeli
)
body_cascade= cv2.CascadeClassifier(
    cv2.data.haarcascades +"haarcascade_fullbody.xml"      #tam vucüt tespiti için
)
while True:
    ret, frame= cap.read()     #kameradan bir kare okur, okuma başarılıysa ret True olur
    if not ret: #kare okunamadıysa (kamera koptu/kapandı)
        print
        break                  #hata vermek yerine döngüden çık
    
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    #kareyi gri tona çevirir(haar cascade gir görüntü bekler)
    faces= face_cascade.detectMultiScale(gray, 1.3, 5) #gri karede yüzleri tespit eder
    bodies= body_cascade.detectMultiScale(gray, 1.1, 3) #gri karede tüm vücutları tespit eder
    
    for (x, y, width, height) in faces:        #tespit edilen her yüz için
        cv2.rectangle(frame, (x, y), (x+ width, y+ height), (255, 0, 0), 2)  #yüzün etrafına mavi dikdörtgen çizecek
    
    for (x, y, width, height) in bodies:        #tespit edilen her yüz için
            cv2.rectangle(frame, (x, y), (x+ width, y+ height), (0, 255, 0), 2)  #vücudun etrafına yeşil dikdörtgen çizecek
    cv2.imshow("Camera", frame)                #işlenmiş kareyi pencerede gösterir
    
    if cv2.waitKey(1)==ord("q"):            #q tuşuna basılırsa
        break                               #döngüden çıkar
    
cap.release()    #kamerayı serbest bırakır (döngü bittikten sonra her karede değil)
cv2.destroyAllWindows()