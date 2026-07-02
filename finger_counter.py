import cv2
from cvzone.HandTrackingModule import HandDetector

# Kamera başlat
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# El tespit edici
detector = HandDetector(maxHands=1, detectionCon=0.8)

# Parmak uçlarının landmark indeksleri
tipIds = [4, 8, 12, 16, 20]

# Parmak sayısını hesaplayan fonksiyon
def countFingers(hand):
    lm = hand["lmList"]
    handType = hand["type"]

    fingers = []

    wrist = lm[0]
    pinky = lm[17]

    palmFront = wrist[0] < pinky[0]

    # El tipine ve avuç yönüne göre başparmak açık/kapalı kontrolü
    if handType == "Right":
        if palmFront:
            # Sağ el, avuç içi: başparmak ucu sağdaysa açık
            fingers.append(1 if lm[4][0] > lm[3][0] else 0)
        else:
            # Sağ el, el tersi: başparmak ucu soldaysa açık
            fingers.append(1 if lm[4][0] < lm[3][0] else 0)
    else:
        if palmFront:
            # Sol el, avuç içi: başparmak ucu soldaysa açık
            fingers.append(1 if lm[4][0] < lm[3][0] else 0)
        else:
            # Sol el, el tersi: başparmak ucu sağdaysa açık
            fingers.append(1 if lm[4][0] > lm[3][0] else 0)

    for tip in tipIds[1:]:
        if lm[tip][1] < lm[tip-2][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)


while True:
    # Kameradan kare oku
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    # El tespiti yap ve landmark'ları çiz
    hands, frame = detector.findHands(frame)

    finger_count = 0

    if hands:
        hand = hands[0]
        finger_count = countFingers(hand)

    # Parmak sayısını ekrana yaz
    cv2.putText(frame,
                f'Parmak: {finger_count}',
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                3)

    # Görüntüyü göster
    cv2.imshow("Parmak Sayaci", frame)

    # 'q' tuşuna basınca döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()