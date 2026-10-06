from modules.face_recognition_module import FaceRecognizer
import cv2

recognizer = FaceRecognizer()

print("COGNITO Face Recognition Test")
print("================================")
print("1. Register via webcam")
print("2. Register via photo")
print("3. Test recognition")
print("4. List registered users")

choice = input("\nEnter choice (1-4): ")

if choice == '1':
    name = input("Enter your name: ")
    recognizer.register_from_webcam(name)

elif choice == '2':
    name  = input("Enter your name: ")
    photo = input("Enter photo path: ")
    recognizer.register_from_photo(name, photo)

elif choice == '3':
    print("Opening webcam for recognition...")
    print("Press Q to quit")
    cap = cv2.VideoCapture(0)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Recognize every 30 frames
        if frame_count % 30 == 0:
            name, confidence = recognizer.recognize(frame)
            if name:
                info = recognizer.get_user_info(name)
                print(f"\nRecognized: {name} "
                      f"({confidence}% confidence)")
                print(f"Greeting: {info['greeting']}")
                print(f"Past sessions: {info['sessions']}")
            else:
                print("No match found...")

        # Display
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0,0), (w,50), (20,20,30), -1)
        cv2.putText(frame,
                    "COGNITO - Face Recognition Test",
                    (10,35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (150,150,255), 2)
        cv2.imshow('Face Recognition Test', frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

elif choice == '4':
    users = recognizer.get_all_users()
    if users:
        print(f"\nRegistered users ({len(users)}):")
        for u in users:
            print(f"  • {u}")
    else:
        print("No users registered yet!")