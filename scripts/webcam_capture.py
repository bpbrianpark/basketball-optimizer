import cv2
import os

def main():
    
    cap = cv2.VideoCapture(0)

    if (not cap.isOpened()):
        print ("webcam capture failed to open.")
        return
    
    is_recording = False
    writer = None
    output_path = "temp_recording.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 20.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try: 
        while True:
            ret, frame = cap.read()

            if (not ret):
                print("Just run into an error capture video frame")
                return 
            
            if writer is not None and is_recording:
                writer.write(frame)

            cv2.imshow("Webcam capture", frame)
            
            key = cv2.waitKey(1) & 0XFF
            if (key == ord("r")):
                is_recording = True
                writer = cv2.VideoWriter(
                    output_path, 
                    fourcc, 
                    fps,
                    (frame_width, frame_height)
                )
                print("Started Recording ...")
            elif (key == ord("s")):
                if is_recording and writer is not None:
                    is_recording = False
                    writer.release()
                    writer = None
                    print("Stopped Recording ...")
                else:
                    print("Recording has not started yet.")
            elif (key == ord("q")):
                print("Quit webcam capture")
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            writer = None
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()