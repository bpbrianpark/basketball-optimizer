import cv2
import os
import customtkinter as ctk
import threading
import numpy as np
import time
from typing import Optional
from PIL import Image, ImageTk

class CTkImageDisplay(ctk.CTkLabel):
    """
    A reusable ctk widget to display OpenCV images.

    It takes OpenCV frames (which are numpy arrays) and displays them in the widget.
    """

    def __init__(self, master) -> None:
        self._textvariable = ctk.StringVar(master, "")
        super().__init__(
            master,
            textvariable=self._textvariable,
            image=None,
        )
    
    def update_frame(self, frame: np.ndarray) -> None:
        """
        Update the widget with a new OpenCV frame.

        Args:
            frame: OpenCV frame (numpy array)
        """

        # OpenCV uses BGR, but CTkinter uses RGB, so we gotta convert it lol
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(rgb_frame)

        self.ctk_image = ctk.CTkImage(
            light_image=pil_image,
            dark_image=pil_image,
            size=(pil_image.width, pil_image.height)
        )

        self.configure(image=self.ctk_image, text="")

        # Tkinter images will be thrown away and garbage collected if we dont keep a reference to it

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Basketball Optimizer")
        self.geometry("800x600")

        self.is_recording = False
        self.writer = None
        self.output_path = "temp_recording.mp4"

        self.cap: Optional[cv2.VideoCapture] = None
        self.fps = 20.0

        self.recording_lock = threading.Lock()

        # This is for threading
        # Why do we need threading, you may ask? Because we want to display the video in a separate thread from the video capture thread.
        self.running = False
        self.video_thread: Optional[threading.Thread] = None

        self._create_ui()

        self.bind('<KeyPress-r>', lambda e: self.start_recording())
        self.bind('<KeyPress-s>', lambda e: self.stop_recording())
        self.bind('<KeyPress-q>', lambda e: self.quit_app())

        self.focus_set()

    def _create_ui(self) -> None:
        """ Create the UI """

        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.pack(side="bottom", fill="both", expand=False, padx=10, pady=10)

        self.record_button = ctk.CTkButton(
            self.controls_frame,
            text="Record (R)",
            command=self.start_recording,
            fg_color="red",
            hover_color="darkred",
            width=50,
        )
        self.record_button.pack(side="left", pady=10, padx=10)

        self.stop_button = ctk.CTkButton(
            self.controls_frame,
            text="Stop (S)",
            command=self.stop_recording,
            fg_color="gray",
            hover_color="darkgray",
            state="disabled",
            width=50,
        )
        self.stop_button.pack(side="right", pady=10, padx=10)

        self.image_frame = ctk.CTkFrame(self)
        self.image_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        self.image_display = CTkImageDisplay(self.image_frame)
        self.image_display.pack(fill="both", expand=True, padx=10, pady=10)

        self.feedback_label = ctk.CTkLabel(
            self.image_frame,
            text="Shot feedback will appear here.",
            font=ctk.CTkFont(size=14),
            fg_color="transparent",
            text_color="white",
            anchor="w",
        )

    def start_recording(self) -> None:
        # Use lock to prevent race conditions
        with self.recording_lock:
            if self.is_recording:
                return
            self.is_recording = True

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            self.writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                (self.frame_width, self.frame_height),
            )

        self.record_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

        print("Started Recording ...")

    def stop_recording(self) -> None:
        """ Stop the video recording. """
        # Use lock to safely stop recording
        with self.recording_lock:
            if not self.is_recording:
                return
            
            self.is_recording = False
            
            if self.writer is not None:
                self.writer.release()
                self.writer = None
        
        # Small delay to ensure writer is fully released before next recording
        # This prevents file handle conflicts
        time.sleep(0.1) 
        
        self.record_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        print("Stopped Recording ...")
    
    def _video_loop(self):
        """Video capture runs in a separate thread to avoid blocking the main thread. """
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Webcam capture failed to open.")
            self.running = False
            return
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                print("Just run into an error capture video frame")
                break
            
            with self.recording_lock:
                if self.is_recording and self.writer is not None:
                    try:
                        self.writer.write(frame)
                    except Exception as e:
                        print(f"Error writing frame: {e}")
                        self.is_recording = False
                        if self.writer is not None:
                            self.writer.release()
                            self.writer = None

            self.after(0, lambda f=frame: self.image_display.update_frame(f))

            # Delay to control frame rate
            threading.Event().wait(1.0 / self.fps)
        
        if self.cap is not None:
            self.cap.release()
        with self.recording_lock:
            if self.writer is not None:
                self.writer.release()
                self.writer = None
        
    def start_video(self):
        if self.running:
            return
        
        self.running = True
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()

    def quit_app(self):
        self.running = False
        if self.video_thread is not None:
            self.video_thread.join(timeout=1.0)
        
        if self.is_recording:
            self.stop_recording()

        self.destroy()

    def on_closing(self):
        self.quit_app()

def main():
    app = App()
    app.start_video()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
    
if __name__ == "__main__":
    main()