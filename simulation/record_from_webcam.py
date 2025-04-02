import numpy as np
import cv2
import os

def generate_video(output_dir="synthetic_frames",
                   image_size=(256, 256),
                   num_frames=100):

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit(1)

    for i in range(num_frames):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, image_size)

        filename = f"frame_{i + 1:04d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break


    print(f"Generated video from webcam with {num_frames} frames and saved in '{output_dir}'.")

def create_video_from_images(image_dir="synthetic_circles",
                              output_video="moving_circle.mp4",
                              fps=30):
    os.makedirs("videos", exist_ok=True)
    output_video = os.path.join("videos", output_video)

    images = sorted([
        img for img in os.listdir(image_dir)
        if img.endswith(".png")
    ])

    if not images:
        print("No images found in directory.")
        return

    first_frame = cv2.imread(os.path.join(image_dir, images[0]), cv2.IMREAD_GRAYSCALE)
    height, width = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

    out.release()
    print(f"Video saved to '{output_video}'.")


if __name__ == "__main__":
    generate_video(output_dir=os.path.join("datasets", "recording"), num_frames=300)
    create_video_from_images(image_dir=os.path.join("datasets", "recording"),
                             output_video="webcam.mp4")

