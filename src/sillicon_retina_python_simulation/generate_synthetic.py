import numpy as np
import cv2
import os

def generate_synthetic_image(output_dir="synthetic_frames",
                               image_size=(128, 128),
                               num_frames=100):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_frames):
        x = np.linspace(0, 255, image_size[1], dtype=np.uint8)
        gradient = np.tile(x, (image_size[0], 1))
        shift = (i * 5) % image_size[1]
        shifted = np.roll(gradient, shift, axis=1)
        filename = f"frame_{i+1:04d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, shifted)

    print(f"Generated {num_frames} synthetic frames in '{output_dir}'.")

def generate_moving_circles(output_dir="synthetic_circles",
                             image_size=(128, 128),
                             num_frames=100,
                             circle_radius=10,
                             speed=2):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_frames):
        img = np.zeros(image_size, dtype=np.uint8)
        center_x = (i * speed) % (image_size[1] + 2 * circle_radius) - circle_radius
        center_y = image_size[0] // 2

        if 0 <= center_x < image_size[1]:
            cv2.circle(img, (center_x, center_y), circle_radius, 255, -1)

        filename = f"frame_{i+1:04d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, img)

    print(f"Saved {num_frames} moving circle images to '{output_dir}'.")


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
    generate_synthetic_image(output_dir="datasets/synthetic_shift")
    create_video_from_images(image_dir="datasets/synthetic_shift",
                             output_video="shift.mp4")

    generate_moving_circles(output_dir="datasets/synthetic_circles")
    create_video_from_images(image_dir="datasets/synthetic_circles",
                             output_video="moving_circle.mp4")

