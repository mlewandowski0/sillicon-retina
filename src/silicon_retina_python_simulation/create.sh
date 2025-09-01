#ffmpeg -framerate 30 -i output/%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 output_video.mp4
ffmpeg -framerate 30 -i output/%04d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p -crf 18 output_video.mp4
