# setup
Install segment anything \ 
Run data/sample_video.py to obtain frames according to world timestamps.
`python sample_video.py --video path/to/input.mp4 --timestamps path/to/world_timestamps_unix.npy --output path/to/output_dir --start 360 --end 660`

Test mode in `sample_video.py` enables sample first 10 frames for debug purpose \

# Run segmentation supported by SAM
Run sam_by_gaze.py to load SAM model, conduct segmentation given the prompt by gaze. Output is overlay and masks images
```python sam_by_gaze.py   \
    --gaze /home/houhao/workspace/VINS-Mono/dataset/Pilot2/gaze_positions.csv   \
    --frames data/frames_test/ \
    --checkpoint model/sam_vit_b_01ec64.pth   \
    --masks output/output_masks \
    --overlays output/output_overlays
```
Option provided for loading gaze or fixations
```
python sam_by_gaze.py   \
    --fixation /home/houhao/workspace/VINS-Mono/dataset/Pilot2/gaze_positions.csv   \
    --frames data/frames_test/ \
    --checkpoint model/sam_vit_b_01ec64.pth   \
    --masks output/output_masks \
    --overlays output/output_overlays
```

# Run classification supported by resnet trained on imagenet or CLIP
Run resnet_segment_classifier.py to classify the segments. 
```
python resnet_segment_classifier.py \
  --frames data/frames_test/ \
  --masks output/output_masks \
  --overlays output/output_overlays \
  --output output/labeled_overlays
```
Or
```
python clip_segment_classifier.py \
  --frames data/frames_test \
  --masks output/output_masks \
  --overlays output/output_overlays \
  --output output/clip_labeled_overlays \
  --prompt_file config/clip_prompts_walking.txt \
  --json_output clip_results.json \
  --fixation /home/houhao/workspace/VINS-Mono/dataset/Pilot2/fixations.csv
```

# CLIP prompts generation
Category: People & Mobility, Navigation & Urban Elements, Vehicles & Public Transport, Everyday Interaction, Public Furniture & Facilities, Green Spaces

# Output
Run data/merge_frames.py to merge frames to a video for demo.
```
python merge_frames.py \
  --frames clip_labeled_overlays \
  --output labeled_video.mp4 \
  --fps 30
```
