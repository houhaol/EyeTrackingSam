# setup
Install segment anything \ 
Run data/sample_video.py to obtain frames according to world timestamps.
`python sample_video.py --video path/to/video --timestamps path/to/world_timestamps_unix.npy --output ./frames_test --test`

Test mode in `sample_video.py` enables sample first 10 frames for debug purpose \

Run sam_by_gaze.py to load SAM model, conduct segmentation given the prompt by gaze. Output is overlay and masks images
```python sam_by_gaze.py \
  --gaze /home/houhao/workspace/VINS-Mono/dataset/Pilot2/ \
  --frames data/frames_test/ \
  --checkpoint model/sam_vit_h_4b8939.pth \
  --masks output/output_masks \
  --overlays output/output_overlays
```
Option provided for loading gaze or fixations


