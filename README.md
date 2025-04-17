# setup
Install segment anything \ 
Run data/sample_video.py to obtain frames according to world timestamps.
`python sample_video.py --video path/to/input.mp4 --timestamps path/to/world_timestamps_unix.npy --output path/to/output_dir --start 360 --end 660`

Test mode in `sample_video.py` enables sample first 10 frames for debug purpose \

```
# Overlay fixation on frames 
python frames_fixation.py --frames ./pilot2_sampled_frames/ --overlays ../output/fixation_overlays --fixation /home/houhao/workspace/VINS-Mono/dataset/Pilot2/fixations.csv
```

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
  --prompt_file config/clip_prompts_walking_material.txt \
  --json_output clip_results.json \
  --fixation /home/houhao/workspace/VINS-Mono/dataset/Pilot2/fixations.csv

python openclip_segment_classifier.py \
--frames data/frames_test \
--masks output/output_masks \
--overlays output/output_overlays \
--output output/clip_labeled_overlays \
--prompt_file config/clip_prompts_walking_material.txt \
--json_output clip_results.json \
--fixation /home/houhao/workspace/VINS-Mono/dataset/Pilot2/fixations.csv \
--confidence_thr 0.5
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


# GTOS finetuned clip to infer
```
# train
python finetune_gtos_clip.py \
  --data_root /home/houhao/workspace/EyeTrackingSam/benchmark/pytorch-material-classification/Tip-Adapter/data/gtos \
  --model ViT-B-32 \
  --pretrained laion2b_s34b_b79k \
  --batch_size 64 \
  --epochs 20 \
  --lr_encoder 1e-5 \
  --lr_head 1e-3 \
  --output_dir ./checkpoints/gtos_vitb32_reduced

# inference
python infer_clip_gtos_materials.py \
  --frames ./data/pilot2_sampled_frames \
  --masks ./output/output_masks \
  --output ./output/material_labeled_output \
  --checkpoint /home/houhao/workspace/EyeTrackingSam/benchmark/checkpoints/gtos_vitb32_history/best_model.pt \
  --classnames_txt /home/houhao/workspace/EyeTrackingSam/benchmark/checkpoints/gtos_vitb32/classnames.txt
```

# create diff images
```
python diff_img_create.py \
     --frame_dir /path/to/frames \
     --mask_dir /path/to/masks \
     --output_dir /path/to/output_diff
```