# setup
Install segment anything \ 
Run data/sample_video.py to obtain frames according to world timestamps.
```
python sample_video.py --video path/to/input.mp4 --timestamps path/to/world_timestamps_unix.npy --output path/to/output_dir --start 360 --end 660
# For large scale data implementation, save data into h5 file
python 00sample_video.py --video ../data/BF002/world.mp4  --timestamps ../data/BF002/world_timestamps_unix.npy --output ../data/BF002 --start 405 --end 1065 --h5 frames_01.h5 --no_image --frame_interval 3
python 00sample_video.py --video ../data/BF004/world.mp4  --timestamps ../data/BF004/world_timestamps_unix.npy --output ../data/BF004 --start 2280 --end 3180 --h5 frames_02.h5 --no_image --frame_interval 3
```

Test mode in `sample_video.py` enables sample first 10 frames for debug purpose \

```
# Overlay fixation on frames 
python frames_fixation.py --frames ./pilot2_sampled_frames/ --overlays ../output/fixation_overlays --fixation /home/houhao/workspace/VINS-Mono/dataset/Pilot2/fixations.csv
```

# 0. Preparetion fixation from neon player


# 1. Run segmentation supported by SAM
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
For large scale implementation:
```
python 01sam_by_gaze.py --frames ../data/pilot2/frames/frames.h5 --checkpoint ../model/sam_vit_b_01ec64.pth --masks ../data/pilot2/masks --gaze ../data/pilot2/gaze.csv 

# verify the overlay results. Randomly sample 10 images to check
python 01masks_sam_verify.py
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
For large scale implementation
```
python 02openclip_segment_classifier.py --frame_h5 ../data/pilot2/frames/frames.h5 --mask_h5 .
./data/pilot2/masks/masks.h5 --output ../data/pilot2/clip_out/ --prompt_file ../config/clip_prompts_walking.txt --json_output ../data/pilot2/clip_out/clip_results.json --fixation ../data/pilot2/fixations.csv --confidence_thr 0.5 --vote_fixation
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

# Behaviors recognition from eye goggles data
```
python behaviors_gaze.py --video pilot2/world.mp4 --gaze pilot2/gaze.csv --imu pilot2/imu.csv --timestamps pilot2/world_timestamps_unix.npy --output pilot/overlay.mp4 --info pilot2/info.json --export_csv pilot2/gaze_imu_behaviors.csv
```

# Environmental Metrics, sky and greenary ratio

# Mask2Former + fixation
1. Mask2Former to segment the video frames. 
```
python 03seg_mask2former.py --h5_path /home/houhao/workspace/EyeTrackingSam/data/pilot2/frames/frames.h5 --output_h5_path /home/houhao/workspace/EyeTrackingSam/data/pilot2/mask2former_semantic_maps.h5
```
2. Given a fixation, we know the start and end frames for the fixation. Then voting for objects this fixation corresponds. 
```
python 03seg_fixation.py --semantic_h5 ../data/BF002/mask2former_semantic_maps_01.h5 --fixation ../data/BF002/fixations.csv --id2label ./id2label.txt --output_csv ../data/BF002/seg_fixation_01.csv
```

Verify the running results: 
```
python 00verify_h5.py
```

# Sky and greenery ratio
python 05sky_greenary_ratio.py --semantic_h5 /home/houhao/workspace/EyeTrackingSam/data/BF002/mask2former_semantic_maps_01.h5 --id2label ./id2label.txt --output_dir /home/houhao/workspace/EyeTrackingSam/data/BF002/sky_greenery_demo --frame_h5 /home/houhao/workspace/EyeTrackingSam/data/BF002/frames_01.h5