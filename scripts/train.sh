export CUDA_VISIBLE_DEVICES=0
python train_mouth.py
python train_face.py --init_num 2000 --densify_grad_threshold 0.0005
python train_fuse.py --opacity_lr 0.001

python synthesize_fuse.py --eval
python metrics.py /root/TKG/test/ours_None/renders/out.mp4 /root/TKG/test/ours_None/gt/out.mp4