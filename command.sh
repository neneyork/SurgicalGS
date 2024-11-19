python train.py -s /root/autodl-tmp/pushing --expname endonerf/pushing_depth_smooth --configs arguments/endonerf/default.py 

python render.py --model_path output/endonerf/pushing_depth_smooth --skip_train --skip_video --configs arguments/endonerf/default.py

python metrics.py --model_path output/endonerf/pushing_depth_smooth


python train.py -s /root/autodl-tmp/cutting --expname endonerf/cutting_depth_smooth --configs arguments/endonerf/default.py 

python render.py --model_path output/endonerf/cutting_depth_smooth --skip_train  --skip_video --configs arguments/endonerf/default.py

python metrics.py --model_path output/endonerf/cutting_depth_smooth


python train.py -s /root/autodl-tmp/StereoMIS --expname StereoMIS --configs arguments/endonerf/default.py 

python render.py --model_path output/StereoMIS --skip_train  --skip_video --configs arguments/endonerf/default.py

python metrics.py --model_path output/StereoMIS


##############################################################

python train.py -s /root/autodl-tmp/pushing --expname endonerf/pushing_test --configs arguments/endonerf/default.py 

python render.py --model_path output/endonerf/pushing_test --skip_train --skip_video --configs arguments/endonerf/default.py

python metrics.py --model_path output/endonerf/pushing_test


python train.py -s /root/autodl-tmp/cutting --expname endonerf/cutting_test --configs arguments/endonerf/default.py 

python render.py --model_path output/endonerf/cutting_test --skip_train --skip_video --configs arguments/endonerf/default.py

python metrics.py --model_path output/endonerf/cutting_test


python train.py -s /root/autodl-tmp/StereoMIS --expname StereoMIS_test --configs arguments/endonerf/default.py 

python render.py --model_path output/StereoMIS_test --skip_train  --skip_video --configs arguments/endonerf/default.py

python metrics.py --model_path output/StereoMIS_test
