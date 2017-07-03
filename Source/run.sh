python recognition2.py --image $1
cd corner/
python salt_and_peper.py --image ../first_iteration.png
python shape_detection.py --image Seccond_iteration_salt_and_peper.png --original ../original_first.png
cd ..
