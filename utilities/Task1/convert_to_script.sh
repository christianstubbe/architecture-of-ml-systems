OUTPUT_FILE="Task01.py"

jupyter nbconvert --to python "Task01_satellite_imgs.ipynb" --output $OUTPUT_FILE
black $OUTPUT_FILE
mv $OUTPUT_FILE "../Experiment_Scripts/Task01.py"

echo "done!"
