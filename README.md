Start the docker file with the following command.

docker run -it --rm --gpus all \
  -v /path/to/retouch:/app/data/retouch \
  -v $(pwd)/data/dataset_configs:/app/data/dataset_configs \
  tta-flow:latest

Make sure to start this container from the root of this project.
Make sure that the retouch data is saved as .npy files (insert file tree here to specify path structure - annotations are in annotations folder and paired with the volumes).
As long as your dataset_config/data.csv has the same structure as given here, you are not limited to using the RETOUCH data.
Note: The "mask" column will only be used for the downstream evaluation, specifically for calculating the Dice Similarity Coefficients. For training and testing the Flow Matching
networks, only the "volume" column will be read. So if you just want to train and test the Flow Matching network without downstream evaluation, leave out the "mask" column.


For inference, either a .npy array is provided with the reference trajectory (..provide a flag and a path for that. Make sure these things are checked and have a logic). Otherwise,
if the flag is set to False, generate a trajectory in the beginning of the training.