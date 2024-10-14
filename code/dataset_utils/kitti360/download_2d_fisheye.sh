#!/bin/bash

train_list=(
		"2013_05_28_drive_0000_sync" # Good. 											OK, 2K - 2K
        # "2013_05_28_drive_0002_sync" # Done, too green
	    "2013_05_28_drive_0003_sync" # Small, landscapes
		# "2013_05_28_drive_0004_sync" # Mix, city and landscape
		# "2013_05_28_drive_0005_sync" # Good, similar to 6
		"2013_05_28_drive_0006_sync" # Good, hilly city                               OK, 2K - 2K
		# "2013_05_28_drive_0007_sync" # Done, open field roads
		# "2013_05_28_drive_0008_sync" # Done, small-height, little too much green
		"2013_05_28_drive_0009_sync" # Good, case abbastanza dense mid-height
	    #"2013_05_28_drive_0010_sync" # Done
		 )
cam_list=("02" "03")

root_dir=KITTI-360
data_2d_dir=data_2d_raw

mkdir -p $root_dir
mkdir -p $root_dir/$data_2d_dir

cd $root_dir 

# fisheye images
for sequence in ${train_list[@]}; do
    for camera in ${cam_list[@]}; do 
	zip_file=${sequence}_image_${camera}.zip
        wget https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/data_2d_raw/${zip_file}
	unzip -d ${data_2d_dir} ${zip_file} 
	rm ${zip_file}
    done
done

# # timestamps
# zip_file=data_timestamps_fisheye.zip
# wget https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/data_2d_raw/${zip_file}
# unzip -d ${data_2d_dir} ${zip_file}
# rm $zip_file
