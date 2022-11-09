import csv
import os
from tqdm import tqdm

start_frame = 0
train_frames = []
test_frames = []

def write2csv(all_frames, file_name):
    # all frames is a nested array: containing multiple curr_frame
    # curr_frame = [participant_id, label, video_name, start_frame, stop_frame]
    header = ['participant_id', 'label', 'video_name', 'start_frame', 'stop_frame']
    csv_path = os.path.join('dataset','annotations_dataset','{}.csv'.format(file_name))
    with open(csv_path, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for frame in all_frames:
            writer.writerow(frame)


video_root = "annotations_dataset/saved_frames"

train_ids = ["s1", "s2", "s3", "s4", "s6"]
test_ids = ["s5"]

for participant_id in os.listdir(video_root):
    for label in os.listdir(video_root+'/'+participant_id):
        for video in tqdm(os.listdir(os.path.join(video_root, participant_id, label))):
            num_items = os.listdir(os.path.join(video_root, participant_id, label, video))
            end_frame = len(num_items)-2
            curr_frame = [participant_id, label, video, start_frame, end_frame]
            if participant_id in train_ids:
                train_frames.append(curr_frame)
            else:
                test_frames.append(curr_frame)

write2csv(train_frames, 'train')
write2csv(test_frames, 'test')


