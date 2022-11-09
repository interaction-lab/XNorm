import os
from tkinter.messagebox import askokcancel
import cv2
from pydub import AudioSegment
from tqdm import tqdm
import csv


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


def buildRGBFrames(video_root, participant_id, label, video):
    # print("BuildRGBFrames")
    video_path = os.path.join(video_root, participant_id, label, video)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frame_idx = 0
    video_name = video[:-4]
    while success:
        frame_name = 'frame_'+str(frame_idx).zfill(10)+'.jpg'
        # print("video_name: {}".format(video_name))
        video_save_dir = os.path.join(video_root, "saved_frames", participant_id, label, video_name)
        if not os.path.isdir(video_save_dir):
            # print("Creating the directory: {}".format(video_save_dir))
            os.makedirs(video_save_dir)
        video_save_path = os.path.join(video_save_dir, frame_name)
        cv2.imwrite(video_save_path, image)
        # print("Read and write {} successfully".format(video_save_path))
        success, image = vidcap.read()
        frame_idx += 1
    # TODO: store path in CSV 
    curr_frame = [participant_id, label, video_name, 0, frame_idx-1]
    return curr_frame


def buildAudio(video_root, participant_id, label, video):
    src = os.path.join(video_root, participant_id, label, video)
    video_name = video[:-4]
    dst = os.path.join(video_root, "saved_frames", participant_id, label, video_name, "{}.wav".format(video_name))
    sound = AudioSegment.from_file(src, format="mp4")
    sound.export(dst, format="wav")
    # print("Success msg")


video_root = "annotations_dataset"
train_frames = []
test_frames = []
for participant_id in os.listdir(video_root):
    if participant_id != 's6':
        continue
    for label in os.listdir(video_root+'/'+participant_id):
        print("Processing Label {}".format(label))
        for video in tqdm(os.listdir(os.path.join(video_root, participant_id, label))):
            # skip some invalid videos
            if video[:2] == '._':
                continue
            curr_frame = buildRGBFrames(video_root, participant_id, label, video)
            if participant_id == 's5' or participant_id == 's6':
                test_frames.append(curr_frame)
            else: 
                train_frames.append(curr_frame)
            buildAudio(video_root, participant_id, label, video)            
write2csv(train_frames, 'train')
write2csv(test_frames, 'test')


